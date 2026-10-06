"""Cache Massive's active and delisted US stock reference records without logging credentials."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
from dotenv import load_dotenv

from ws import universe
from ws.security_master import MASSIVE_EXCHANGES, massive_stock_candidates
from ws.store.atomic import atomic_path, atomic_text

BASE = "https://api.massive.com/v3/reference/tickers"
LIMIT = 1000
MIN_REQUEST_INTERVAL_SECONDS = 12.2  # User's Massive limit: five requests per minute.


def reference_params(
    active: bool, ticker_type: str | None = None, exchange: str | None = None
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "market": "stocks",
        "active": str(active).lower(),
        "limit": LIMIT,
        "sort": "ticker",
        "order": "asc",
    }
    if ticker_type:
        params["type"] = ticker_type
    if exchange:
        params["exchange"] = exchange
    return params


def without_api_key(url: str) -> str:
    parts = urlsplit(url)
    query = [(key, value) for key, value in parse_qsl(parts.query) if key.lower() != "apikey"]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def canonical_digest(rows: list[dict[str, Any]]) -> str:
    lines = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    return hashlib.sha256(lines.encode()).hexdigest()


class RateLimitedReferenceClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.client = httpx.Client(timeout=45)
        self.last_request: float | None = None

    def get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        for attempt in range(4):
            wait = (
                MIN_REQUEST_INTERVAL_SECONDS
                if self.last_request is None
                else MIN_REQUEST_INTERVAL_SECONDS - (time.monotonic() - self.last_request)
            )
            if wait > 0:
                time.sleep(wait)
            self.last_request = time.monotonic()
            try:
                response = self.client.get(url, params={**(params or {}), "apiKey": self.api_key})
                if response.status_code == 429:
                    delay = float(response.headers.get("Retry-After", "60"))
                    time.sleep(max(delay, MIN_REQUEST_INTERVAL_SECONDS))
                    continue
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500 or attempt == 3:
                    raise RuntimeError(
                        f"Massive reference request failed with HTTP {exc.response.status_code}"
                    ) from None
            except httpx.TransportError:
                if attempt == 3:
                    raise RuntimeError("Massive reference request failed after retries") from None
            time.sleep(max(2**attempt, MIN_REQUEST_INTERVAL_SECONDS))
        raise RuntimeError("Massive reference request exhausted retries")


def fetch_status(
    client: RateLimitedReferenceClient,
    root: Path,
    active: bool,
    ticker_type: str | None = None,
    exchange: str | None = None,
) -> list[dict[str, Any]]:
    status = "active" if active else "inactive"
    filter_suffix = "".join(f"-{value}" for value in (ticker_type, exchange) if value)
    status_dir = root / f"{status}{filter_suffix}"
    status_dir.mkdir(parents=True, exist_ok=True)
    state_path = root / f"massive-reference-{status}{filter_suffix}-checkpoint.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state.get("complete"):
        rows: list[dict[str, Any]] = []
        for path in sorted(status_dir.glob("page-*.jsonl.gz")):
            with gzip.open(path, "rt", encoding="utf-8") as source:
                rows.extend(json.loads(line) for line in source)
        return rows
    next_url = state.get("next_url")
    page_index = int(state.get("page_index", 0))
    params: dict[str, Any] | None = None
    if not next_url:
        next_url = BASE
        params = reference_params(active, ticker_type, exchange)
    while next_url:
        payload = client.get(next_url, params)
        params = None
        if payload.get("status") not in {"OK", "DELAYED"}:
            raise RuntimeError(f"Massive reference response status was {payload.get('status')!r}")
        page_rows = payload.get("results") or []
        write_jsonl_gzip(status_dir / f"page-{page_index + 1:05d}.jsonl.gz", page_rows)
        next_url = without_api_key(payload["next_url"]) if payload.get("next_url") else None
        page_index += 1
        state = {
            "status": status,
            "page_index": page_index,
            "next_url": next_url,
            "complete": next_url is None,
        }
        atomic_text(state_path, json.dumps(state, separators=(",", ":")) + "\n")
        print(
            f"Massive {status}{filter_suffix} reference: page {page_index}, {len(page_rows)} rows this page",
            flush=True,
        )
    all_rows: list[dict[str, Any]] = []
    for path in sorted(status_dir.glob("page-*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as source:
            all_rows.extend(json.loads(line) for line in source)
    return all_rows


def write_jsonl_gzip(path: Path, rows: list[dict[str, Any]]) -> None:
    with atomic_path(path) as temporary, gzip.open(temporary, "wt", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> None:
    load_dotenv(".env")
    api_key = os.environ.get("MASSIVE_API_KEY")
    if not api_key:
        raise SystemExit("MASSIVE_API_KEY is missing; key value was not read from the command line")
    root = Path("data/universe/massive-reference-2026-10-05")
    root.mkdir(parents=True, exist_ok=True)
    client = RateLimitedReferenceClient(api_key)
    # Alpaca's saved asset snapshot supplies current active names. The costly Massive
    # crawl is narrowed to inactive common stocks, which are the survivorship gap here.
    inactive = [
        row
        for exchange in sorted(MASSIVE_EXCHANGES)
        for row in fetch_status(client, root, active=False, ticker_type="CS", exchange=exchange)
    ]
    massive_candidates = massive_stock_candidates(inactive)
    alpaca_assets_path = sorted(Path("data/universe").glob("alpaca_assets_*.jsonl"))[-1]
    alpaca_assets = [json.loads(line) for line in alpaca_assets_path.read_text().splitlines() if line.strip()]
    alpaca_candidates = universe.listed_stock_candidates(alpaca_assets)
    candidates = universe.listed_stock_candidates(alpaca_candidates + massive_candidates)
    inactive_path = root / "inactive-common-stocks.jsonl.gz"
    write_jsonl_gzip(inactive_path, inactive)
    candidate_payload = {
        "as_of": "2026-10-05",
        "source": "Alpaca current assets plus Massive /v3/reference/tickers inactive common stocks",
        "candidate_count": len(candidates),
        "symbols": [row["symbol"] for row in candidates],
        "records": candidates,
        "alpaca_source_snapshot": alpaca_assets_path.name,
        "alpaca_candidate_count": len(alpaca_candidates),
        "massive_candidate_count": len(massive_candidates),
        "limitations": [
            "This active/inactive snapshot includes delisted symbols but is not a historical listing-date table.",
            "Historical status is represented by provider records and available bar dates; it does not certify "
            "complete exchange membership.",
            "The stock proxy retains the project's exchange, symbol-pattern, and company-name filters.",
            "Massive inactive history is restricted to ticker type CS; delisted ADRC/OS classes are not added.",
        ],
    }
    candidates_path = root / "stock-candidates.json"
    atomic_text(candidates_path, json.dumps(candidate_payload, indent=2) + "\n")
    manifest = {
        "source": "Massive /v3/reference/tickers",
        "as_of": "2026-10-05",
        "active_records": len(alpaca_candidates),
        "inactive_common_stock_records": len(inactive),
        "candidate_records_after_project_filter": len(candidates),
        "alpaca_candidate_records": len(alpaca_candidates),
        "massive_candidate_records": len(massive_candidates),
        "known_delisted_symbols_present": {
            symbol: symbol in {row["symbol"] for row in candidates} for symbol in ["SIVB", "FRC", "BBBY"]
        },
        "inactive_type_filter": "CS",
        "inactive_exchange_filters": sorted(MASSIVE_EXCHANGES),
        "active_sha256": canonical_digest(alpaca_candidates),
        "inactive_common_stocks_sha256": canonical_digest(inactive),
        "candidate_manifest": str(candidates_path),
        "rate_limit_seconds_between_requests": MIN_REQUEST_INTERVAL_SECONDS,
        "no_api_key_persisted": True,
        "created_at": datetime.now(UTC).isoformat(),
    }
    atomic_text(root / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
