"""Resumably cache Massive daily or intraday bars for recently delisted candidates."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import httpx
from dotenv import load_dotenv

from ws.schema import Bar
from ws.store.atomic import atomic_text
from ws.store.event_store import BarStore

BASE_URL = "https://api.massive.com"
REQUEST_INTERVAL_SECONDS = 12.2


def without_api_key(url: str) -> str:
    parts = urlsplit(url)
    query = [(key, value) for key, value in parse_qsl(parts.query) if key.lower() != "apikey"]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


class Client:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.http = httpx.Client(timeout=90)
        self.last_request: float | None = None

    def get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        for attempt in range(4):
            delay = REQUEST_INTERVAL_SECONDS
            if self.last_request is not None:
                delay = max(0, REQUEST_INTERVAL_SECONDS - (time.monotonic() - self.last_request))
            if delay:
                time.sleep(delay)
            self.last_request = time.monotonic()
            try:
                response = self.http.get(url, params={**(params or {}), "apiKey": self.api_key})
                if response.status_code == 429:
                    retry_after = float(response.headers.get("Retry-After", "60"))
                    time.sleep(max(retry_after, REQUEST_INTERVAL_SECONDS))
                    continue
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500 or attempt == 3:
                    raise RuntimeError(f"Massive bars request failed with HTTP {exc.response.status_code}") from None
            except httpx.TransportError:
                if attempt == 3:
                    raise RuntimeError("Massive bars request failed after retries") from None
            time.sleep(max(2**attempt, REQUEST_INTERVAL_SECONDS))
        raise RuntimeError("Massive bars request exhausted retries")


def recently_delisted_candidates(path: Path, start: date) -> list[str]:
    payload = json.loads(path.read_text())
    return sorted(
        {
            row["symbol"]
            for row in payload["records"]
            if row.get("status") == "inactive"
            and row.get("delisted_utc")
            and date.fromisoformat(str(row["delisted_utc"])[:10]) >= start
        }
    )


def fetch_ticker(
    client: Client,
    symbol: str,
    start: date,
    end: date,
    multiplier: int,
    timespan: str,
    adjusted: bool,
) -> list[Bar]:
    ticker = quote(symbol, safe=".-")
    final_date = end - timedelta(days=1)
    url: str | None = (
        f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start}/{final_date}"
    )
    params: dict[str, Any] | None = {"adjusted": str(adjusted).lower(), "sort": "asc", "limit": 50_000}
    bars: list[Bar] = []
    while url:
        payload = client.get(url, params)
        params = None
        if payload.get("status") not in {"OK", "DELAYED", "DELAYED/SIP"}:
            raise RuntimeError(f"Massive bars response status was {payload.get('status')!r} for {symbol}")
        for row in payload.get("results") or []:
            bars.append(
                Bar(
                    symbol=symbol,
                    ts=datetime.fromtimestamp(row["t"] / 1000, tz=UTC),
                    open=row["o"],
                    high=row["h"],
                    low=row["l"],
                    close=row["c"],
                    volume=row["v"],
                    vwap=row.get("vw"),
                    trade_count=row.get("n"),
                )
            )
        next_url = payload.get("next_url")
        url = without_api_key(next_url) if next_url else None
    return bars


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("data/universe/massive-reference-2026-10-05/stock-candidates.json"),
    )
    parser.add_argument("--start", type=date.fromisoformat, default=date(2023, 7, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 7, 10), help="exclusive end")
    parser.add_argument("--timespan", choices=["day", "minute"], default="day")
    parser.add_argument("--multiplier", type=int, default=1)
    parser.add_argument("--adjusted", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--store", required=True, help="distinct BarStore timeframe, e.g. 1Day_massive_raw")
    parser.add_argument("--skip-store", help="skip tickers already covered in this store, e.g. 1Day_raw")
    args = parser.parse_args()
    if args.end <= args.start or args.multiplier < 1:
        raise ValueError("invalid date interval or multiplier")
    load_dotenv(".env")
    api_key = os.environ.get("MASSIVE_API_KEY")
    if not api_key:
        raise SystemExit("MASSIVE_API_KEY is missing; key value was not read from the command line")
    symbols = recently_delisted_candidates(args.candidates, args.start)
    output = BarStore(Path("data"), args.store)
    output_bars = output.read() if output.dir.exists() else None
    cached = (
        set(output_bars["symbol"].unique().to_list())
        if output_bars is not None and not output_bars.is_empty()
        else set()
    )
    skipped = set()
    if args.skip_store:
        skipped_bars = BarStore(Path("data"), args.skip_store).read()
        if not skipped_bars.is_empty():
            skipped = set(skipped_bars["symbol"].unique().to_list())
    state_path = Path("data/state") / f"massive_{args.store}_{args.start}_{args.end}.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    client = Client(api_key)
    for i, symbol in enumerate(symbols, 1):
        if symbol in state or symbol in cached or symbol in skipped:
            continue
        bars = fetch_ticker(client, symbol, args.start, args.end, args.multiplier, args.timespan, args.adjusted)
        added = output.write(bars)
        state[symbol] = {"bars": added, "completed_at": datetime.now(UTC).isoformat()}
        atomic_text(state_path, json.dumps(state, sort_keys=True) + "\n")
        print(f"Massive {args.store} {symbol}: {added} bars ({i}/{len(symbols)})", flush=True)
    manifest_path = Path("data/state") / f"{args.store}-massive-manifest.json"
    atomic_text(
        manifest_path,
        json.dumps(
            {
                "source": "Massive /v2/aggs/ticker/{ticker}/range",
                "timeframe": args.store,
                "timespan": args.timespan,
                "multiplier": args.multiplier,
                "adjusted_for_splits": args.adjusted,
                "start_inclusive": args.start.isoformat(),
                "end_exclusive": args.end.isoformat(),
                "rate_limit_seconds_between_requests": REQUEST_INTERVAL_SECONDS,
                "candidate_count": len(symbols),
                "cached_or_skipped_count": len(set(symbols) & (cached | skipped)),
                "completed_symbol_count": len(state),
                "no_api_key_persisted": True,
            },
            indent=2,
        )
        + "\n",
    )
    print(f"Massive {args.store} complete: {len(state)}/{len(symbols)} candidates")


if __name__ == "__main__":
    main()
