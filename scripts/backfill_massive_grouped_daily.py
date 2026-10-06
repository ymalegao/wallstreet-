"""Cache broad unadjusted daily bars via Massive's one-request-per-session endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from dotenv import load_dotenv

from scripts.backfill_massive_tickers import RateLimitedReferenceClient
from ws.calendar import is_session, session_open
from ws.schema import Bar
from ws.store.atomic import atomic_text
from ws.store.grouped_bar_store import GroupedBarStore

BASE_URL = "https://api.massive.com/v2/aggs/grouped/locale/us/market/stocks"
STORE_NAME = "massive_1Day_raw"


def candidate_symbols(path: Path) -> set[str]:
    payload = json.loads(path.read_text())
    return {row["symbol"] for row in payload["records"]}


def sessions(start: date, end: date) -> list[date]:
    result = []
    current = start
    while current < end:
        if is_session(current):
            result.append(current)
        current += timedelta(days=1)
    return result


def bars_for_session(rows: list[dict[str, Any]], symbols: set[str], session: date) -> list[Bar]:
    ts = session_open(session)
    result = []
    for row in rows:
        symbol = str(row.get("T", "")).upper()
        if symbol not in symbols:
            continue
        result.append(
            Bar(
                symbol=symbol,
                ts=ts,
                open=row["o"],
                high=row["h"],
                low=row["l"],
                close=row["c"],
                volume=row["v"],
                vwap=row.get("vw"),
                trade_count=row.get("n"),
            )
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("data/universe/massive-reference-2026-10-05/stock-candidates.json"),
    )
    parser.add_argument("--start", type=date.fromisoformat, default=date(2024, 7, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 7, 10), help="exclusive end")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    if args.end <= args.start:
        raise ValueError("end must be after start")
    load_dotenv(".env")
    api_key = os.environ.get("MASSIVE_API_KEY")
    if not api_key:
        raise SystemExit("MASSIVE_API_KEY is missing; key value was not read from the command line")

    allowed = candidate_symbols(args.candidates)
    allowed_digest = hashlib.sha256("\n".join(sorted(allowed)).encode()).hexdigest()
    calendar_sessions = sessions(args.start, args.end)
    store = GroupedBarStore(args.data_dir, STORE_NAME)
    state_path = args.data_dir / "state" / f"{STORE_NAME}-{args.start}-{args.end}.json"
    manifest_path = args.data_dir / "state" / f"{STORE_NAME}-manifest.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest and manifest.get("candidate_symbols_sha256") != allowed_digest:
        raise RuntimeError("Cached grouped bars belong to a different candidate pool")
    if manifest and (
        manifest.get("start_inclusive") != args.start.isoformat()
        or manifest.get("end_exclusive") != args.end.isoformat()
    ):
        raise RuntimeError("Cached grouped bars use a different date interval")
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    manifest = {
        "status": "IN_PROGRESS",
        "source": "Massive /v2/aggs/grouped/locale/us/market/stocks/{date}",
        "store": f"grouped:{STORE_NAME}",
        "adjustment": "unadjusted",
        "include_otc": False,
        "candidate_symbols_sha256": allowed_digest,
        "candidate_symbol_count": len(allowed),
        "start_inclusive": args.start.isoformat(),
        "end_exclusive": args.end.isoformat(),
        "trading_sessions": len(calendar_sessions),
        "rate_limit_seconds_between_requests": 12.2,
        "no_api_key_persisted": True,
    }
    atomic_text(manifest_path, json.dumps(manifest, indent=2) + "\n")
    client = RateLimitedReferenceClient(api_key)
    for index, session in enumerate(calendar_sessions, 1):
        key = session.isoformat()
        if key in state:
            continue
        cached_path = store.dir / f"{key}.parquet"
        if cached_path.exists():
            cached_count = pl.read_parquet(cached_path).height
            state[key] = {"candidate_bars": cached_count, "recovered_from_atomic_cache": True}
            atomic_text(state_path, json.dumps(state, sort_keys=True) + "\n")
            continue
        response = client.get(
            f"{BASE_URL}/{key}",
            {"adjusted": "false", "include_otc": "false"},
        )
        if response.get("status") not in {"OK", "DELAYED"}:
            raise RuntimeError(f"Massive grouped daily response status was {response.get('status')!r} for {key}")
        bars = bars_for_session(response.get("results") or [], allowed, session)
        count = store.write_session(session, bars)
        state[key] = {"candidate_bars": count, "completed_at": datetime.now(UTC).isoformat()}
        atomic_text(state_path, json.dumps(state, sort_keys=True) + "\n")
        print(f"Massive grouped daily {key}: {count} stock bars ({index}/{len(calendar_sessions)})", flush=True)

    total_bars = sum(row["candidate_bars"] for row in state.values())
    atomic_text(
        manifest_path,
        json.dumps(
            {
                "status": "COMPLETE",
                "source": "Massive /v2/aggs/grouped/locale/us/market/stocks/{date}",
                "store": f"grouped:{STORE_NAME}",
                "adjustment": "unadjusted",
                "include_otc": False,
                "candidate_symbols_sha256": allowed_digest,
                "candidate_symbol_count": len(allowed),
                "start_inclusive": args.start.isoformat(),
                "end_exclusive": args.end.isoformat(),
                "trading_sessions": len(calendar_sessions),
                "cached_sessions": len(state),
                "candidate_bars": total_bars,
                "rate_limit_seconds_between_requests": 12.2,
                "no_api_key_persisted": True,
            },
            indent=2,
        )
        + "\n",
    )
    print(f"Massive grouped daily complete: {len(state)}/{len(calendar_sessions)} sessions, {total_bars} bars")


if __name__ == "__main__":
    main()
