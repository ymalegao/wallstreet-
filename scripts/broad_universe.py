"""Cache a broad listed-equity master and resumably backfill daily SIP bars.

The current Alpaca asset endpoint is not point-in-time security classification. This script keeps
the raw metadata snapshot, applies a documented heuristic to remove obvious funds/derivatives,
then lets historical price/liquidity rules build the date-specific watchlist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx

from ws import sources_config, universe
from ws.config import Settings, load_settings
from ws.ingest import bars
from ws.store.atomic import atomic_text, file_lock
from ws.store.event_store import BarStore


def batches(values: list[str], size: int = 10) -> Iterator[list[str]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def asset_snapshot(st: Settings, out: Path) -> list[dict[str, Any]]:
    st.require("alpaca_key", "alpaca_secret")
    headers = {"APCA-API-KEY-ID": st.alpaca_key or "", "APCA-API-SECRET-KEY": st.alpaca_secret or ""}
    assets: dict[tuple[str, str], dict[str, Any]] = {}
    with httpx.Client(headers=headers, timeout=30) as client:
        for status in ("active", "inactive"):
            response = client.get(
                "https://paper-api.alpaca.markets/v2/assets",
                params={"status": status, "asset_class": "us_equity"},
            )
            response.raise_for_status()
            for raw in response.json():
                row = {key: raw.get(key) for key in (
                    "symbol", "name", "exchange", "status", "tradable", "marginable", "shortable", "fractionable"
                )}
                row["attributes"] = raw.get("attributes") or []
                assets[(str(row.get("symbol") or ""), status)] = row
    rows = sorted(assets.values(), key=lambda row: (str(row.get("symbol")), str(row.get("status"))))
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).date().isoformat()
    path = out / f"alpaca_assets_{stamp}.jsonl"
    atomic_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    atomic_text(
        out / f"alpaca_assets_{stamp}.meta.json",
        json.dumps(
            {
                "source": "Alpaca /v2/assets",
                "queried_at": datetime.now(UTC).isoformat(),
                "statuses": ["active", "inactive"],
                "asset_class": "us_equity",
                "records": len(rows),
                "note": "Current metadata snapshot, not point-in-time historical membership.",
            },
            indent=2,
        )
        + "\n",
    )
    return rows


def backfill(st: Settings, symbols: list[str], start: date, end: date, adjustment: str) -> None:
    st.require("alpaca_key", "alpaca_secret")
    feed = sources_config.verified(sources_config.load(), "bars.feed")
    provenance = {
        "feed": feed,
        "adjustment": adjustment,
        "timeframe": "1Day",
        "symbols_sha256": hashlib.sha256(",".join(symbols).encode()).hexdigest(),
    }
    step = f"broad-daily-bars-{adjustment}-{start.isoformat()}-{end.isoformat()}-v1"
    state_path = st.data_dir / "state" / f"{step}.json"
    manifest_path = st.data_dir / "state" / f"{step}-manifest.json"
    lock_path = st.data_dir / "state" / f"{step}.lock"
    with file_lock(lock_path):
        expected = {**provenance, "start": start.isoformat(), "end": end.isoformat()}
        has_progress = state_path.exists() and bool(json.loads(state_path.read_text()))
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != expected and has_progress:
            raise RuntimeError("Broad-bar provenance/range changed; use a new data directory.")
        atomic_text(manifest_path, json.dumps(expected, indent=2) + "\n")
        done = json.loads(state_path.read_text()) if state_path.exists() else {}
        store = BarStore(st.data_dir, f"1Day_{adjustment}")
        s = datetime.combine(start, datetime.min.time(), UTC)
        e = datetime.combine(end, datetime.min.time(), UTC)
        with bars.make_client(st.alpaca_key or "", st.alpaca_secret or "") as client:
            for i, group in enumerate(batches(symbols), start=1):
                key = ",".join(group)
                if key in done:
                    continue
                downloaded = list(
                    bars.stock_bars(client, group, s, e, timeframe="1Day", feed=feed, adjustment=adjustment)
                )
                got = {item.symbol for item in downloaded}
                count = store.write(downloaded)
                done[key] = {
                    "rows": count,
                    "empty": sorted(set(group) - got),
                    "completed_at": datetime.now(UTC).isoformat(),
                }
                atomic_text(state_path, json.dumps(done, indent=2) + "\n")
                if i % 25 == 0 or i == 1:
                    print(
                        f"broad daily batches {i}/{(len(symbols) + 9) // 10}; "
                        f"rows in batch={count}; empty={len(group) - len(got)}",
                        flush=True,
                    )
    rows = sum(v["rows"] for v in done.values())
    print(f"broad daily complete: {len(symbols)} symbols, {len(done)} batches, {rows} bars", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2024, 10, 4))
    parser.add_argument("--end", type=date.fromisoformat, default=datetime.now(UTC).date())
    parser.add_argument("--adjustment", choices=["raw", "all"], default="raw")
    parser.add_argument(
        "--skip-download", action="store_true", help="refresh/calc the asset list without requesting bars"
    )
    args = parser.parse_args()
    st = load_settings()
    out = st.data_dir / "universe"
    assets = asset_snapshot(st, out)
    selected = universe.listed_stock_candidates(assets)
    symbols = sorted({str(a["symbol"]) for a in selected} | {"SPY"})
    stamp = datetime.now(UTC).date().isoformat()
    atomic_text(
        out / f"listed_stock_candidates_{stamp}.json",
        json.dumps(
            {
                "as_of": stamp,
                "source": "Alpaca /v2/assets",
                "symbols": symbols,
                "candidate_count": len(symbols),
                "filters": {
                    "exchanges": sorted(universe.EXCHANGES),
                    "symbol_pattern": universe.SYMBOL_PATTERN.pattern,
                    "excluded_name_terms": list(universe.EXCLUDED_NAME_TERMS),
                },
                "limitations": [
                    "Alpaca does not label security type here; ETF/fund filtering by name is approximate.",
                    "Current active/inactive status is not point-in-time historical membership.",
                    "Price and liquidity eligibility must be calculated using prior sessions only.",
                ],
            },
            indent=2,
        )
        + "\n",
    )
    print(f"asset snapshot={len(assets)}; listed-stock proxy={len(symbols)}", flush=True)
    if not args.skip_download:
        backfill(st, symbols, args.start, args.end, args.adjustment)


if __name__ == "__main__":
    main()
