"""Convert raw Alpaca daily bars to the clock-filtered PIT Parquet layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from ws.schema import Bar
from ws.store.atomic import atomic_path, atomic_text
from ws.store.pit_bars import PITBarStore, daily_bar_to_pit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, default=Path("configs/universe/sp100-current.json"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/bars/1Day_raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/curated/bars_1d_pit"))
    parser.add_argument("--start", type=date.fromisoformat, default=date(2015, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=datetime.now(UTC).date() + timedelta(days=1))
    parser.add_argument("--vendor-delay-minutes", type=int, default=15)
    args = parser.parse_args()
    universe_manifest = json.loads(args.universe.read_text())
    symbols = universe_manifest["symbols"]
    store = PITBarStore(args.output_dir)
    total = 0
    absent: list[str] = []
    source_files: dict[str, str] = {}
    for symbol in symbols:
        path = args.raw_dir / f"{symbol}.parquet"
        if not path.exists():
            absent.append(symbol)
            continue
        source_files[symbol] = hashlib.sha256(path.read_bytes()).hexdigest()
        frame = pl.read_parquet(path)
        rows = frame.filter(
            (pl.col("ts").dt.date() >= args.start)
            & (pl.col("ts").dt.date() < args.end)
        ).sort("ts")
        converted = [
            daily_bar_to_pit(
                Bar.model_validate(row), vendor_delay=timedelta(minutes=args.vendor_delay_minutes)
            )
            for row in rows.iter_rows(named=True)
        ]
        total += store.write(converted)
        print(f"{symbol}: {len(converted)} source rows", flush=True)
    parquet_files = sorted(args.output_dir.glob("year=*/month=*/part-*.parquet"))
    file_hashes = {
        str(path.relative_to(args.output_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in parquet_files
    }
    if parquet_files:
        stored = pl.scan_parquet([str(path) for path in parquet_files]).select(
            pl.len().alias("rows"),
            pl.col("symbol").n_unique().alias("symbols"),
            pl.col("event_time").min().alias("first_event_time"),
            pl.col("event_time").max().alias("last_event_time"),
        ).collect().to_dicts()[0]
    else:
        stored = {"rows": 0, "symbols": 0, "first_event_time": None, "last_event_time": None}
    for key in ("first_event_time", "last_event_time"):
        if isinstance(stored[key], datetime):
            stored[key] = stored[key].isoformat()
    manifest = {
        "schema_version": 1,
        "universe_as_of": universe_manifest.get("as_of"),
        "universe_source_url": universe_manifest.get("source_url"),
        "universe_sha256": hashlib.sha256(args.universe.read_bytes()).hexdigest(),
        "requested_symbols": symbols,
        "requested_start": args.start.isoformat(),
        "requested_end_exclusive": args.end.isoformat(),
        "source": "Alpaca SIP daily bars, adjustment=raw",
        "available_at_rule": f"XNYS session close + {args.vendor_delay_minutes} minutes",
        "corporate_action_adjustment": None,
        "source_bar_file_sha256": source_files,
        "curated_files_sha256": file_hashes,
        "coverage": stored,
        "symbols_without_raw_files": absent,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    snapshot_id = hashlib.sha256(canonical.encode()).hexdigest()
    snapshot_root = Path("data/snapshots") / snapshot_id
    for source_path in parquet_files:
        relative = source_path.relative_to(args.output_dir)
        snapshot_file = snapshot_root / "bars" / relative
        if snapshot_file.exists():
            if hashlib.sha256(snapshot_file.read_bytes()).hexdigest() != file_hashes[str(relative)]:
                raise RuntimeError(f"immutable snapshot file has changed: {snapshot_file}")
            continue
        with atomic_path(snapshot_file) as temp:
            shutil.copyfile(source_path, temp)
    snapshot_path = snapshot_root / "manifest.json"
    atomic_text(snapshot_path, json.dumps({"snapshot_id": snapshot_id, **manifest}, indent=2) + "\n")
    atomic_text(Path("data/snapshots/current.json"), json.dumps({"snapshot_id": snapshot_id}, indent=2) + "\n")
    print(
        f"inserted {total} PIT bars; stored {stored['rows']} rows for {stored['symbols']} symbols; "
        f"snapshot={snapshot_id}; no raw file for {len(absent)} symbols: {absent}"
    )


if __name__ == "__main__":
    main()
