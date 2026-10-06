"""Freeze timestamp-safe JEV news inputs for a point-in-time daily top-25 universe."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from ws.labels import assign_cycles, cycle_table, samples
from ws.store.atomic import atomic_text
from ws.store.event_store import EventStore


def exact_deduplicate(events: pl.DataFrame) -> pl.DataFrame:
    """Drop exact same-ticker copies within 48 hours, retaining the earliest available text."""
    if events.is_empty():
        return events
    events = events.sort("first_seen_ts", "event_id")
    seen: dict[tuple[str, str], datetime] = {}
    keep: list[dict[str, Any]] = []
    for row in events.iter_rows(named=True):
        text = f"{row['headline']}\n{row['body']}"
        digest = hashlib.sha256(text.encode()).hexdigest()
        tickers = []
        for ticker in row["tickers"]:
            key = (ticker, digest)
            previous = seen.get(key)
            if previous is None or row["first_seen_ts"] - previous > timedelta(hours=48):
                tickers.append(ticker)
                seen[key] = row["first_seen_ts"]
        if tickers:
            row["tickers"] = tickers
            keep.append(row)
    return pl.DataFrame(keep, schema=events.schema) if keep else events.clear()


def canonical_json(rows: list[dict[str, Any]]) -> str:
    return json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n"


def build(args: argparse.Namespace) -> None:
    universe = json.loads(args.universe.read_text())
    window = universe["decision_window"]
    start = date.fromisoformat(args.start or window["start_inclusive"])
    end = date.fromisoformat(args.end or window["end_exclusive"])
    if start >= end:
        raise ValueError("start must precede the exclusive end")
    selection = universe["selection_by_session"]
    selected_pairs = {
        (date.fromisoformat(session), ticker)
        for session, tickers in selection.items()
        if start <= date.fromisoformat(session) < end
        for ticker in tickers
    }

    events = (
        EventStore(args.data_dir)
        .read()
        .filter(
            (pl.col("source") == "alpaca_news")
            & (pl.col("first_seen_ts") >= datetime.combine(start, datetime.min.time(), UTC))
            & (pl.col("first_seen_ts") < datetime.combine(end, datetime.min.time(), UTC))
        )
    )
    events = exact_deduplicate(events)
    cycles = cycle_table(start, end)
    assigned = assign_cycles(events, cycles, latency=timedelta(minutes=args.latency_minutes))
    membership = pl.DataFrame(
        [{"session": session, "ticker": ticker} for session, ticker in sorted(selected_pairs)],
        schema={"session": pl.Date, "ticker": pl.String},
    )
    assigned = assigned.join(membership, on=["session", "ticker"], how="inner")
    grouped = samples(assigned)
    by_id = {row["event_id"]: row for row in events.iter_rows(named=True)}
    output = []
    for row in grouped.iter_rows(named=True):
        ids = sorted(row["event_ids"], key=lambda event_id: (by_id[event_id]["first_seen_ts"], event_id))
        output.append(
            {
                "ticker": row["ticker"],
                "cycle": row["cycle_ts"].isoformat(),
                "session": row["session"].isoformat(),
                "events": [
                    {
                        "id": event_id,
                        "available_at": by_id[event_id]["first_seen_ts"].isoformat(),
                        "text": (by_id[event_id]["headline"] + "\n" + by_id[event_id]["body"])[:7000],
                    }
                    for event_id in ids
                ],
            }
        )
    output.sort(key=lambda row: (row["cycle"], row["ticker"]))
    raw = canonical_json(output)
    digest = hashlib.sha256(raw.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(args.output, raw)
    manifest = {
        "status": "FROZEN_EXPLORATORY_NEWS_INPUTS",
        "period": {"start_inclusive": start.isoformat(), "end_exclusive": end.isoformat()},
        "universe_path": str(args.universe),
        "universe_sha256": hashlib.sha256(args.universe.read_bytes()).hexdigest(),
        "universe_sessions": len(selection),
        "selected_ticker_cycles": len(selected_pairs) * 2,
        "news_ticker_cycles": len(output),
        "event_ticker_scores": sum(len(row["events"]) for row in output),
        "latency_minutes": args.latency_minutes,
        "latency_status": "UNVERIFIED source-level assumption; shown for exploratory sensitivity only",
        "deduplication": "exact headline+body SHA-256 per ticker within 48 hours; fuzzy threshold remains UNVERIFIED",
        "input_sha256": digest,
        "input_path": str(args.output),
        "time_window_filter": "first_seen_ts in [start midnight UTC, end midnight UTC)",
        "scope": "JEV signal test; no Kronos forecasts or Kronos-derived exclusions",
    }
    atomic_text(args.manifest, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--universe", type=Path, default=Path("data/kronos-pit-2024-2026/universe.json"))
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None, help="exclusive date")
    parser.add_argument("--latency-minutes", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("data/jev-pit-replay-inputs.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/jev-pit-replay-manifest.json"))
    build(parser.parse_args())


if __name__ == "__main__":
    main()
