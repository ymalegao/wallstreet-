"""Freeze a bounded replay input set before scoring, using only as-of features."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from ws import calendar as cal
from ws import universe
from ws.labels import assign_cycles, cycle_table, samples
from ws.market_time import daily_model_timestamp
from ws.store.atomic import atomic_text
from ws.store.event_store import BarStore, EventStore


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("data/research-manifest.json"))
    ap.add_argument("--prefix", default="", help="prefix for frozen replay and Kronos input filenames")
    a = ap.parse_args()
    root = Path("data")
    manifest = json.loads(a.manifest.read_text())
    start, end = date.fromisoformat(manifest["start"]), date.fromisoformat(manifest["end"])
    events = EventStore(root).read().filter(pl.col("source") == "alpaca_news")
    # Only exact text copies are removed until the approximate threshold is validated.
    events = events.sort("first_seen_ts", "event_id")
    rows = []
    seen: dict[tuple[str, str], object] = {}
    for r in events.iter_rows(named=True):
        text = r["headline"] + "\n" + r["body"]
        digest = hashlib.sha256(text.encode()).hexdigest()
        tickers = []
        for ticker in r["tickers"]:
            key = (ticker, digest)
            if key not in seen or r["first_seen_ts"] - seen[key] > timedelta(hours=48):
                tickers.append(ticker)
                seen[key] = r["first_seen_ts"]
        if tickers:
            r["tickers"] = tickers
            rows.append(r)
    events = pl.DataFrame(rows, schema=events.schema)
    assigned = assign_cycles(events, cycle_table(start, end), latency=timedelta(minutes=5))
    assigned = assigned.filter(pl.col("ticker").is_in(manifest["symbols"]) & (pl.col("session") < end))
    daily_symbols = sorted(set(manifest["symbols"]) | {"SPY"})
    daily = BarStore(root, "1Day_raw").read(daily_symbols)
    eligibility = universe.eligibility(daily).filter(pl.col("eligible")).rename({"symbol": "ticker"})
    groups = samples(assigned).join(eligibility, on=["ticker", "session"], how="inner")
    by_id = {r["event_id"]: r for r in events.iter_rows(named=True)}
    output = []
    for r in groups.iter_rows(named=True):
        ids = sorted(r["event_ids"], key=lambda i: (by_id[i]["first_seen_ts"], i))
        items = [
            {
                "id": i,
                "available_at": by_id[i]["first_seen_ts"].isoformat(),
                "text": (by_id[i]["headline"] + "\n" + by_id[i]["body"])[:7000],
            }
            for i in ids
        ]
        output.append(
            {"ticker": r["ticker"], "cycle": r["cycle_ts"].isoformat(), "session": str(r["session"]), "events": items}
        )
    replay_path = root / f"{a.prefix}replay-inputs.json"
    kronos_path = root / f"{a.prefix}kronos-inputs.json"
    atomic_text(replay_path, json.dumps(output, indent=1) + "\n")
    jobs = []
    for ticker, session in sorted({(r["ticker"], r["session"]) for r in output}):
        day = date.fromisoformat(session)
        hist = daily.filter((pl.col("symbol") == ticker) & (pl.col("ts").dt.date() < day)).sort("ts").tail(400)
        if hist.height != 400:
            raise RuntimeError(f"Insufficient pre-decision history for {ticker} {day}")
        dates = [str(cal.add_sessions(day, k)) for k in range(6)]
        history = [{**r, "ts": daily_model_timestamp(r["ts"]).isoformat()} for r in hist.iter_rows(named=True)]
        jobs.append({"ticker": ticker, "session": session, "history": history, "future_sessions": dates})
    atomic_text(kronos_path, json.dumps(jobs) + "\n")
    print(
        f"Frozen {len(output)} ticker-cycle samples, {sum(len(r['events']) for r in output)} event/ticker scores, "
        f"{len(jobs)} daily price forecasts"
    )


if __name__ == "__main__":
    main()
