"""Add SPY forecast jobs matching the frozen broad-watchlist daily sessions."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from ws.market_time import daily_model_timestamp
from ws.store.atomic import atomic_text
from ws.store.event_store import BarStore


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, default=Path("data/broad-kronos-inputs.json"))
    ap.add_argument("--output", type=Path, default=Path("data/broad-kronos-sweep-inputs.json"))
    a = ap.parse_args()

    jobs: list[dict[str, Any]] = json.loads(a.inputs.read_text())
    spy = BarStore(Path("data"), "1Day_raw").read(["SPY"]).sort("ts")
    if spy.is_empty():
        raise RuntimeError("No cached SPY daily bars")
    sessions = sorted({job["session"] for job in jobs})
    for session in sessions:
        day = date.fromisoformat(session)
        history = spy.filter(pl.col("ts").dt.date() < day).tail(400)
        if history.height != 400:
            raise RuntimeError(f"Insufficient SPY history before {day}")
        matching = next(job for job in jobs if job["session"] == session)
        jobs.append(
            {
                "ticker": "SPY",
                "session": session,
                "history": [
                    {**row, "ts": daily_model_timestamp(row["ts"]).isoformat()} for row in history.iter_rows(named=True)
                ],
                "future_sessions": matching["future_sessions"],
            }
        )
    atomic_text(a.output, json.dumps(jobs) + "\n")
    print(f"Prepared {len(jobs)} forecasts including {len(sessions)} matched SPY sessions")


if __name__ == "__main__":
    main()
