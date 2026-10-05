"""Freeze 90-bar Kronos inputs through the actual morning/afternoon decision cycles."""

from __future__ import annotations

import argparse
import json
from bisect import bisect_left
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from ws import calendar as cal
from ws.labels import regular_hours
from ws.store.atomic import atomic_text
from ws.store.event_store import BarStore
from ws.timeutil import ET


def naive_local(dt: datetime) -> str:
    return dt.astimezone(ET).replace(tzinfo=None).isoformat()


def make_30m(bars: pl.DataFrame) -> dict[tuple[str, date], list[dict[str, Any]]]:
    grouped: dict[tuple[str, date], list[dict[str, Any]]] = defaultdict(list)
    for row in bars.sort("ts").iter_rows(named=True):
        local = row["ts"].astimezone(ET)
        grouped[(row["symbol"], local.date())].append({**row, "local_ts": local})
    result: dict[tuple[str, date], list[dict[str, Any]]] = {}
    for key, rows in grouped.items():
        by_ts = {row["local_ts"].time(): row for row in rows}
        session_open = datetime.combine(key[1], time(9, 30), tzinfo=ET)
        session_close = cal.session_close(key[1]).astimezone(ET)
        out = []
        start = session_open
        while start + timedelta(minutes=30) <= session_close:
            a, b = by_ts.get(start.time()), by_ts.get((start + timedelta(minutes=15)).time())
            if a is not None and b is not None:
                volume = float(a["volume"] + b["volume"])
                amount = float(a["volume"] * a["vwap"] + b["volume"] * b["vwap"])
                out.append(
                    {
                        "ts": start.replace(tzinfo=None).isoformat(),
                        "open": float(a["open"]),
                        "high": max(float(a["high"]), float(b["high"])),
                        "low": min(float(a["low"]), float(b["low"])),
                        "close": float(b["close"]),
                        "volume": volume,
                        "amount": amount,
                    }
                )
            start += timedelta(minutes=30)
        result[key] = out
    return result


def future_starts(cycle: datetime, interval: int) -> tuple[list[str], date]:
    session = cycle.astimezone(ET).date()
    target = cal.add_sessions(session, 5)
    future = []
    for offset in range(6):
        d = cal.add_sessions(session, offset)
        opened = cal.session_open(d).astimezone(ET)
        closed = cal.session_close(d).astimezone(ET)
        start = opened
        while start + timedelta(minutes=interval) <= closed:
            if start >= cycle.astimezone(ET):
                future.append(start.replace(tzinfo=None).isoformat())
            start += timedelta(minutes=interval)
    return future, target


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--signals", type=Path, default=Path("data/broad-replay-inputs.json"))
    ap.add_argument("--output-dir", type=Path, default=Path("data"))
    ap.add_argument("--lookback", type=int, default=90)
    a = ap.parse_args()

    samples = json.loads(a.signals.read_text())
    symbols = sorted({row["ticker"] for row in samples} | {"SPY"})
    bars = regular_hours(BarStore(Path("data"), "15Min").read(symbols)).sort("ts")
    bars15: dict[tuple[str, date], list[dict[str, Any]]] = defaultdict(list)
    for row in bars.iter_rows(named=True):
        local = row["ts"].astimezone(ET)
        bars15[(row["symbol"], local.date())].append(
            {
                "ts": local.replace(tzinfo=None).isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "amount": float(row["volume"] * row["vwap"]),
            }
        )
    bars30 = make_30m(bars)
    by_symbol: dict[int, dict[str, list[dict[str, Any]]]] = {15: {}, 30: {}}
    for (symbol, _), rows in bars15.items():
        by_symbol[15].setdefault(symbol, []).extend(rows)
    for (symbol, _), rows in bars30.items():
        by_symbol[30].setdefault(symbol, []).extend(rows)
    for interval in (15, 30):
        for rows in by_symbol[interval].values():
            rows.sort(key=lambda row: row["ts"])
    times_by_symbol = {
        interval: {
            symbol: [datetime.fromisoformat(row["ts"]).replace(tzinfo=ET) for row in rows]
            for symbol, rows in series.items()
        }
        for interval, series in by_symbol.items()
    }
    cycles = sorted({row["cycle"] for row in samples})
    output: dict[int, list[dict[str, Any]]] = {15: [], 30: []}
    excluded: dict[int, int] = {15: 0, 30: 0}

    for cycle_str in cycles:
        cycle = datetime.fromisoformat(cycle_str)
        et_cycle = cycle.astimezone(ET)
        interval = 15 if et_cycle.time() == time(9, 45) else 30
        future, target_session = future_starts(cycle, interval)
        if not future:
            raise RuntimeError(f"No future bars for {cycle_str}")
        candidates = sorted({row["ticker"] for row in samples if row["cycle"] == cycle_str} | {"SPY"})
        for ticker in candidates:
            rows = by_symbol[interval].get(ticker, [])
            timeline = times_by_symbol[interval].get(ticker, [])
            index = bisect_left(timeline, et_cycle)
            history = rows[max(0, index - a.lookback) : index]
            if len(history) != a.lookback:
                excluded[interval] += 1
                continue
            if datetime.fromisoformat(history[-1]["ts"]).replace(tzinfo=ET) + timedelta(minutes=interval) != et_cycle:
                excluded[interval] += 1
                continue
            output[interval].append(
                {
                    "ticker": ticker,
                    "session": str(et_cycle.date()),
                    "cycle": cycle_str,
                    "interval_minutes": interval,
                    "target_session": str(target_session),
                    "history": history,
                    "future_timestamps": future,
                }
            )

    for interval, name in [(15, "morning-15m"), (30, "afternoon-30m")]:
        path = a.output_dir / f"kronos-intraday-{name}-inputs.json"
        atomic_text(path, json.dumps(output[interval]) + "\n")
        spy_count = sum(x["ticker"] == "SPY" for x in output[interval])
        horizon = len(output[interval][0]["future_timestamps"]) if output[interval] else 0
        print(
            f"{name}: {len(output[interval])} jobs; {spy_count} SPY; "
            f"excluded {excluded[interval]}; horizon {horizon} bars",
            flush=True,
        )


if __name__ == "__main__":
    main()
