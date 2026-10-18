"""Synthetic point-in-time bars for pipeline v0 integration tests."""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime, time
from pathlib import Path

import exchange_calendars as xcals

from ws.bars import Bar
from ws.store.pit_bars import PITBarStore, daily_bar_to_pit

SYMBOLS = tuple(f"SYM{index:02d}" for index in range(30))


def build_fixture(root: Path) -> tuple[PITBarStore, Path, tuple[str, ...]]:
    """Write three years of deterministic synthetic bars and a matching universe file."""
    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range("2021-01-04", "2023-12-29")
    rng = random.Random(20211229)
    bars = []
    for symbol_index, symbol in enumerate(SYMBOLS):
        close = 40.0 + symbol_index * 4.0
        drift = (symbol_index - (len(SYMBOLS) - 1) / 2) * 0.00003
        for session in sessions:
            session_date = session.date()
            ts = datetime.combine(session_date, time(12), UTC)
            opening = close * (1.0 + rng.gauss(0.0, 0.0003))
            close = max(1.0, close * (1.0 + drift + rng.gauss(0.0, 0.0015)))
            volume = 500_000.0 + (symbol_index * 10_000.0) + rng.uniform(-20_000.0, 20_000.0)
            bars.append(
                daily_bar_to_pit(
                    Bar(
                        symbol=symbol,
                        ts=ts,
                        open=opening,
                        high=max(opening, close) * 1.001,
                        low=min(opening, close) * 0.999,
                        close=close,
                        volume=volume,
                        vwap=(opening + close) / 2.0,
                        trade_count=1000,
                    )
                )
            )

    spy_rng = random.Random(20211230)
    spy_close = 100.0
    for session in sessions:
        session_date = session.date()
        ts = datetime.combine(session_date, time(12), UTC)
        spy_open = spy_close * (1.0 + spy_rng.gauss(0.0, 0.0002))
        spy_close = max(1.0, spy_close * (1.00025 + spy_rng.gauss(0.0, 0.002)))
        bars.append(
            daily_bar_to_pit(
                Bar(
                    symbol="SPY",
                    ts=ts,
                    open=spy_open,
                    high=max(spy_open, spy_close) * 1.001,
                    low=min(spy_open, spy_close) * 0.999,
                    close=spy_close,
                    volume=8_000_000.0,
                    vwap=(spy_open + spy_close) / 2.0,
                    trade_count=10_000,
                )
            )
        )

    store = PITBarStore(root / "bars")
    store.write(bars)
    universe_path = root / "universe.json"
    universe_path.write_text(
        json.dumps(
            {
                "index": "synthetic-test-universe",
                "as_of": "2023-12-29",
                "symbols": list(SYMBOLS),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return store, universe_path, SYMBOLS
