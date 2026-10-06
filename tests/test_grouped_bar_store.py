from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from ws.schema import Bar
from ws.store.bar_loader import read_bar_store
from ws.store.grouped_bar_store import GroupedBarStore


def test_grouped_daily_store_round_trips_by_date_and_symbol(tmp_path: Path) -> None:
    store = GroupedBarStore(tmp_path, "massive_1Day_raw")
    session = date(2025, 1, 2)
    bars = [
        Bar(
            symbol=symbol,
            ts=datetime(2025, 1, 2, 14, 30, tzinfo=UTC),
            open=10,
            high=11,
            low=9,
            close=10,
            volume=100,
        )
        for symbol in ["AAPL", "MSFT"]
    ]

    assert store.write_session(session, bars) == 2
    assert store.write_session(session, bars) == 2
    assert read_bar_store(tmp_path, "grouped:massive_1Day_raw", ["MSFT"])["symbol"].to_list() == ["MSFT"]
