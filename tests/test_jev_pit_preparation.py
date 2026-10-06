from __future__ import annotations

import importlib.util
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

_SPEC = importlib.util.spec_from_file_location(
    "prepare_jev_pit_replay", Path(__file__).parents[1] / "scripts/prepare_jev_pit_replay.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_PREPARE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PREPARE)


def test_exact_deduplicate_is_per_ticker_and_uses_first_available_copy() -> None:
    base = datetime(2025, 1, 2, tzinfo=UTC)
    events = pl.DataFrame(
        [
            {"event_id": "first", "first_seen_ts": base, "tickers": ["AAPL"], "headline": "H", "body": "B"},
            {
                "event_id": "duplicate",
                "first_seen_ts": base + timedelta(hours=4),
                "tickers": ["AAPL", "MSFT"],
                "headline": "H",
                "body": "B",
            },
            {
                "event_id": "later",
                "first_seen_ts": base + timedelta(hours=50),
                "tickers": ["AAPL"],
                "headline": "H",
                "body": "B",
            },
        ]
    )
    result = _PREPARE.exact_deduplicate(events)
    assert result["event_id"].to_list() == ["first", "duplicate", "later"]
    assert result["tickers"].to_list() == [["AAPL"], ["MSFT"], ["AAPL"]]
