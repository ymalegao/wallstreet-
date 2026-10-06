from __future__ import annotations

from datetime import date

from scripts.backfill_massive_grouped_daily import bars_for_session
from ws.calendar import session_open


def test_grouped_daily_response_filters_to_candidate_stocks() -> None:
    bars = bars_for_session(
        [
            {"T": "AAPL", "o": 10, "h": 11, "l": 9, "c": 10.5, "v": 100, "n": 4},
            {"T": "ETF", "o": 10, "h": 11, "l": 9, "c": 10.5, "v": 100, "n": 4},
        ],
        {"AAPL"},
        date(2025, 1, 2),
    )

    assert len(bars) == 1
    assert bars[0].symbol == "AAPL"
    assert bars[0].ts == session_open(date(2025, 1, 2))
    assert bars[0].close == 10.5
