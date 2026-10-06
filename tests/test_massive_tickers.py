from __future__ import annotations

from scripts.backfill_massive_tickers import reference_params


def test_inactive_reference_query_is_restricted_to_common_stocks() -> None:
    assert reference_params(active=False, ticker_type="CS", exchange="XNAS") == {
        "market": "stocks",
        "active": "false",
        "limit": 1000,
        "sort": "ticker",
        "order": "asc",
        "type": "CS",
        "exchange": "XNAS",
    }


def test_active_query_defaults_to_all_types() -> None:
    assert "type" not in reference_params(active=True)
