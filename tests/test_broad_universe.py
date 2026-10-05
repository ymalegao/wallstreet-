from datetime import date, datetime, timedelta

import polars as pl

from ws.universe import listed_stock_candidates, point_in_time_top_n


def test_listed_stock_filter_keeps_common_shares_and_drops_obvious_non_stocks() -> None:
    assets = [
        {"symbol": "AAPL", "name": "Apple Inc. Common Stock", "exchange": "NASDAQ", "status": "active"},
        {"symbol": "BRK.B", "name": "Berkshire Hathaway Class B", "exchange": "NYSE", "status": "active"},
        {"symbol": "GME", "name": "GameStop Corp. Class A", "exchange": "NYSE", "status": "active"},
        {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust", "exchange": "ARCA", "status": "active"},
        {"symbol": "QQQ", "name": "Invesco QQQ Trust, Series 1", "exchange": "NASDAQ", "status": "active"},
        {"symbol": "SQQQ", "name": "ProShares UltraPro Short QQQ", "exchange": "NASDAQ", "status": "active"},
        {"symbol": "PDI", "name": "PIMCO Dynamic Income Fund", "exchange": "NYSE", "status": "active"},
        {"symbol": "CPT", "name": "Camden Property Trust", "exchange": "NYSE", "status": "active"},
        {"symbol": "ABC.WS", "name": "Example Corporation Warrant", "exchange": "NYSE", "status": "active"},
        {"symbol": "DEAD", "name": "Former Corporation", "exchange": "OTC", "status": "inactive"},
        {"symbol": "003CVR016", "name": "Old contingent value right", "exchange": "NYSE", "status": "inactive"},
    ]

    result = listed_stock_candidates(assets)

    assert [row["symbol"] for row in result] == ["AAPL", "BRK.B", "CPT", "GME"]


def test_listed_stock_filter_prefers_active_duplicate_metadata() -> None:
    assets = [
        {"symbol": "ABC", "name": "Former Name", "exchange": "NYSE", "status": "inactive"},
        {"symbol": "ABC", "name": "Current Name", "exchange": "NYSE", "status": "active"},
    ]

    result = listed_stock_candidates(assets)

    assert len(result) == 1
    assert result[0]["name"] == "Current Name"


def test_point_in_time_top_n_uses_only_prior_twenty_sessions() -> None:
    start = datetime(2024, 1, 1)
    rows = []
    for day in range(21):
        rows.extend(
            [
                {"symbol": "AAA", "ts": start + timedelta(days=day), "close": 10.0, "volume": 10_000_000.0},
                {"symbol": "BBB", "ts": start + timedelta(days=day), "close": 10.0, "volume": 5_000_000.0},
            ]
        )
    # BBB spikes on the selection date; a point-in-time screen must ignore today's volume.
    rows[-1]["volume"] = 100_000_000.0
    selected = point_in_time_top_n(pl.DataFrame(rows), ["AAA", "BBB"], date(2024, 1, 21), date(2024, 1, 22), n=1)
    assert selected.select("symbol").to_series().to_list() == ["AAA"]
