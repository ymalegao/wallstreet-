from ws.universe import listed_stock_candidates


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
