from __future__ import annotations

from ws.security_master import massive_stock_candidates


def test_massive_security_master_includes_delisted_common_stock_and_excludes_etf() -> None:
    records = [
        {
            "ticker": "SIVB",
            "name": "SVB Financial Group",
            "market": "stocks",
            "locale": "us",
            "currency_name": "usd",
            "primary_exchange": "XNAS",
            "active": False,
            "type": "CS",
            "delisted_utc": "2023-03-28T04:00:00Z",
        },
        {
            "ticker": "SPY",
            "name": "SPDR S&P 500 ETF Trust",
            "market": "stocks",
            "locale": "us",
            "currency_name": "usd",
            "primary_exchange": "ARCX",
            "active": True,
            "type": "ETF",
        },
        {
            "ticker": "XYZ",
            "name": "A foreign share",
            "market": "stocks",
            "locale": "global",
            "currency_name": "usd",
            "primary_exchange": "XNAS",
            "active": True,
            "type": "CS",
        },
        {
            "ticker": "BADF",
            "name": "Bad Example Fund",
            "market": "stocks",
            "locale": "us",
            "currency_name": "usd",
            "primary_exchange": "XNYS",
            "active": True,
            "type": "FUND",
        },
        {
            "ticker": "OSHR",
            "name": "Ordinary Shares Inc",
            "market": "stocks",
            "locale": "us",
            "currency_name": "usd",
            "primary_exchange": "XNAS",
            "active": True,
            "type": "OS",
        },
    ]
    candidates = massive_stock_candidates(records)
    assert [row["symbol"] for row in candidates] == ["OSHR", "SIVB"]
    assert candidates[1]["status"] == "inactive"
    assert candidates[1]["delisted_utc"] == "2023-03-28T04:00:00Z"
