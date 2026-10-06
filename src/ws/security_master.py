"""Reference-record conversion for broad US stock-universe construction."""

from __future__ import annotations

from typing import Any

from ws import universe

MASSIVE_EXCHANGES = {"XNAS": "NASDAQ", "XNYS": "NYSE", "XASE": "AMEX"}
COMMON_STOCK_TYPES = {"CS", "ADRC", "OS"}


def massive_stock_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Massive US-stock records to the project's listed-stock proxy schema."""
    assets = []
    for row in records:
        if row.get("market") != "stocks" or row.get("locale") != "us":
            continue
        if row.get("type") not in COMMON_STOCK_TYPES and row.get("type") is not None:
            continue
        if row.get("currency_name") not in {None, "usd"}:
            continue
        exchange = MASSIVE_EXCHANGES.get(str(row.get("primary_exchange") or ""))
        if exchange is None:
            continue
        symbol = str(row.get("ticker") or "").strip().upper()
        if not symbol:
            continue
        assets.append(
            {
                "symbol": symbol,
                "name": row.get("name") or "",
                "exchange": exchange,
                "status": "active" if row.get("active") else "inactive",
                "delisted_utc": row.get("delisted_utc"),
                "cik": row.get("cik"),
                "share_class_figi": row.get("share_class_figi"),
                "type": row.get("type"),
                "source": "Massive /v3/reference/tickers",
            }
        )
    return universe.listed_stock_candidates(assets)
