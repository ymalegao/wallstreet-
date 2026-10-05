"""Alpaca historical bars for stocks and crypto.

Stocks use the SIP feed with ``adjustment=all`` (split- and dividend-adjusted) so that returns are
correct across corporate actions. The free plan serves SIP history except the most recent 15
minutes, which is fine for backfill and labels.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import Any

from ws.ingest.http import RateLimitedClient
from ws.schema import Bar
from ws.timeutil import parse_rfc3339

BASE_URL = "https://data.alpaca.markets"


def make_client(key: str, secret: str) -> RateLimitedClient:
    return RateLimitedClient(BASE_URL, headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}, max_per_sec=3.0)


def _iter_pages(client: RateLimitedClient, path: str, params: dict[str, Any]) -> Iterator[Bar]:
    seen_tokens: set[str] = set()
    while True:
        page = client.get_json(path, params)
        for symbol, rows in (page.get("bars") or {}).items():
            for r in rows:
                yield Bar(
                    symbol=symbol,
                    ts=parse_rfc3339(r["t"]),
                    open=r["o"],
                    high=r["h"],
                    low=r["l"],
                    close=r["c"],
                    volume=r["v"],
                    vwap=r.get("vw"),
                    trade_count=r.get("n"),
                )
        token = page.get("next_page_token")
        if not token:
            return
        if token in seen_tokens:
            raise RuntimeError("Alpaca bars repeated a pagination token")
        seen_tokens.add(token)
        params["page_token"] = token


def stock_bars(
    client: RateLimitedClient,
    symbols: list[str],
    start: datetime,
    end: datetime,
    timeframe: str = "15Min",
    feed: str = "sip",
    adjustment: str = "all",
) -> Iterator[Bar]:
    params: dict[str, Any] = {
        "symbols": ",".join(symbols),
        "timeframe": timeframe,
        "start": start.isoformat().replace("+00:00", "Z"),
        "end": end.isoformat().replace("+00:00", "Z"),
        "adjustment": adjustment,
        "feed": feed,
        "limit": 10_000,
        "sort": "asc",
    }
    return _iter_pages(client, "/v2/stocks/bars", params)


def crypto_bars(
    client: RateLimitedClient, symbols: list[str], start: datetime, end: datetime, timeframe: str = "1Hour"
) -> Iterator[Bar]:
    params: dict[str, Any] = {
        "symbols": ",".join(symbols),
        "timeframe": timeframe,
        "start": start.isoformat().replace("+00:00", "Z"),
        "end": end.isoformat().replace("+00:00", "Z"),
        "limit": 10_000,
        "sort": "asc",
    }
    return _iter_pages(client, "/v1beta3/crypto/us/bars", params)
