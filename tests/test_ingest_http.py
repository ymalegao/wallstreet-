"""Pagination and retry behaviour against a mocked transport (doc-shaped responses)."""

from datetime import UTC, date, datetime, timedelta

import httpx
import polars as pl

from ws import universe
from ws.ingest import alpaca_news
from ws.ingest.http import RateLimitedClient


def test_alpaca_news_follows_page_tokens_and_retries_429():
    calls: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(dict(req.url.params))
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        if "page_token" not in req.url.params:
            return httpx.Response(200, json={"news": [{"id": 1}], "next_page_token": "abc"})
        return httpx.Response(200, json={"news": [{"id": 2}], "next_page_token": None})

    client = RateLimitedClient(base_url="https://x", max_per_sec=1000, transport=httpx.MockTransport(handler))
    s, e = datetime(2025, 1, 2, tzinfo=UTC), datetime(2025, 1, 3, tzinfo=UTC)
    got = list(alpaca_news.iter_raw(client, s, e, symbols=["AAPL", "MSFT"]))
    assert [g["id"] for g in got] == [1, 2]
    assert calls[1]["start"] == "2025-01-02T00:00:00Z" and calls[1]["symbols"] == "AAPL,MSFT"
    assert calls[1]["limit"] == "50" and calls[1]["sort"] == "asc"
    assert calls[2]["page_token"] == "abc"


def test_universe_eligibility_is_point_in_time():
    # Price jumps from $3 to $50 on day index 25; dollar volume is above $20M throughout.
    rows = [
        {
            "symbol": "X",
            "ts": datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=i),
            "close": 3.0 if i < 25 else 50.0,
            "volume": 1e7,
        }
        for i in range(30)
    ]
    e = universe.eligibility(pl.DataFrame(rows)).filter(pl.col("eligible"))
    # Day 25's own close may not qualify day 25: first eligible session is day 26 (2025-01-27).
    assert e["session"].min() == date(2025, 1, 27)
