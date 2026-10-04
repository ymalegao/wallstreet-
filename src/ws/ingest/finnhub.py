"""Finnhub company news (free tier: ~1 year of history, REST only). Used as a second source."""

from __future__ import annotations

from datetime import date
from typing import Any

from ws.ingest.http import RateLimitedClient
from ws.ingest.text import html_to_text
from ws.schema import Event, TsOrigin
from ws.timeutil import parse_epoch, utcnow

BASE_URL = "https://finnhub.io/api/v1"
SOURCE = "finnhub"


def make_client(token: str) -> RateLimitedClient:
    # Free tier: 60 calls/min.
    return RateLimitedClient(BASE_URL, headers={"X-Finnhub-Token": token}, max_per_sec=0.9)


def company_news(client: RateLimitedClient, symbol: str, start: date, end: date) -> list[dict[str, Any]]:
    data = client.get_json("/company-news", {"symbol": symbol, "from": start.isoformat(), "to": end.isoformat()})
    return list(data or [])


def normalize(raw: dict[str, Any], symbol: str) -> Event:
    published = parse_epoch(raw["datetime"])
    related = [s for s in str(raw.get("related") or "").split(",") if s]
    return Event(
        event_id=f"{SOURCE}:{raw['id']}",
        source=SOURCE,
        source_id=str(raw["id"]),
        first_seen_ts=published,
        ts_origin=TsOrigin.VENDOR,
        published_ts=published,
        ingested_at=utcnow(),
        tickers=sorted({symbol, *related}),
        headline=html_to_text(raw.get("headline")),
        body=html_to_text(raw.get("summary")),
        url=raw.get("url") or "",
        kind="news",
        meta={"vendor": raw.get("source") or "", "category": raw.get("category") or ""},
    )
