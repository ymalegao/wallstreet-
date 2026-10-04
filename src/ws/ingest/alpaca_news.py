"""Alpaca News API (Benzinga feed): historical REST backfill and live WebSocket stream.

Endpoint shapes follow Alpaca's public docs; ``scripts/probe_apis.py`` checks every field this
module relies on against the live API before any backfill is trusted.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import Any

from ws.ingest.http import RateLimitedClient
from ws.ingest.text import html_to_text
from ws.schema import Event, TsOrigin
from ws.timeutil import parse_rfc3339, utcnow

BASE_URL = "https://data.alpaca.markets"
NEWS_PATH = "/v1beta1/news"
STREAM_URL = "wss://stream.data.alpaca.markets/v1beta1/news"
PAGE_LIMIT = 50  # documented maximum page size
SOURCE = "alpaca_news"


def make_client(key: str, secret: str, max_per_sec: float = 3.0) -> RateLimitedClient:
    # Free plan: 200 requests/min. 3/s keeps headroom.
    return RateLimitedClient(
        BASE_URL,
        headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
        max_per_sec=max_per_sec,
    )


def iter_raw(
    client: RateLimitedClient,
    start: datetime,
    end: datetime,
    symbols: list[str] | None = None,
    include_content: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield raw article dicts in ascending time order for [start, end)."""
    params: dict[str, Any] = {
        "start": start.isoformat().replace("+00:00", "Z"),
        "end": end.isoformat().replace("+00:00", "Z"),
        "limit": PAGE_LIMIT,
        "sort": "asc",
        "include_content": str(include_content).lower(),
    }
    if symbols:
        params["symbols"] = ",".join(symbols)
    while True:
        page = client.get_json(NEWS_PATH, params)
        yield from page.get("news", [])
        token = page.get("next_page_token")
        if not token:
            return
        params["page_token"] = token


def normalize(raw: dict[str, Any], *, observed_at: datetime | None = None) -> Event:
    """Normalize one raw article.

    ``observed_at`` is set only for live-stream items: our receipt time becomes ``first_seen_ts``.
    For backfill, ``first_seen_ts`` is the vendor's ``created_at`` (``ts_origin=vendor``).
    """
    published = parse_rfc3339(raw["created_at"])
    updated = parse_rfc3339(raw["updated_at"]) if raw.get("updated_at") else None
    # Live: the earliest moment *we* could act is our receipt time. Backfill: vendor publish time;
    # the backtest adds the latency measured live (probe report) on top via next_decision_cycle().
    first_seen = observed_at if observed_at else published
    summary = html_to_text(raw.get("summary"))
    content = html_to_text(raw.get("content"))
    return Event(
        event_id=f"{SOURCE}:{raw['id']}",
        source=SOURCE,
        source_id=str(raw["id"]),
        first_seen_ts=first_seen,
        ts_origin=TsOrigin.OBSERVED if observed_at else TsOrigin.VENDOR,
        published_ts=published,
        updated_ts=updated,
        ingested_at=utcnow(),
        tickers=list(raw.get("symbols") or []),
        headline=html_to_text(raw.get("headline")),
        body=content or summary,
        url=raw.get("url") or "",
        kind="news",
        meta={"author": raw.get("author") or "", "vendor": raw.get("source") or ""},
    )


async def stream(key: str, secret: str, on_event: Callable[[Event, dict[str, Any]], None]) -> None:
    """Consume the live news WebSocket forever, calling ``on_event(event, raw)`` for each article."""
    import websockets

    async with websockets.connect(STREAM_URL) as ws:
        await ws.send(json.dumps({"action": "auth", "key": key, "secret": secret}))
        await ws.send(json.dumps({"action": "subscribe", "news": ["*"]}))
        async for msg in ws:
            received = utcnow()
            for item in json.loads(msg):
                if item.get("T") == "n":
                    on_event(normalize(item, observed_at=received), item)
                elif item.get("T") == "error":
                    raise RuntimeError(f"alpaca stream error: {item}")
