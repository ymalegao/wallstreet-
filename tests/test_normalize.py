"""Normalizer tests. Payloads below are shaped after each provider's public docs, NOT captured
responses; scripts/probe_apis.py checks the same fields against the live APIs."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from ws.ingest import alpaca_news, edgar, finnhub
from ws.schema import Event, TsOrigin
from ws.timeutil import ET

ALPACA_DOC_SHAPE = {
    "id": 24843171,
    "headline": "Apple Reports <b>Q1</b> Results",
    "author": "Benzinga Newsdesk",
    "created_at": "2022-01-27T21:31:23Z",
    "updated_at": "2022-01-27T21:45:00Z",
    "summary": "Revenue up 11%",
    "content": "<p>Apple &amp; co reported...</p>",
    "url": "https://example.com/a",
    "symbols": ["aapl", "AAPL"],
    "source": "benzinga",
}


def test_alpaca_backfill_uses_vendor_time():
    e = alpaca_news.normalize(ALPACA_DOC_SHAPE)
    assert e.event_id == "alpaca_news:24843171"
    assert e.ts_origin is TsOrigin.VENDOR
    assert e.first_seen_ts == datetime(2022, 1, 27, 21, 31, 23, tzinfo=UTC)
    assert e.tickers == ["AAPL"]
    assert e.headline == "Apple Reports Q1 Results"
    assert e.body == "Apple & co reported..."
    assert e.revised


def test_alpaca_live_uses_receipt_time():
    seen = datetime(2022, 1, 27, 21, 31, 25, tzinfo=UTC)
    e = alpaca_news.normalize(ALPACA_DOC_SHAPE, observed_at=seen)
    assert e.ts_origin is TsOrigin.OBSERVED
    assert e.first_seen_ts == seen
    assert e.published_ts == datetime(2022, 1, 27, 21, 31, 23, tzinfo=UTC)


EDGAR_ROW = {
    "accessionNumber": "0000320193-24-000081",
    "filingDate": "2024-08-01",
    "acceptanceDateTime": "2024-08-01T16:30:45.000Z",
    "form": "8-K",
    "items": "2.02,9.01",
    "primaryDocument": "aapl-20240801.htm",
    "cik": 320193,
    "tickers": ["AAPL"],
}


@pytest.mark.parametrize(
    ("tz", "expected"),
    [
        ("utc", datetime(2024, 8, 1, 16, 30, 45, tzinfo=UTC)),
        ("et", datetime(2024, 8, 1, 16, 30, 45, tzinfo=ET).astimezone(UTC)),
    ],
)
def test_edgar_acceptance_timezone_is_explicit(tz, expected):
    e = edgar.normalize(EDGAR_ROW, acceptance_tz=tz)
    assert e.first_seen_ts == expected
    assert e.kind == "8-K"
    assert "Item 2.02: Results of operations" in e.headline
    assert e.url.endswith("/320193/000032019324000081/aapl-20240801.htm")


def test_finnhub_epoch_and_related():
    raw = {"id": 7, "datetime": 1700000000, "headline": "X", "summary": "y", "related": "MSFT,AAPL", "url": "u"}
    e = finnhub.normalize(raw, "AAPL")
    assert e.first_seen_ts == datetime.fromtimestamp(1700000000, UTC)
    assert e.tickers == ["AAPL", "MSFT"]


def test_event_rejects_naive_and_acausal_timestamps():
    now = datetime.now(UTC)
    base = dict(event_id="s:1", source="s", source_id="1", ts_origin=TsOrigin.VENDOR, headline="h")
    with pytest.raises(ValidationError):
        Event(**base, first_seen_ts=datetime(2024, 1, 1), published_ts=now, ingested_at=now)
    with pytest.raises(ValidationError):
        Event(**base, first_seen_ts=now + timedelta(hours=1), published_ts=now, ingested_at=now)
