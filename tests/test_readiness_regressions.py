import runpy
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest

from ws.ingest.http import RateLimitedClient, retry_delay
from ws.quality import audit
from ws.schema import Event, TsOrigin
from ws.store.atomic import atomic_path
from ws.store.event_store import EventStore


def event():
    t = datetime(2025, 1, 2, 22, 30, tzinfo=UTC)
    return Event(
        event_id="x:1",
        source="x",
        source_id="1",
        first_seen_ts=t,
        published_ts=t,
        ingested_at=t,
        ts_origin=TsOrigin.VENDOR,
        tickers=["X"],
        headline="example",
    )


def test_independent_open_writers_preserve_first_write(tmp_path):
    a, b = EventStore(tmp_path), EventStore(tmp_path)
    a._known_ids()
    b._known_ids()
    assert a.write([event()]) == 1
    assert b.write([event()]) == 0
    assert EventStore(tmp_path).read().height == 1


def test_failed_atomic_write_preserves_prior_file(tmp_path):
    path = tmp_path / "x"
    path.write_text("old")
    with pytest.raises(RuntimeError), atomic_path(path) as tmp:
        tmp.write_text("incomplete")
        raise RuntimeError("power failure")
    assert path.read_text() == "old"
    assert list(tmp_path.glob(".pending-*")) == []


def test_dq_handles_evening_timestamps_and_rejects_missing_bars(tmp_path):
    EventStore(tmp_path).write([event()])
    cfg = {
        "edgar": {"acceptance_tz": "utc"},
        "bars": {"feed": "sip"},
        "dedupe": {"jaccard_threshold": "UNVERIFIED"},
        "latency": {"news_minutes": "UNVERIFIED"},
    }
    out = audit(tmp_path, cfg)
    assert out["metrics"]["outside_clock_hours_fraction"] == 1.0
    assert out["status"] == "FAIL"
    assert "no intraday bars" in out["failures"]
    assert "unverified latency.news_minutes" in out["failures"]


def test_backfill_range_respects_both_boundaries_and_state_dimensions(tmp_path):
    mod = runpy.run_path(str(Path(__file__).parents[1] / "scripts/backfill.py"))
    months, state = mod["months"], mod["State"](tmp_path, "bars")
    spans = list(months(date(2025, 1, 15), date(2025, 3, 4)))
    assert spans[0][0].date() == date(2025, 1, 15)
    assert spans[-1][1].date() == date(2025, 3, 4)
    old = state.key(symbol="AAPL", start="2025-01-01", end="2025-02-01", feed="sip")
    new = state.key(symbol="AAPL", start="2025-01-01", end="2025-03-01", feed="sip")
    state.mark(old, 10)
    assert new not in state.done


def test_transport_failure_retried_and_http_date_supported(monkeypatch):
    monkeypatch.setattr("ws.ingest.http.time.sleep", lambda _: None)
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("temporary", request=request)
        return httpx.Response(200, json={"ok": True})

    with RateLimitedClient("https://test", transport=httpx.MockTransport(handler)) as c:
        assert c.get_json("/") == {"ok": True}
    assert len(calls) == 2
    assert retry_delay("Wed, 21 Oct 2015 07:28:00 GMT", 0) == 0
    assert retry_delay("invalid", 2) == 4
