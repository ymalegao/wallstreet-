from datetime import UTC, datetime, timedelta

from ws.schema import Event, TsOrigin
from ws.store.dedupe import jaccard, near_duplicate_clusters, shingles
from ws.store.event_store import EventStore

T0 = datetime(2025, 5, 1, 13, 0, tzinfo=UTC)
STORY = (
    "Acme Corp said on Thursday it would acquire Widget Inc for 4.2 billion dollars in cash, "
    "a 35 percent premium to the closing price, in a deal expected to close in the fourth quarter "
    "subject to regulatory approval and a vote of Widget shareholders."
)


def ev(i: int, text: str, minutes: int, tickers=("ACME",), source="alpaca_news") -> Event:
    t = T0 + timedelta(minutes=minutes)
    return Event(
        event_id=f"{source}:{i}",
        source=source,
        source_id=str(i),
        first_seen_ts=t,
        ts_origin=TsOrigin.VENDOR,
        published_ts=t,
        ingested_at=T0 + timedelta(days=30),
        tickers=list(tickers),
        headline="",
        body=text,
    )


def test_syndicated_copy_merges_into_earliest():
    events = [
        ev(2, STORY + " Shares rose in premarket trading.", 20, source="finnhub"),
        ev(1, STORY, 0),
        ev(3, "Acme reports quarterly earnings above expectations with strong cloud growth.", 5),
    ]
    c = near_duplicate_clusters(events, threshold=0.8)
    assert c["finnhub:2"] == "alpaca_news:1"
    assert c["alpaca_news:3"] == "alpaca_news:3"


def test_outside_window_or_disjoint_tickers_not_merged():
    events = [ev(1, STORY, 0), ev(2, STORY, 60 * 72), ev(3, STORY, 10, tickers=("ZZZ",))]
    c = near_duplicate_clusters(events, threshold=0.8)
    assert c["alpaca_news:2"] == "alpaca_news:2"
    assert c["alpaca_news:3"] == "alpaca_news:3"


def test_jaccard_basics():
    a = shingles(STORY)
    assert jaccard(a, a) == 1.0
    assert jaccard(a, shingles("completely different words about something else entirely here")) == 0.0


def test_event_store_append_only_first_write_wins(tmp_path):
    store = EventStore(tmp_path)
    assert store.write([ev(1, "original text body", 0)]) == 1
    revised = ev(1, "REVISED text body", 0)
    assert store.write([revised, ev(2, "other", 1)]) == 1
    df = EventStore(tmp_path).read()  # fresh instance must also see existing ids
    assert df.height == 2
    assert df.filter(df["event_id"] == "alpaca_news:1")["body"].item() == "original text body"
    assert EventStore(tmp_path).write([revised]) == 0
