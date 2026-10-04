"""Historical backfill. Downloads raw payloads first (always safe), normalizes second (needs verified config).

    uv run python scripts/backfill.py news --start 2015-01-01
    uv run python scripts/backfill.py daily-bars --start 2016-01-01     # raw (unadjusted), for the universe filter
    uv run python scripts/backfill.py intraday-bars --start 2016-01-01  # 15Min, adjusted, eligible tickers + SPY
    uv run python scripts/backfill.py edgar
    uv run python scripts/backfill.py finnhub
    uv run python scripts/backfill.py normalize

Each step is resumable: finished months/tickers are recorded in data/state/<step>.json.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from ws import sources_config, universe
from ws.config import Settings, load_settings
from ws.ingest import alpaca_news, bars, edgar, finnhub
from ws.schema import Event
from ws.store.event_store import BarStore, EventStore
from ws.store.raw_store import RawStore


class State:
    def __init__(self, root: Path, step: str) -> None:
        self.path = root / "state" / f"{step}.json"
        self.done: set[str] = set(json.loads(self.path.read_text())) if self.path.exists() else set()

    def mark(self, key: str) -> None:
        self.done.add(key)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(sorted(self.done)))


def months(start: date, end: date) -> Iterable[tuple[datetime, datetime]]:
    cur = date(start.year, start.month, 1)
    while cur <= end:
        nxt = date(cur.year + cur.month // 12, cur.month % 12 + 1, 1)
        yield datetime(cur.year, cur.month, 1, tzinfo=UTC), datetime(nxt.year, nxt.month, 1, tzinfo=UTC)
        cur = nxt


def news(st: Settings, start: date, end: date) -> None:
    st.require("alpaca_key", "alpaca_secret")
    raw, state = RawStore(st.data_dir), State(st.data_dir, "news")
    client = alpaca_news.make_client(st.alpaca_key or "", st.alpaca_secret or "")
    for s, e in months(start, end):
        key = s.strftime("%Y-%m")
        if key in state.done:
            continue
        batch: list[dict[str, Any]] = []
        for item in alpaca_news.iter_raw(client, s, min(e, datetime.now(UTC))):
            batch.append(item)
            if len(batch) >= 5000:
                raw.write("alpaca_news", batch)
                batch = []
        raw.write("alpaca_news", batch)
        if e <= datetime.now(UTC):
            state.mark(key)
        print(f"news {key} done ({client.calls} requests so far)")


def news_tickers(st: Settings) -> list[str]:
    seen: set[str] = set()
    for rec in RawStore(st.data_dir).iter("alpaca_news"):
        seen.update(s.upper() for s in rec["payload"].get("symbols") or [])
    return sorted(t for t in seen if t.isascii() and "/" not in t)  # crypto pairs handled separately


def _bars_step(
    st: Settings, step: str, tickers: list[str], fetch: Callable[[list[str]], Iterable[Any]], store: BarStore
) -> None:
    state = State(st.data_dir, step)
    todo = [t for t in tickers if t not in state.done]
    for i in range(0, len(todo), 50):
        chunk = todo[i : i + 50]
        store.write(fetch(chunk))
        for t in chunk:
            state.mark(t)
        print(f"{step}: {min(i + 50, len(todo))}/{len(todo)}")


def daily_bars(st: Settings, start: date, end: date) -> None:
    st.require("alpaca_key", "alpaca_secret")
    feed = sources_config.verified(sources_config.load(), "bars.feed")
    client = bars.make_client(st.alpaca_key or "", st.alpaca_secret or "")
    s, e = datetime.combine(start, datetime.min.time(), UTC), datetime.combine(end, datetime.min.time(), UTC)
    tickers = sorted(set(news_tickers(st)) | {"SPY"})
    _bars_step(
        st,
        "daily_bars",
        tickers,
        lambda ch: bars.stock_bars(client, ch, s, e, timeframe="1Day", feed=feed, adjustment="raw"),
        BarStore(st.data_dir, "1Day_raw"),
    )


def intraday_bars(st: Settings, start: date, end: date) -> None:
    st.require("alpaca_key", "alpaca_secret")
    feed = sources_config.verified(sources_config.load(), "bars.feed")
    client = bars.make_client(st.alpaca_key or "", st.alpaca_secret or "")
    daily = BarStore(st.data_dir, "1Day_raw").read()
    tickers = sorted(set(universe.ever_eligible(daily)) | {"SPY"})
    print(f"{len(tickers)} tickers ever eligible")
    s, e = datetime.combine(start, datetime.min.time(), UTC), datetime.combine(end, datetime.min.time(), UTC)
    _bars_step(
        st,
        "intraday_bars",
        tickers,
        lambda ch: bars.stock_bars(client, ch, s, e, timeframe="15Min", feed=feed, adjustment="all"),
        BarStore(st.data_dir, "15Min"),
    )


def edgar_step(st: Settings) -> None:
    st.require("sec_user_agent")
    client = edgar.make_client(st.sec_user_agent or "")
    cik_map = edgar.ticker_to_cik(client)
    daily = BarStore(st.data_dir, "1Day_raw").read()
    tickers = universe.ever_eligible(daily) if not daily.is_empty() else news_tickers(st)
    raw, state = RawStore(st.data_dir), State(st.data_dir, "edgar")
    for t in tickers:
        if t in state.done:
            continue
        if t in cik_map:
            raw.write("edgar", edgar.iter_filings(client, cik_map[t]))
        state.mark(t)


def finnhub_step(st: Settings) -> None:
    st.require("finnhub_key")
    client = finnhub.make_client(st.finnhub_key or "")
    daily = BarStore(st.data_dir, "1Day_raw").read()
    tickers = universe.ever_eligible(daily)
    raw, state = RawStore(st.data_dir), State(st.data_dir, "finnhub")
    end = datetime.now(UTC).date()
    for t in tickers:
        if t in state.done:
            continue
        for k in range(0, 365, 30):  # small windows: the endpoint truncates large ranges
            rows = finnhub.company_news(client, t, end - timedelta(days=k + 30), end - timedelta(days=k))
            raw.write("finnhub", [{"symbol": t, **r} for r in rows])
        state.mark(t)


def normalize(st: Settings) -> None:
    cfg = sources_config.load()
    tz = sources_config.verified(cfg, "edgar.acceptance_tz")
    raw, store = RawStore(st.data_dir), EventStore(st.data_dir)
    rejects = st.data_dir / "state" / "rejects.jsonl"
    rejects.parent.mkdir(parents=True, exist_ok=True)

    def run(source: str, fn: Callable[[dict[str, Any]], Event]) -> None:
        good: list[Event] = []
        bad = 0
        with rejects.open("a") as rej:
            for rec in raw.iter(source):
                try:
                    good.append(fn(rec["payload"]))
                except Exception as exc:
                    bad += 1
                    rej.write(
                        json.dumps({"source": source, "error": str(exc), "payload": rec["payload"]}, default=str) + "\n"
                    )
                if len(good) >= 20_000:
                    store.write(good)
                    good = []
        store.write(good)
        print(f"normalize {source}: rejects={bad}")

    run("alpaca_news", alpaca_news.normalize)
    run("edgar", lambda p: edgar.normalize(p, acceptance_tz=tz))
    run("finnhub", lambda p: finnhub.normalize(p, p["symbol"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["news", "daily-bars", "intraday-bars", "edgar", "finnhub", "normalize"])
    ap.add_argument("--start", type=date.fromisoformat, default=date(2016, 1, 1))
    ap.add_argument("--end", type=date.fromisoformat, default=datetime.now(UTC).date())
    a = ap.parse_args()
    st = load_settings()
    {
        "news": lambda: news(st, a.start, a.end),
        "daily-bars": lambda: daily_bars(st, a.start, a.end),
        "intraday-bars": lambda: intraday_bars(st, a.start, a.end),
        "edgar": lambda: edgar_step(st),
        "finnhub": lambda: finnhub_step(st),
        "normalize": lambda: normalize(st),
    }[a.step]()


if __name__ == "__main__":
    main()
