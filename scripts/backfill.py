"""Bounded, resumable downloads. All date intervals are [start, end), in UTC.

State v2 includes symbols, dates, feed and adjustment. Old ticker-only state is not reused.
Writes are atomic; interrupted chunks can be fetched again without duplicating normalized rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from ws import sources_config, universe
from ws.config import Settings, load_settings
from ws.ingest import alpaca_news, bars, edgar, finnhub
from ws.schema import Event
from ws.store.atomic import atomic_text, file_lock
from ws.store.event_store import BarStore, EventStore
from ws.store.raw_store import RawStore
from ws.timeutil import parse_rfc3339


class State:
    def __init__(self, root: Path, step: str) -> None:
        self.path = root / "state" / f"{step}-v2.json"
        self.done: dict[str, Any] = json.loads(self.path.read_text()) if self.path.exists() else {}

    @staticmethod
    def key(**parts: Any) -> str:
        return json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))

    def mark(self, key: str, count: int) -> None:
        # Caller holds the whole-step lock. Empty responses are evidence, not silently lost.
        self.done[key] = {"rows": count, "completed_at": datetime.now(UTC).isoformat()}
        atomic_text(self.path, json.dumps(self.done, indent=2) + "\n")


def months(start: date, end: date) -> Iterator[tuple[datetime, datetime]]:
    if end <= start:
        raise ValueError("end must be after start (exclusive)")
    cur = start
    while cur < end:
        nxt = min(date(cur.year + cur.month // 12, cur.month % 12 + 1, 1), end)
        yield datetime.combine(cur, datetime.min.time(), UTC), datetime.combine(nxt, datetime.min.time(), UTC)
        cur = nxt


def chunks(rows: Iterable[Any], size: int = 2000) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def read_symbols(path: Path) -> list[str]:
    """Read explicit symbols from a JSON list or a universe manifest."""
    payload = json.loads(path.read_text())
    symbols = payload.get("selected_symbols", payload.get("symbols")) if isinstance(payload, dict) else payload
    if not isinstance(symbols, list) or not symbols or not all(isinstance(symbol, str) for symbol in symbols):
        raise ValueError(f"{path} must contain a non-empty symbols list or universe manifest")
    return sorted({symbol.strip().upper() for symbol in symbols if symbol.strip()})


def news(st: Settings, start: date, end: date, symbols: list[str] | None = None) -> None:
    st.require("alpaca_key", "alpaca_secret")
    raw, state = RawStore(st.data_dir), State(st.data_dir, "news")
    with alpaca_news.make_client(st.alpaca_key or "", st.alpaca_secret or "") as client:
        for s, e in months(start, end):
            e = min(e, datetime.now(UTC))
            if e <= s:
                continue
            key = State.key(start=s, end=e, symbols=symbols or "*", content=True)
            if key in state.done:
                continue
            n = 0
            for batch in chunks(alpaca_news.iter_raw(client, s, e, symbols=symbols)):
                # Endpoints sometimes include the end boundary. Never trust response order.
                batch = [r for r in batch if s <= parse_rfc3339(r["created_at"]) < e]
                raw.write("alpaca_news", batch)
                n += len(batch)
            state.mark(key, n)
            print(f"news {s.date()}..{e.date()}: {n} rows", flush=True)


def news_tickers(st: Settings) -> list[str]:
    seen: set[str] = set()
    for rec in RawStore(st.data_dir).iter("alpaca_news"):
        seen.update(s.upper() for s in rec["payload"].get("symbols") or [])
    return sorted(t for t in seen if t.isascii() and "/" not in t and "\\" not in t)


def bars_step(
    st: Settings, start: date, end: date, intraday: bool, symbols: list[str] | None, daily_adjusted: bool = False
) -> None:
    st.require("alpaca_key", "alpaca_secret")
    feed = sources_config.verified(sources_config.load(), "bars.feed")
    if intraday:
        step, tf, adj = "intraday_bars", "15Min", "all"
    else:
        step, tf, adj = "daily_bars_all", "1Day", "all"
        if not daily_adjusted:
            step, adj = "daily_bars", "raw"
    state = State(st.data_dir, step)
    if symbols is None:
        symbols = universe.ever_eligible(BarStore(st.data_dir, "1Day_raw").read()) if intraday else news_tickers(st)
    symbols = sorted(set(symbols) | {"SPY"})
    timeframe = "15Min" if intraday else "1Day_all" if daily_adjusted else "1Day_raw"
    store = BarStore(st.data_dir, timeframe)
    manifest = st.data_dir / "state" / f"{step}-manifest.json"
    provenance = {"feed": feed, "adjustment": adj, "timeframe": tf}
    if manifest.exists() and json.loads(manifest.read_text()) != provenance:
        raise RuntimeError("Bar provenance changed; use a new data directory rather than mixing feeds/adjustments")
    atomic_text(manifest, json.dumps(provenance, indent=2) + "\n")
    with bars.make_client(st.alpaca_key or "", st.alpaca_secret or "") as client:
        for s, e in months(start, end):
            for offset in range(0, len(symbols), 10):
                syms = symbols[offset : offset + 10]
                key = State.key(start=s, end=e, symbols=syms, **provenance)
                if key in state.done:
                    continue
                seen: set[str] = set()
                n = 0
                for batch in chunks(bars.stock_bars(client, syms, s, e, timeframe=tf, feed=feed, adjustment=adj)):
                    batch = [b for b in batch if s <= b.ts < e]
                    seen.update(b.symbol for b in batch)
                    n += len(batch)
                    store.write(batch)
                state.mark(key, n)
                print(f"{step} {s.date()}..{e.date()}: {n} rows; empty={sorted(set(syms) - seen)}", flush=True)


def edgar_step(st: Settings, start: date, end: date, symbols: list[str] | None) -> None:
    st.require("sec_user_agent")
    cfg = sources_config.load()
    tz = sources_config.verified(cfg, "edgar.acceptance_tz")
    raw, state = RawStore(st.data_dir), State(st.data_dir, "edgar")
    s, e = datetime.combine(start, datetime.min.time(), UTC), datetime.combine(end, datetime.min.time(), UTC)
    with edgar.make_client(st.sec_user_agent or "") as client:
        mapping = edgar.ticker_to_cik(client)
        for symbol in symbols or news_tickers(st):
            key = State.key(symbol=symbol, start=start, end=end, content=True)
            if key in state.done:
                continue
            if symbol not in mapping:
                print(f"edgar {symbol}: unresolved CIK (not marked complete)", flush=True)
                continue
            n = 0
            for row in edgar.iter_filings(client, mapping[symbol]):
                if not s <= edgar.parse_acceptance(row["acceptanceDateTime"], tz) < e:
                    continue
                row["body"] = edgar.fetch_text(client, row)
                raw.write("edgar", [row])
                n += 1
            state.mark(key, n)
            print(f"edgar {symbol}: {n} documents", flush=True)


def finnhub_step(st: Settings, start: date, end: date, symbols: list[str] | None) -> None:
    st.require("finnhub_key")
    raw, state = RawStore(st.data_dir), State(st.data_dir, "finnhub")
    with finnhub.make_client(st.finnhub_key or "") as client:
        for symbol in symbols or news_tickers(st):
            for s, e in months(start, end):
                key = State.key(symbol=symbol, start=s, end=e)
                if key in state.done:
                    continue
                rows = finnhub.company_news(client, symbol, s.date(), (e - timedelta(days=1)).date())
                rows = [r for r in rows if s.timestamp() <= r["datetime"] < e.timestamp()]
                raw.write("finnhub", [{"symbol": symbol, **r} for r in rows])
                state.mark(key, len(rows))
                print(f"finnhub {symbol} {s.date()}: {len(rows)} rows", flush=True)


def normalize(st: Settings) -> int:
    cfg = sources_config.load()
    raw, store = RawStore(st.data_dir), EventStore(st.data_dir)
    state = State(st.data_dir, "normalize")
    rejected: list[dict[str, str]] = []
    count = 0
    for source in ["alpaca_news", "edgar", "finnhub"]:
        good: list[Event] = []
        for rec in raw.iter(source):
            p = rec["payload"]
            try:
                if source == "alpaca_news":
                    event = alpaca_news.normalize(p)
                elif source == "edgar":
                    event = edgar.normalize(
                        p, acceptance_tz=sources_config.verified(cfg, "edgar.acceptance_tz"), body=p.get("body", "")
                    )
                else:
                    event = finnhub.normalize(p, p["symbol"])
                event = Event.model_validate({**event.model_dump(), "ingested_at": parse_rfc3339(rec["fetched_at"])})
                good.append(event)
            except Exception as exc:
                rejected.append(
                    {
                        "source": source,
                        "id": str(p.get("id", p.get("accessionNumber", ""))),
                        "error": type(exc).__name__,
                    }
                )
            if len(good) >= 2000:
                count += store.write(good)
                good = []
        count += store.write(good)
    atomic_text(st.data_dir / "state" / "rejects.jsonl", "".join(json.dumps(r) + "\n" for r in rejected))
    state.mark("summary", count)
    print(f"normalized new={count}, rejects={len(rejected)}", flush=True)
    return len(rejected)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "step", choices=["news", "daily-bars", "daily-bars-all", "intraday-bars", "edgar", "finnhub", "normalize"]
    )
    ap.add_argument("--start", type=date.fromisoformat, default=date(2016, 1, 1))
    ap.add_argument("--end", type=date.fromisoformat, default=datetime.now(UTC).date(), help="exclusive UTC date")
    ap.add_argument("--symbols", help="comma-separated explicit integration universe")
    ap.add_argument("--symbols-file", type=Path, help="JSON symbol list or universe manifest with selected_symbols")
    a = ap.parse_args()
    if a.symbols and a.symbols_file:
        ap.error("use only one of --symbols and --symbols-file")
    symbols = sorted(set(a.symbols.upper().split(","))) if a.symbols else None
    if a.symbols_file:
        symbols = read_symbols(a.symbols_file)
    st = load_settings()
    # Serialize a source's download/checkpoint transaction, while providers share HTTP budgets.
    lock = hashlib.sha256(a.step.encode()).hexdigest()[:12]
    with file_lock(st.data_dir / "state" / f"run-{lock}.lock"):
        if a.step == "news":
            news(st, a.start, a.end, symbols)
        elif a.step in ("daily-bars", "daily-bars-all", "intraday-bars"):
            bars_step(st, a.start, a.end, a.step == "intraday-bars", symbols, a.step == "daily-bars-all")
        elif a.step == "edgar":
            edgar_step(st, a.start, a.end, symbols)
        elif a.step == "finnhub":
            finnhub_step(st, a.start, a.end, symbols)
        elif normalize(st):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
