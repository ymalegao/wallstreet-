"""Validate every data-API assumption the pipeline makes, against the live APIs.

Run this before any backfill is trusted:

    uv run python scripts/probe_apis.py --ws-seconds 600

It writes docs/data-probe-report.md (commit it) and raw samples to data/probe/. Each check prints
PASS / FAIL / MEASURE. MEASURE values (EDGAR timezone, dedupe threshold, news latency, bar feed)
are copied by a human into configs/sources.yaml after reading the report.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from ws.config import load_settings
from ws.ingest import alpaca_news, bars, edgar, finnhub
from ws.store.dedupe import jaccard, shingles
from ws.timeutil import ET, parse_epoch, parse_rfc3339

PROBE_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "JPM", "XOM", "PFE", "WMT"]
DELISTED = ["SIVB", "FRC"]  # failed March/May 2023; tests survivorship-free history
OUT = Path("data/probe")


@dataclass
class Report:
    lines: list[str] = field(default_factory=list)
    status: Counter[str] = field(default_factory=Counter)

    def section(self, title: str) -> None:
        self.lines += ["", f"## {title}", ""]
        print(f"\n== {title}")

    def check(self, status: str, name: str, detail: str = "") -> None:
        self.status[status] += 1
        line = f"- **{status}** {name}" + (f": {detail}" if detail else "")
        self.lines.append(line)
        print(line)

    def text(self, s: str) -> None:
        self.lines.append(s)

    def guard(self, name: str, fn: Any) -> Any:
        try:
            return fn()
        except Exception as exc:  # report and continue: one failing source must not hide the others
            self.check("FAIL", name, f"{type(exc).__name__}: {exc}")
            return None


def save(name: str, obj: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, default=str))


def pct(xs: list[float], q: float) -> float:
    return statistics.quantiles(xs, n=100)[int(q) - 1] if len(xs) >= 2 else (xs[0] if xs else float("nan"))


def last_weekday(days_back: int) -> date:
    d = datetime.now(UTC).date() - timedelta(days=days_back)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def day_range(d: date) -> tuple[datetime, datetime]:
    s = datetime(d.year, d.month, d.day, tzinfo=UTC)
    return s, s + timedelta(days=1)


# --------------------------------------------------------------------------- Alpaca news


def probe_alpaca_news(r: Report, key: str, secret: str) -> list[dict[str, Any]]:
    r.section("Alpaca News API (REST)")
    client = alpaca_news.make_client(key, secret)
    d = last_weekday(3)
    raw = list(alpaca_news.iter_raw(client, *day_range(d)))
    save("alpaca_news_day", raw[:200])
    r.check("PASS" if raw else "FAIL", f"articles on {d}", str(len(raw)))
    if not raw:
        return []
    needed = ["id", "headline", "created_at", "updated_at", "summary", "content", "symbols", "url", "source"]
    for f in needed:
        have = sum(1 for a in raw if a.get(f) not in (None, "", []))
        r.check("PASS" if have else "FAIL", f"field `{f}` populated", f"{have}/{len(raw)}")
    created = [parse_rfc3339(a["created_at"]) for a in raw]
    r.check("PASS" if created == sorted(created) else "FAIL", "sort=asc respected across pages")
    r.check("PASS" if client.calls > 1 else "MEASURE", "pagination exercised", f"{client.calls} requests")
    lags = [
        (parse_rfc3339(a["updated_at"]) - parse_rfc3339(a["created_at"])).total_seconds() / 60
        for a in raw
        if a.get("updated_at")
    ]
    revised = [x for x in lags if x > 0]
    r.check(
        "MEASURE",
        "revised after publication (updated_at > created_at)",
        f"{len(revised)}/{len(lags)}; lag minutes p50={pct(revised, 50):.1f} p95={pct(revised, 95):.1f}"
        if revised
        else "0",
    )
    nsym = [len(a.get("symbols") or []) for a in raw]
    r.check("MEASURE", "symbols per article", f"mean={statistics.mean(nsym):.2f}, zero={nsym.count(0)}")
    r.check("MEASURE", "vendors", str(Counter(a.get("source") for a in raw).most_common(5)))
    r.text("\nHistory depth (articles on one weekday in mid-March of each year, all symbols):\n")
    for year in range(2015, datetime.now(UTC).year + 1):
        d = date(year, 3, 16)
        while d.weekday() >= 5:
            d += timedelta(days=1)
        n = r.guard(
            f"history {year}",
            lambda d=d: sum(1 for _ in alpaca_news.iter_raw(client, *day_range(d), include_content=False)),
        )
        if n is not None:
            r.text(f"- {d}: {n}")
    return raw


async def _ws_collect(key: str, secret: str, seconds: int) -> list[tuple[datetime, dict[str, Any]]]:
    import websockets

    got: list[tuple[datetime, dict[str, Any]]] = []
    async with websockets.connect(alpaca_news.STREAM_URL) as ws:
        await ws.send(json.dumps({"action": "auth", "key": key, "secret": secret}))
        await ws.send(json.dumps({"action": "subscribe", "news": ["*"]}))
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=max(0.1, end - time.monotonic()))
            except TimeoutError:
                break
            now = datetime.now(UTC)
            got += [(now, item) for item in json.loads(msg)]
    return got


def probe_alpaca_ws(r: Report, key: str, secret: str, seconds: int) -> None:
    r.section(f"Alpaca News WebSocket ({seconds}s)")
    msgs = asyncio.run(_ws_collect(key, secret, seconds))
    save("alpaca_ws", [{"received": t, **m} for t, m in msgs])
    ctl = [m for _, m in msgs if m.get("T") != "n"]
    r.check("MEASURE", "control messages", json.dumps(ctl)[:300])
    news = [(t, m) for t, m in msgs if m.get("T") == "n"]
    r.check("PASS" if news else "MEASURE", "news messages received", str(len(news)))
    if news:
        lag = [(t - parse_rfc3339(m["created_at"])).total_seconds() for t, m in news]
        r.check(
            "MEASURE",
            "receipt minus created_at (seconds) -> sets backtest latency",
            f"p50={pct(lag, 50):.1f} p95={pct(lag, 95):.1f} max={max(lag):.1f}",
        )


# --------------------------------------------------------------------------- Alpaca bars


def probe_bars(r: Report, key: str, secret: str) -> None:
    r.section("Alpaca bars")
    client = bars.make_client(key, secret)
    d = last_weekday(5)
    s, e = day_range(d)
    for feed in ("sip", "iex"):
        rows = r.guard(
            f"15Min SPY feed={feed}",
            lambda feed=feed: list(bars.stock_bars(client, ["SPY"], s, e + timedelta(hours=6), feed=feed)),
        )
        if rows is None:
            continue
        et_times = [b.ts.astimezone(ET).time() for b in rows]
        pre = sum(1 for t in et_times if t.hour < 9 or (t.hour == 9 and t.minute < 30))
        post = sum(1 for t in et_times if t.hour >= 16)
        r.check(
            "PASS" if rows else "FAIL",
            f"feed={feed} SPY 15Min bars on {d}",
            f"{len(rows)} bars, pre-market={pre}, after-hours={post}",
        )
        if rows:
            first_rth = next(
                (b for b in rows if b.ts.astimezone(ET).time() >= datetime(2000, 1, 1, 9, 30).time()), None
            )
            r.check(
                "MEASURE",
                f"feed={feed} first RTH bar `t` (bar START expected at 09:30 ET)",
                first_rth.ts.astimezone(ET).isoformat() if first_rth else "none",
            )
    # Split handling: NVDA 10-for-1 split effective 2024-06-10.
    s2 = datetime(2024, 6, 6, tzinfo=UTC)
    e2 = datetime(2024, 6, 12, tzinfo=UTC)
    for adj in ("raw", "all"):
        rows = r.guard(
            f"NVDA daily adjustment={adj}",
            lambda adj=adj: list(bars.stock_bars(client, ["NVDA"], s2, e2, timeframe="1Day", adjustment=adj)),
        )
        if rows:
            r.check("MEASURE", f"NVDA closes adjustment={adj}", ", ".join(f"{b.ts.date()}:{b.close:.2f}" for b in rows))
    for sym in DELISTED:
        rows = r.guard(
            f"delisted {sym}",
            lambda sym=sym: list(
                bars.stock_bars(
                    client, [sym], datetime(2023, 1, 3, tzinfo=UTC), datetime(2023, 2, 1, tzinfo=UTC), timeframe="1Day"
                )
            ),
        )
        if rows is not None:
            r.check(
                "PASS" if rows else "FAIL",
                f"history for delisted {sym} (survivorship)",
                f"{len(rows)} daily bars Jan 2023",
            )
    crypto = r.guard("crypto BTC/USD", lambda: list(bars.crypto_bars(client, ["BTC/USD"], s, e)))
    if crypto is not None:
        r.check("PASS" if crypto else "FAIL", "crypto BTC/USD 1Hour bars", str(len(crypto)))


# --------------------------------------------------------------------------- EDGAR

_HDR = re.compile(r"ACCEPTANCE-DATETIME:\s*(\d{14})")


def probe_edgar(r: Report, user_agent: str) -> None:
    r.section("SEC EDGAR")
    client = edgar.make_client(user_agent)
    cik_map = r.guard("company_tickers.json", lambda: edgar.ticker_to_cik(client))
    if not cik_map:
        return
    r.check("PASS", "ticker->CIK map", f"{len(cik_map)} tickers; AAPL={cik_map.get('AAPL')}")
    rows = r.guard("submissions AAPL", lambda: list(edgar.iter_filings(client, cik_map["AAPL"])))
    if not rows:
        return
    save("edgar_aapl_rows", rows[:50])
    for f in ("acceptanceDateTime", "items", "primaryDocument", "accessionNumber"):
        r.check("PASS" if all(f in x for x in rows) else "FAIL", f"field `{f}` present", f"{len(rows)} 8-K/4 rows")
    verdicts = Counter()
    for row in [x for x in rows if x["form"] == "8-K"][:5]:
        acc = row["accessionNumber"]
        url = f"https://www.sec.gov/Archives/edgar/data/{row['cik']}/{acc.replace('-', '')}/{acc}-index-headers.html"
        m = _HDR.search(client.get(url).text)
        if not m:
            r.check("FAIL", f"header for {acc}", "ACCEPTANCE-DATETIME not found")
            continue
        hdr_et = datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=ET)
        as_utc = edgar.parse_acceptance(row["acceptanceDateTime"], "utc")
        as_et = edgar.parse_acceptance(row["acceptanceDateTime"], "et")
        v = "utc" if as_utc == hdr_et else "et" if as_et == hdr_et else "neither"
        verdicts[v] += 1
        r.text(f"  - {acc}: json={row['acceptanceDateTime']} header(ET)={hdr_et.isoformat()} -> {v}")
    r.check(
        "MEASURE" if len(verdicts) == 1 and "neither" not in verdicts else "FAIL",
        "acceptanceDateTime timezone -> configs/sources.yaml edgar.acceptance_tz",
        dict(verdicts).__repr__(),
    )


# --------------------------------------------------------------------------- Finnhub + cross-source similarity


def probe_finnhub(r: Report, token: str) -> dict[str, list[dict[str, Any]]]:
    r.section("Finnhub company news")
    client = finnhub.make_client(token)
    end = datetime.now(UTC).date()
    out: dict[str, list[dict[str, Any]]] = {}
    for t in PROBE_TICKERS:
        rows = r.guard(f"finnhub {t}", lambda t=t: finnhub.company_news(client, t, end - timedelta(days=7), end))
        out[t] = rows or []
    save("finnhub_7d", out)
    n = sum(len(v) for v in out.values())
    r.check("PASS" if n else "FAIL", "articles last 7 days across probe tickers", str(n))
    old = r.guard(
        "finnhub history",
        lambda: finnhub.company_news(client, "AAPL", end - timedelta(days=400), end - timedelta(days=393)),
    )
    if old is not None:
        r.check("MEASURE", "AAPL articles ~13 months ago (free-tier history limit)", str(len(old)))
    if n:
        r.check("MEASURE", "vendors", str(Counter(a.get("source") for v in out.values() for a in v).most_common(8)))
    return out


def probe_similarity(r: Report, key: str, secret: str, fh: dict[str, list[dict[str, Any]]]) -> None:
    r.section("Cross-source near-duplicate study (headline + summary, same ticker, within 48h)")
    client = alpaca_news.make_client(key, secret)
    end = datetime.now(UTC)
    al = list(alpaca_news.iter_raw(client, end - timedelta(days=7), end, symbols=PROBE_TICKERS, include_content=False))
    docs: list[tuple[str, str, datetime, str]] = []  # (source, ticker, ts, text)
    for a in al:
        for t in a.get("symbols") or []:
            if t in PROBE_TICKERS:
                docs.append(("alpaca", t, parse_rfc3339(a["created_at"]), f"{a['headline']} {a.get('summary') or ''}"))
    for t, rows in fh.items():
        for a in rows:
            docs.append(("finnhub", t, parse_epoch(a["datetime"]), f"{a['headline']} {a.get('summary') or ''}"))
    sh = [shingles(d[3]) for d in docs]
    bands: dict[str, list[tuple[float, int, int]]] = {
        k: [] for k in ("0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0")
    }
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            a, b = docs[i], docs[j]
            if a[0] == b[0] or a[1] != b[1] or abs((a[2] - b[2]).total_seconds()) > 48 * 3600:
                continue
            s = jaccard(sh[i], sh[j])
            key_ = list(bands)[min(int(s / 0.2), 4)]
            bands[key_].append((s, i, j))
    for k, v in bands.items():
        r.text(f"- {k}: {len(v)} pairs")
    r.text("\nExamples per band (read these and choose `dedupe.jaccard_threshold`):\n")
    for k, v in bands.items():
        if k == "0.0-0.2":
            continue
        for s, i, j in sorted(v, reverse=True)[:3]:
            a, b = docs[i], docs[j]
            lead = "alpaca" if a[2] < b[2] else "finnhub"
            r.text(f"- [{k}] J={s:.2f} ({a[1]}, {lead} first by {abs((a[2] - b[2]).total_seconds()) / 60:.0f} min)")
            r.text(f"  - {a[0]}: {a[3][:160]}")
            r.text(f"  - {b[0]}: {b[3][:160]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws-seconds", type=int, default=0, help="also sample the live news WebSocket (market hours)")
    args = ap.parse_args()
    st = load_settings()
    r = Report()
    r.text(f"# Data probe report\n\nRun at {datetime.now(UTC).isoformat()} by scripts/probe_apis.py.")
    if st.alpaca_key and st.alpaca_secret:
        r.guard("alpaca news", lambda: probe_alpaca_news(r, st.alpaca_key, st.alpaca_secret))
        if args.ws_seconds:
            r.guard("alpaca ws", lambda: probe_alpaca_ws(r, st.alpaca_key, st.alpaca_secret, args.ws_seconds))
        r.guard("alpaca bars", lambda: probe_bars(r, st.alpaca_key, st.alpaca_secret))
    else:
        r.check("FAIL", "Alpaca", "ALPACA_API_KEY / ALPACA_SECRET_KEY not set")
    if st.sec_user_agent:
        r.guard("edgar", lambda: probe_edgar(r, st.sec_user_agent))
    else:
        r.check("FAIL", "EDGAR", "SEC_USER_AGENT not set")
    fh: dict[str, list[dict[str, Any]]] = {}
    if st.finnhub_key:
        fh = r.guard("finnhub", lambda: probe_finnhub(r, st.finnhub_key)) or {}
    else:
        r.check("FAIL", "Finnhub", "FINNHUB_API_KEY not set")
    if st.alpaca_key and st.alpaca_secret and fh:
        r.guard("similarity", lambda: probe_similarity(r, st.alpaca_key, st.alpaca_secret, fh))
    r.lines.insert(2, f"\n**Summary:** {dict(r.status)}\n")
    Path("docs/data-probe-report.md").write_text("\n".join(r.lines) + "\n")
    print(f"\nwrote docs/data-probe-report.md  {dict(r.status)}")


if __name__ == "__main__":
    try:
        main()
    except httpx.ConnectError as exc:
        raise SystemExit(f"network error: {exc}") from exc
