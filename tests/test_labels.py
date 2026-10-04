"""Label correctness: timing, regular-hours prices, and no use of future data."""

import math
from datetime import date, datetime, timedelta

import polars as pl
import pytest

from ws import calendar as cal
from ws.labels import assign_cycles, cycle_table, daily_closes, label, samples, trailing_beta
from ws.timeutil import ET, UTC

SESSIONS = [
    date(2025, 3, 10),
    date(2025, 3, 11),
    date(2025, 3, 12),
    date(2025, 3, 13),
    date(2025, 3, 14),
    date(2025, 3, 17),
]


def price(symbol: str, ts: datetime) -> float:
    """Deterministic, strictly increasing intraday path; ACME moves twice as fast as SPY."""
    minutes = (ts - datetime(2025, 3, 10, tzinfo=UTC)).total_seconds() / 60
    return 100.0 + (0.002 if symbol == "SPY" else 0.004) * minutes


def make_bars(symbols=("SPY", "ACME"), extended=True, override=None) -> pl.DataFrame:
    rows = []
    for d in SESSIONS:
        start = datetime.combine(d, datetime.min.time(), tzinfo=ET).replace(
            hour=4 if extended else 9, minute=0 if extended else 30
        )
        end = datetime.combine(d, datetime.min.time(), tzinfo=ET).replace(hour=20 if extended else 16)
        t = start
        while t < end:
            tu = t.astimezone(UTC)
            for s in symbols:
                in_session = cal.session_open(d) <= tu and tu + timedelta(minutes=15) <= cal.session_close(d)
                c = price(s, tu + timedelta(minutes=15)) if in_session else 9999.0  # junk outside RTH
                if override:
                    c = override(s, tu, c)
                rows.append({"symbol": s, "ts": tu, "open": c, "high": c, "low": c, "close": c, "volume": 1.0})
            t += timedelta(minutes=15)
    return pl.DataFrame(rows, schema_overrides={"ts": pl.Datetime("us", "UTC")})


def events(*items) -> pl.DataFrame:
    return pl.DataFrame(
        [{"event_id": i, "first_seen_ts": ts.astimezone(UTC), "tickers": t} for i, ts, t in items],
        schema={"event_id": pl.Utf8, "first_seen_ts": pl.Datetime("us", "UTC"), "tickers": pl.List(pl.Utf8)},
    )


def build(ev: pl.DataFrame, bars: pl.DataFrame, **kw) -> pl.DataFrame:
    cyc = cycle_table(SESSIONS[0], SESSIONS[-1])
    return label(samples(assign_cycles(ev, cyc)), bars, horizons=(1, 2), **kw)


def test_entry_and_exit_prices_and_returns():
    ev = events(("a", datetime(2025, 3, 10, 8, 0, tzinfo=ET), ["ACME"]))
    out = build(ev, make_bars(), beta_window=None)
    row = out.row(0, named=True)
    entry_ts = datetime(2025, 3, 10, 9, 45, tzinfo=ET).astimezone(UTC)
    exit_ts = datetime(2025, 3, 11, 15, 30, tzinfo=ET).astimezone(UTC)
    assert row["cycle_ts"] == entry_ts and row["exit_ts_1"] == exit_ts
    assert row["entry_px"] == pytest.approx(price("ACME", entry_ts))
    assert row["exit_px_1"] == pytest.approx(price("ACME", exit_ts))
    exp_ret = price("ACME", exit_ts) / price("ACME", entry_ts) - 1
    exp_mret = price("SPY", exit_ts) / price("SPY", entry_ts) - 1
    assert row["ret_1"] == pytest.approx(exp_ret)
    assert row["abret_1"] == pytest.approx(exp_ret - exp_mret)


def test_events_before_same_cycle_collapse_into_one_sample():
    ev = events(
        ("a", datetime(2025, 3, 10, 7, 0, tzinfo=ET), ["ACME"]),
        ("b", datetime(2025, 3, 10, 9, 30, tzinfo=ET), ["ACME"]),
        ("c", datetime(2025, 3, 10, 9, 50, tzinfo=ET), ["ACME"]),  # next cycle
    )
    out = build(ev, make_bars(), beta_window=None)
    assert out.height == 2
    assert sorted(out.row(0, named=True)["event_ids"]) == ["a", "b"]


def test_extended_hours_prices_never_used():
    ev = events(("a", datetime(2025, 3, 10, 8, 0, tzinfo=ET), ["ACME"]))
    out = build(ev, make_bars(), beta_window=None)
    for col in ("entry_px", "exit_px_1", "exit_px_2", "mkt_entry_px", "mkt_exit_px_1"):
        assert out[col].max() < 1000


def test_labels_ignore_data_after_exit_and_entry_ignores_data_after_cycle():
    ev = events(("a", datetime(2025, 3, 10, 8, 0, tzinfo=ET), ["ACME"]))
    base = build(ev, make_bars(), beta_window=None)
    exit2 = base["exit_ts_2"].item()
    poisoned = make_bars(override=lambda s, ts, c: c * 50 if ts + timedelta(minutes=15) > exit2 else c)
    after = build(ev, poisoned, beta_window=None)
    for col in ("entry_px", "exit_px_1", "exit_px_2", "abret_1", "abret_2"):
        assert after[col].item() == pytest.approx(base[col].item())
    cycle = base["cycle_ts"].item()
    poisoned = make_bars(override=lambda s, ts, c: c * 50 if ts + timedelta(minutes=15) > cycle else c)
    assert build(ev, poisoned, beta_window=None)["entry_px"].item() == pytest.approx(base["entry_px"].item())


def test_trailing_beta_uses_only_prior_sessions():
    daily = daily_closes(make_bars())
    betas = trailing_beta(daily, window=3, min_obs=2).filter(pl.col("symbol") == "ACME").sort("session")
    target = SESSIONS[4]
    b = betas.filter(pl.col("session") == target)["beta"].item()
    assert b is not None and not math.isnan(b)
    # Wreck the target session's own prices: its beta must not change.
    bumped = make_bars(override=lambda s, ts, c: c * (3 if s == "ACME" and ts.astimezone(ET).date() == target else 1))
    b2 = (
        trailing_beta(daily_closes(bumped), window=3, min_obs=2)
        .filter((pl.col("symbol") == "ACME") & (pl.col("session") == target))["beta"]
        .item()
    )
    assert b2 == pytest.approx(b)
