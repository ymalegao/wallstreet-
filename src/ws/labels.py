"""Forward-return labels that match exactly what the bot could have traded.

Evaluation unit: one (ticker, decision cycle) pair. Every event mentioning a ticker that becomes
actionable before the same cycle is folded into one sample, so syndicated copies and follow-up
stories cannot inflate the sample count or the t-statistics.

For a sample at decision cycle C (see calendar.py) and holding period h sessions:
  entry price  = close of the last 15-min bar ending at or before C (and no more than ``tolerance`` earlier)
  exit price   = same rule at the afternoon cycle of the session h sessions after C's session
  raw return   = exit / entry - 1
  abnormal     = raw - beta * SPY return over the identical interval
  beta         = trailing daily-return beta using only sessions strictly before C's session
                 (beta = 1 when ``beta_window`` is None: market-adjusted returns)

Nothing in a label may depend on information after its exit timestamp, and no feature may depend
on information after C. tests/test_labels.py checks both.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl

from ws import calendar as cal
from ws.timeutil import ET

BAR_LEN = timedelta(minutes=15)
MARKET = "SPY"


def cycle_table(start: date, end: date) -> pl.DataFrame:
    """All decision cycles between two dates: columns cycle_ts, session, slot (0 = morning, 1 = afternoon)."""
    rows = []
    d = start if cal.is_session(start) else cal.next_session(start)
    while d <= end:
        m, a = cal.cycles_for_session(d)
        rows += [{"cycle_ts": m, "session": d, "slot": 0}, {"cycle_ts": a, "session": d, "slot": 1}]
        d = cal.next_session(d)
    return pl.DataFrame(rows, schema={"cycle_ts": pl.Datetime("us", "UTC"), "session": pl.Date, "slot": pl.Int8})


def assign_cycles(events: pl.DataFrame, cycles: pl.DataFrame, latency: timedelta = cal.DEFAULT_LATENCY) -> pl.DataFrame:
    """Explode events to (event_id, ticker) and attach the first cycle at or after first_seen_ts + latency."""
    ev = (
        events.select("event_id", "first_seen_ts", "tickers")
        .explode("tickers", empty_as_null=True)
        .rename({"tickers": "ticker"})
        .drop_nulls("ticker")
        .with_columns(ready_ts=pl.col("first_seen_ts") + latency)
        .sort("ready_ts")
    )
    return ev.join_asof(cycles.sort("cycle_ts"), left_on="ready_ts", right_on="cycle_ts", strategy="forward")


def samples(assigned: pl.DataFrame) -> pl.DataFrame:
    """Collapse to one row per (ticker, cycle) with the list of contributing events."""
    return (
        assigned.drop_nulls("cycle_ts")
        .group_by("ticker", "cycle_ts", "session", "slot")
        .agg(pl.col("event_id").alias("event_ids"), pl.col("first_seen_ts").max().alias("last_event_ts"))
        .sort("cycle_ts", "ticker")
    )


def _price_at(bars: pl.DataFrame, keys: pl.DataFrame, ts_col: str, out: str, tolerance: timedelta) -> pl.DataFrame:
    """For each (symbol, ts) in keys: close of the last bar ending at or before ts, within tolerance."""
    b = bars.select(pl.col("symbol"), (pl.col("ts") + BAR_LEN).alias("bar_end"), pl.col("close").alias(out)).sort(
        "bar_end"
    )
    return (
        keys.sort(ts_col)
        .join_asof(
            b,
            left_on=ts_col,
            right_on="bar_end",
            by="symbol",
            strategy="backward",
            tolerance=tolerance,
            check_sortedness=False,  # globally sorted by time, hence sorted within every symbol
        )
        .drop("bar_end")
    )


def regular_hours(bars: pl.DataFrame) -> pl.DataFrame:
    """Keep only bars fully inside a regular NYSE session (drops pre/post-market and holiday bars)."""
    if bars.is_empty():
        return bars
    days = bars.select(pl.col("ts").dt.convert_time_zone(str(ET)).dt.date().unique().alias("session"))
    hours = [
        {"session": d, "sess_open": cal.session_open(d), "sess_close": cal.session_close(d)}
        for d in days["session"].to_list()
        if cal.is_session(d)
    ]
    if not hours:
        return bars.clear()
    sess = pl.DataFrame(
        hours,
        schema={"session": pl.Date, "sess_open": pl.Datetime("us", "UTC"), "sess_close": pl.Datetime("us", "UTC")},
    )
    return (
        bars.with_columns(session=pl.col("ts").dt.convert_time_zone(str(ET)).dt.date())
        .join(sess, on="session", how="inner")
        .filter((pl.col("ts") >= pl.col("sess_open")) & (pl.col("ts") + BAR_LEN <= pl.col("sess_close")))
        .drop("session", "sess_open", "sess_close")
    )


def daily_closes(bars: pl.DataFrame) -> pl.DataFrame:
    """Session close per symbol (last regular-hours bar of each session), with its daily return."""
    return (
        regular_hours(bars)
        .with_columns(session=pl.col("ts").dt.convert_time_zone(str(ET)).dt.date())
        .sort("ts")
        .group_by("symbol", "session")
        .agg(pl.col("close").last())
        .sort("symbol", "session")
        .with_columns(ret=pl.col("close").pct_change().over("symbol"))
    )


def trailing_beta(daily: pl.DataFrame, window: int, min_obs: int | None = None) -> pl.DataFrame:
    """Beta of each symbol vs SPY using returns from sessions strictly before each session."""
    min_obs = min_obs or window // 2
    mkt = daily.filter(pl.col("symbol") == MARKET).select("session", pl.col("ret").alias("mret"))
    d = daily.join(mkt, on="session", how="inner").sort("symbol", "session")

    def rmean(e: pl.Expr) -> pl.Expr:
        return e.rolling_mean(window_size=window, min_samples=min_obs)

    d = d.with_columns(
        xy=pl.col("ret") * pl.col("mret"),
        xx=pl.col("mret") ** 2,
    ).with_columns(
        cov=(rmean(pl.col("xy")) - rmean(pl.col("ret")) * rmean(pl.col("mret"))).over("symbol"),
        var=(rmean(pl.col("xx")) - rmean(pl.col("mret")) ** 2).over("symbol"),
    )
    # shift(1): beta used on session t is estimated through session t-1 only.
    return d.select("symbol", "session", (pl.col("cov") / pl.col("var")).shift(1).over("symbol").alias("beta"))


def label(
    samples_df: pl.DataFrame,
    bars: pl.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 10),
    beta_window: int | None = 252,
    tolerance: timedelta = timedelta(minutes=30),
) -> pl.DataFrame:
    """Attach entry price, and per horizon h: exit timestamp, raw, market and abnormal returns."""
    bars = regular_hours(bars)
    keys = samples_df.with_columns(symbol=pl.col("ticker"))
    out = _price_at(bars, keys, "cycle_ts", "entry_px", tolerance)
    mkt_keys = out.select(pl.lit(MARKET).alias("symbol"), "cycle_ts").unique()
    out = out.join(
        _price_at(bars, mkt_keys, "cycle_ts", "mkt_entry_px", tolerance).drop("symbol"), on="cycle_ts", how="left"
    )

    if beta_window:
        betas = trailing_beta(daily_closes(bars), beta_window)
        out = out.join(betas, on=["symbol", "session"], how="left")
    else:
        out = out.with_columns(beta=pl.lit(1.0))

    for h in horizons:
        exits = {s: cal.cycles_for_session(cal.add_sessions(s, h))[1] for s in out["session"].unique().to_list()}
        exit_col = f"exit_ts_{h}"
        out = out.with_columns(
            pl.col("session").replace_strict(exits, return_dtype=pl.Datetime("us", "UTC")).alias(exit_col)
        )
        out = _price_at(bars, out, exit_col, f"exit_px_{h}", tolerance)
        mk = out.select(pl.lit(MARKET).alias("symbol"), pl.col(exit_col)).unique()
        out = out.join(
            _price_at(bars, mk, exit_col, f"mkt_exit_px_{h}", tolerance).drop("symbol"), on=exit_col, how="left"
        )
        out = out.with_columns(
            (pl.col(f"exit_px_{h}") / pl.col("entry_px") - 1).alias(f"ret_{h}"),
            (pl.col(f"mkt_exit_px_{h}") / pl.col("mkt_entry_px") - 1).alias(f"mret_{h}"),
        ).with_columns((pl.col(f"ret_{h}") - pl.col("beta") * pl.col(f"mret_{h}")).alias(f"abret_{h}"))
    return out.drop("symbol").sort("cycle_ts", "ticker")


def as_of_ok(df: pl.DataFrame, feature_ts_col: str) -> bool:
    """True if every feature timestamp is at or before its decision cycle (no look-ahead)."""
    return bool(df.select((pl.col(feature_ts_col) <= pl.col("cycle_ts")).all()).item())


def first_cycle_after(ts: datetime, latency: timedelta = cal.DEFAULT_LATENCY) -> datetime:
    return cal.next_decision_cycle(ts, latency)
