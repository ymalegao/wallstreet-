"""Point-in-time tradable universe, built from our own bars (no hindsight index membership).

A ticker is eligible on session t if, using only sessions before t, its last close >= $5 and its
20-session median dollar volume >= $20M. Survivorship-free provided bars exist for delisted names
(checked by the API probe).
"""

from __future__ import annotations

import polars as pl

MIN_PRICE = 5.0
MIN_DOLLAR_VOLUME = 20e6
WINDOW = 20


def eligibility(daily: pl.DataFrame) -> pl.DataFrame:
    """Per-session eligibility flags.

    ``daily``: symbol, ts (session bar start), close, volume. Use UNADJUSTED bars: split-adjusted
    history understates past prices and would wrongly fail the $5 filter.
    """
    d = daily.sort("symbol", "ts").with_columns(
        session=pl.col("ts").dt.date(),
        dv=pl.col("close") * pl.col("volume"),
    )
    d = d.with_columns(
        prev_close=pl.col("close").shift(1).over("symbol"),
        med_dv=pl.col("dv").rolling_median(WINDOW, min_samples=WINDOW).shift(1).over("symbol"),
    )
    return d.select(
        "symbol",
        "session",
        ((pl.col("prev_close") >= MIN_PRICE) & (pl.col("med_dv") >= MIN_DOLLAR_VOLUME))
        .fill_null(False)
        .alias("eligible"),
    )


def ever_eligible(daily: pl.DataFrame) -> list[str]:
    e = eligibility(daily)
    return sorted(e.filter(pl.col("eligible"))["symbol"].unique().to_list())
