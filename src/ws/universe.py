"""Point-in-time tradable universe, built from our own bars without index membership lists.

A ticker is eligible on session t if, using only sessions before t, its last close >= $5 and its
20-session median dollar volume >= $20M. The selector avoids index-membership hindsight, but its
survivorship coverage still depends on the candidate master and historical bars available for
inactive and delisted names.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

import polars as pl

EXCHANGES = {"NASDAQ", "NYSE", "AMEX"}
SYMBOL_PATTERN = re.compile(r"^[A-Z]{1,5}(?:[.-][A-Z0-9]{1,3})?$")
EXCLUDED_NAME_TERMS = (
    " ETF",
    "ETF ",
    "ETN",
    "EXCHANGE TRADED",
    "EXCHANGE-TRADED",
    " FUND",
    " FUNDS",
    "MUTUAL FUND",
    "PROSHARES",
    "DIREXION",
    "ISHARES",
    "VANGUARD",
    "SPDR",
    "WISDOMTREE",
    "FIRST TRUST",
    "INVESCO QQQ TRUST",
    "ULTRA",
    "LEVERAGED",
    "INVERSE",
    "2X",
    "3X",
    "4X",
    "5X",
    "WARRANT",
    "WARRANTS",
    " UNIT",
    " UNITS",
    "RIGHTS",
    " PREFERRED",
    "PREFERENCE SHARES",
    "DEPOSITARY SHARES",
    "NOTES DUE",
)

MIN_PRICE = 5.0
MIN_DOLLAR_VOLUME = 20e6
WINDOW = 20


def listed_stock_candidates(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select a broad listed-stock proxy; retain ordinary share classes and ADRs."""
    candidates = []
    for asset in assets:
        symbol = str(asset.get("symbol") or "").strip().upper()
        name = str(asset.get("name") or "").upper()
        if not SYMBOL_PATTERN.fullmatch(symbol) or asset.get("exchange") not in EXCHANGES:
            continue
        if any(term in name for term in EXCLUDED_NAME_TERMS):
            continue
        candidates.append(asset)
    # Prefer the active record where Alpaca exposes the same symbol in both endpoints.
    candidates.sort(key=lambda asset: (asset["symbol"], asset.get("status") == "active"))
    return sorted({a["symbol"]: a for a in candidates}.values(), key=lambda a: a["symbol"])


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


def point_in_time_top_n(
    daily: pl.DataFrame,
    candidate_symbols: list[str],
    start: date,
    end: date,
    n: int = 25,
) -> pl.DataFrame:
    """Select the most liquid eligible stocks per session using only prior daily bars.

    Dates use a half-open ``[start, end)`` interval. Caller supplies a broad stock-only
    candidate pool; the returned rank is point-in-time with respect to price and liquidity.
    """
    if n < 1:
        raise ValueError("n must be positive")
    if not candidate_symbols:
        raise ValueError("candidate_symbols cannot be empty")
    d = daily.filter(pl.col("symbol").is_in(candidate_symbols)).sort("symbol", "ts")
    d = d.with_columns(session=pl.col("ts").dt.date(), dv=pl.col("close") * pl.col("volume"))
    d = d.with_columns(
        prev_close=pl.col("close").shift(1).over("symbol"),
        med_dv=pl.col("dv").rolling_median(WINDOW, min_samples=WINDOW).shift(1).over("symbol"),
    )
    selected = (
        d.filter(
            (pl.col("session") >= start)
            & (pl.col("session") < end)
            & (pl.col("prev_close") >= MIN_PRICE)
            & (pl.col("med_dv") >= MIN_DOLLAR_VOLUME)
        )
        .sort(["session", "med_dv", "symbol"], descending=[False, True, False])
        .group_by("session", maintain_order=True)
        .head(n)
        .select("session", "symbol", "prev_close", "med_dv")
        .sort("session", "med_dv", "symbol", descending=[False, True, False])
    )
    return selected
