"""Shared momentum, volatility, and liquidity calculations for pipeline v0."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite, log, sqrt
from statistics import mean, stdev

import polars as pl

MIN_COST_HISTORY_RETURNS = 21


def _groups(bars: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return each symbol's bars ordered by event time."""
    required = {"symbol", "event_time", "close", "volume"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"bars are missing columns: {sorted(missing)}")
    groups: dict[str, pl.DataFrame] = {}
    for raw_symbol, group in bars.group_by("symbol", maintain_order=False):
        symbol = raw_symbol[0] if isinstance(raw_symbol, tuple) else raw_symbol
        groups[str(symbol)] = group.sort("event_time")
    return groups


def momentum_12_1(bars: pl.DataFrame) -> pl.DataFrame:
    """Calculate close 21 sessions ago divided by close 252 sessions ago minus one."""
    rows: list[dict[str, str | float | None]] = []
    for symbol, group in _groups(bars).items():
        closes = [float(value) for value in group.get_column("close").to_list()]
        value = closes[-22] / closes[-253] - 1.0 if len(closes) >= 253 else None
        rows.append({"symbol": symbol, "mom_12_1": value})
    return pl.DataFrame(rows, schema={"symbol": pl.String, "mom_12_1": pl.Float64})


def realized_vol_21d(bars: pl.DataFrame) -> pl.DataFrame:
    """Calculate annualized sample standard deviation of each symbol's last 21 log returns."""
    rows: list[dict[str, str | float | None]] = []
    for symbol, group in _groups(bars).items():
        closes = [float(value) for value in group.get_column("close").to_list()]
        if len(closes) >= MIN_COST_HISTORY_RETURNS + 1:
            values = [
                log(closes[index] / closes[index - 1])
                for index in range(len(closes) - MIN_COST_HISTORY_RETURNS, len(closes))
            ]
            volatility = stdev(values) * sqrt(252)
        else:
            volatility = None
        rows.append({"symbol": symbol, "vol_21d": volatility})
    return pl.DataFrame(rows, schema={"symbol": pl.String, "vol_21d": pl.Float64})


def adv_dollars_20d(bars: pl.DataFrame) -> pl.DataFrame:
    """Calculate each symbol's mean close times volume over its last 20 sessions."""
    rows: list[dict[str, str | float | None]] = []
    for symbol, group in _groups(bars).items():
        closes = group.get_column("close").tail(20)
        volumes = group.get_column("volume").tail(20)
        notionals = [float(close) * float(volume) for close, volume in zip(closes, volumes, strict=True)]
        adv = mean(notionals) if len(notionals) == 20 else None
        rows.append({"symbol": symbol, "adv_dollars_20d": adv})
    return pl.DataFrame(rows, schema={"symbol": pl.String, "adv_dollars_20d": pl.Float64})


def cost_history_inputs(bars: pl.DataFrame) -> dict[str, tuple[float, float]]:
    """Return valid (20-session ADV, annualized 21-return volatility) pairs.

    Twenty-one close-to-close returns require 22 daily bars. A close-time decision
    using these bars can therefore be costed at the next session's open without
    relying on the fill-day bar.
    """
    if bars.is_empty():
        return {}
    adv = {
        row["symbol"]: row["adv_dollars_20d"]
        for row in adv_dollars_20d(bars).iter_rows(named=True)
    }
    volatility = {
        row["symbol"]: row["vol_21d"]
        for row in realized_vol_21d(bars).iter_rows(named=True)
    }
    inputs: dict[str, tuple[float, float]] = {}
    for symbol, annual_volatility in volatility.items():
        average_dollars = adv.get(symbol)
        if annual_volatility is None or average_dollars is None:
            continue
        average_dollars = float(average_dollars)
        annual_volatility = float(annual_volatility)
        if (
            not isfinite(average_dollars)
            or average_dollars <= 0
            or not isfinite(annual_volatility)
            or annual_volatility < 0
        ):
            continue
        inputs[symbol] = (average_dollars, annual_volatility)
    return inputs


def portfolio_vol_21d(bars: pl.DataFrame, weights: Mapping[str, float]) -> float:
    """Return annualized sample volatility for weighted daily portfolio log returns."""
    if not weights:
        return 0.0
    groups = _groups(bars)
    normalized = {symbol.strip().upper(): float(weight) for symbol, weight in weights.items() if float(weight) > 0}
    if not normalized:
        return 0.0
    if any(not isfinite(weight) for weight in normalized.values()):
        raise ValueError("portfolio weights must be finite")
    if any(symbol not in groups for symbol in normalized):
        return 0.0

    closes = {
        symbol: {
            event_time: float(price)
            for event_time, price in zip(group.get_column("event_time"), group.get_column("close"), strict=True)
        }
        for symbol, group in groups.items()
        if symbol in normalized
    }
    sessions = sorted(set.intersection(*(set(prices) for prices in closes.values())))
    if len(sessions) < 22:
        return 0.0
    returns = [
        sum(weight * log(closes[symbol][current] / closes[symbol][prior]) for symbol, weight in normalized.items())
        for prior, current in zip(sessions[-22:-1], sessions[-21:], strict=True)
    ]
    return stdev(returns) * sqrt(252)
