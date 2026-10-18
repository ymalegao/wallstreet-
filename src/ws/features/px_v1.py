"""Price and volume feature set shared by training and pipeline inference."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from math import isfinite, log, sqrt
from statistics import mean, stdev
from typing import Any

import polars as pl

from ws.pipeline_v0.metrics import adv_dollars_20d, momentum_12_1, realized_vol_21d

PX_V1_FEATURES = (
    "ret_1d",
    "ret_5d",
    "ret_21d",
    "ret_63d",
    "ret_126d",
    "mom_12_1",
    "vol_21d",
    "vol_63d",
    "vol_ratio",
    "dist_ma_20",
    "dist_ma_200",
    "dist_high_252",
    "adv_dollars_20d",
    "volume_ratio",
    "range_21d",
    "beta_252",
)
PX_V1_RAW_FEATURES = (*PX_V1_FEATURES, "adv_dollars_20d_raw")
_WINDOW_RETURNS = {"ret_1d": 1, "ret_5d": 5, "ret_21d": 21, "ret_63d": 63, "ret_126d": 126}


def compute_px_v1(bars: pl.DataFrame, spy_bars: pl.DataFrame) -> pl.DataFrame:
    """Compute one latest-visible, cross-sectionally ranked feature row per symbol.

    Callers enforce ``available_at <= tick`` before passing bars. ``spy_bars`` is kept
    separate from the tradable universe and is used only for the market beta feature.
    """
    _validate_bars(bars, require_high=True)
    _validate_bars(spy_bars, require_high=False)
    if bars.is_empty():
        return _empty_frame()

    groups = _groups(bars)
    spy_group = _groups(spy_bars).get("SPY")
    momentum = _metric_lookup(momentum_12_1(bars), "mom_12_1")
    volatility_21 = _metric_lookup(realized_vol_21d(bars), "vol_21d")
    adv = _metric_lookup(adv_dollars_20d(bars), "adv_dollars_20d")

    rows: list[dict[str, str | float | None]] = []
    for symbol in sorted(groups):
        group = groups[symbol]
        closes = _numbers(group, "close")
        highs = _numbers(group, "high")
        lows = _numbers(group, "low")
        volumes = _numbers(group, "volume")
        vol_21 = volatility_21.get(symbol)
        vol_63 = _annualized_volatility(closes, 63)
        raw_adv = adv.get(symbol)
        row: dict[str, str | float | None] = {
            "symbol": symbol,
            **{name: _return(closes, sessions) for name, sessions in _WINDOW_RETURNS.items()},
            "mom_12_1": momentum.get(symbol),
            "vol_21d": vol_21,
            "vol_63d": vol_63,
            "vol_ratio": _ratio(vol_21, vol_63),
            "dist_ma_20": _distance_from_mean(closes, 20),
            "dist_ma_200": _distance_from_mean(closes, 200),
            "dist_high_252": _distance_from_high(closes, highs, 252),
            "adv_dollars_20d_raw": raw_adv,
            "adv_dollars_20d": raw_adv,
            "volume_ratio": _volume_ratio(volumes, 5, 60),
            "range_21d": _mean_range(highs, lows, closes, 21),
            "beta_252": _beta_252(group, spy_group),
        }
        rows.append(row)

    ranked = pl.DataFrame(rows, schema=_schema()).sort("symbol")
    for name in PX_V1_FEATURES:
        values = ranked.get_column(name).to_list()
        ranked = ranked.with_columns(pl.Series(name, _percentile_ranks(values), dtype=pl.Float64))
    return ranked.select(["symbol", *PX_V1_FEATURES, "adv_dollars_20d_raw"])


def _validate_bars(bars: pl.DataFrame, *, require_high: bool) -> None:
    required = {"symbol", "event_time", "close", "volume"}
    if require_high:
        required |= {"high", "low"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"bars are missing columns: {sorted(missing)}")


def _groups(bars: pl.DataFrame) -> dict[str, pl.DataFrame]:
    groups: dict[str, pl.DataFrame] = {}
    for raw_symbol, group in bars.group_by("symbol", maintain_order=False):
        symbol = raw_symbol[0] if isinstance(raw_symbol, tuple) else raw_symbol
        groups[str(symbol).strip().upper()] = group.sort("event_time")
    return groups


def _metric_lookup(frame: pl.DataFrame, value_column: str) -> dict[str, float | None]:
    return {
        str(row["symbol"]): float(row[value_column]) if row[value_column] is not None else None
        for row in frame.iter_rows(named=True)
    }


def _numbers(group: pl.DataFrame, column: str) -> list[float]:
    return [float(value) for value in group.get_column(column).to_list()]


def _return(closes: Sequence[float], sessions: int) -> float | None:
    if len(closes) <= sessions or closes[-sessions - 1] == 0:
        return None
    return closes[-1] / closes[-sessions - 1] - 1.0


def _annualized_volatility(closes: Sequence[float], sessions: int) -> float | None:
    if len(closes) <= sessions:
        return None
    returns = [log(closes[index] / closes[index - 1]) for index in range(len(closes) - sessions, len(closes))]
    return stdev(returns) * sqrt(252)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    value = numerator / denominator
    return value if isfinite(value) else None


def _distance_from_mean(closes: Sequence[float], window: int) -> float | None:
    if len(closes) < window:
        return None
    average = mean(closes[-window:])
    return closes[-1] / average - 1.0 if average else None


def _distance_from_high(closes: Sequence[float], highs: Sequence[float], window: int) -> float | None:
    if len(closes) < window or len(highs) < window:
        return None
    maximum = max(highs[-window:])
    return closes[-1] / maximum - 1.0 if maximum else None


def _volume_ratio(volumes: Sequence[float], short_window: int, long_window: int) -> float | None:
    if len(volumes) < long_window:
        return None
    long_mean = mean(volumes[-long_window:])
    return mean(volumes[-short_window:]) / long_mean if long_mean else None


def _mean_range(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], window: int) -> float | None:
    if min(len(highs), len(lows), len(closes)) < window:
        return None
    values = [
        (high - low) / close
        for high, low, close in zip(highs[-window:], lows[-window:], closes[-window:], strict=True)
    ]
    return mean(values)


def _beta_252(symbol_bars: pl.DataFrame, spy_bars: pl.DataFrame | None) -> float | None:
    if spy_bars is None:
        return None
    symbol_closes = {
        timestamp: float(close)
        for timestamp, close in zip(symbol_bars.get_column("event_time"), symbol_bars.get_column("close"), strict=True)
    }
    spy_closes = {
        timestamp: float(close)
        for timestamp, close in zip(spy_bars.get_column("event_time"), spy_bars.get_column("close"), strict=True)
    }
    sessions = sorted(spy_closes)
    if len(sessions) < 253:
        return None
    paired_returns: list[tuple[float, float]] = []
    beta_sessions = sessions[-253:]
    for previous, current in pairwise(beta_sessions):
        if previous not in symbol_closes or current not in symbol_closes:
            return None
        paired_returns.append(
            (log(symbol_closes[current] / symbol_closes[previous]), log(spy_closes[current] / spy_closes[previous]))
        )
    if len(paired_returns) < 252:
        return None
    asset_returns = [pair[0] for pair in paired_returns]
    market_returns = [pair[1] for pair in paired_returns]
    market_mean = mean(market_returns)
    asset_mean = mean(asset_returns)
    market_variance = sum((value - market_mean) ** 2 for value in market_returns)
    if market_variance == 0:
        return None
    covariance = sum(
        (asset - asset_mean) * (market - market_mean)
        for asset, market in paired_returns
    )
    beta = covariance / market_variance
    return beta if isfinite(beta) else None


def _percentile_ranks(values: Sequence[float | None]) -> list[float | None]:
    """Map average one-based ranks to [0, 1], preserving nulls and ties."""
    valid = [
        (index, float(value))
        for index, value in enumerate(values)
        if value is not None and isfinite(float(value))
    ]
    if not valid:
        return [None] * len(values)
    valid.sort(key=lambda item: item[1])
    result: list[float | None] = [None] * len(values)
    count = len(valid)
    index = 0
    while index < count:
        end = index + 1
        while end < count and valid[end][1] == valid[index][1]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        percentile = (average_rank - 1.0) / (count - 1.0) if count > 1 else 0.0
        for source_index, _ in valid[index:end]:
            result[source_index] = percentile
        index = end
    return result


def _schema() -> dict[str, Any]:
    return {"symbol": pl.String, **{name: pl.Float64 for name in PX_V1_RAW_FEATURES}}


def _empty_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=_schema()).select(["symbol", *PX_V1_FEATURES, "adv_dollars_20d_raw"])
