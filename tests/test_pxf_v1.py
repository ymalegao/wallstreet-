from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from math import log
from statistics import mean, stdev

import polars as pl
import pytest

from ws.features.pxf_v1 import (
    _fundamental_features,
    _close_by_session,
    _idiosyncratic_volatility,
    _industry_momentum,
    _max_daily_return,
    _month_index,
    _monthly_returns,
    _residual_momentum,
)


def test_pxf_fundamental_ratios_match_hand_calculation() -> None:
    values = _fundamental_features(
        {
            "stale": False,
            "shares": 100.0,
            "book_equity": 500.0,
            "net_income": 50.0,
            "operating_cash_flow": 60.0,
            "revenue": 300.0,
            "cost_of_revenue": 100.0,
            "gross_profit": None,
            "assets": 200.0,
            "assets_prior": 100.0,
        },
        latest_close=10.0,
    )

    assert values["log_me"] == pytest.approx(log(1000.0))
    assert values["bm"] == pytest.approx(0.5)
    assert values["ep"] == pytest.approx(0.05)
    assert values["cfp"] == pytest.approx(0.06)
    assert values["gp_at"] == pytest.approx(1.0)
    assert values["roe"] == pytest.approx(0.1)
    assert values["asset_growth"] == pytest.approx(1.0)


def test_industry_momentum_requires_three_available_symbols() -> None:
    values = _industry_momentum(
        {"A": 0.3, "B": 0.1, "C": 0.2, "D": None, "E": 0.5, "F": 0.4},
        {"A": "BusEq", "B": "BusEq", "C": "BusEq", "D": "BusEq", "E": "Other", "F": "Other"},
    )

    assert values["A"] == pytest.approx(0.2)
    assert values["B"] == pytest.approx(0.2)
    assert values["C"] == pytest.approx(0.2)
    assert values["D"] == pytest.approx(0.2)
    assert values["E"] is None
    assert values["F"] is None


def test_residual_momentum_matches_36_month_regression_and_11_month_sample_scale() -> None:
    decision_month = _month_index(date(2024, 6, 1))
    close_months = list(range(decision_month - 37, decision_month))
    market_returns = [((index % 7) - 3) * 0.008 for index in range(36)]
    residual_pattern = [((index % 5) - 2) * 0.003 for index in range(36)]
    asset_returns = [0.004 + 1.25 * market + residual for market, residual in zip(market_returns, residual_pattern)]
    symbol = _monthly_close_frame(close_months, asset_returns, initial_close=100.0)
    market = _monthly_close_frame(close_months, market_returns, initial_close=200.0)

    actual = _residual_momentum(symbol, _monthly_returns(market), decision_month)

    residuals = _hand_ols_residuals(asset_returns, market_returns)
    selected = residuals[24:35]
    assert actual == pytest.approx(sum(selected) / stdev(selected))


def test_idiosyncratic_volatility_uses_intercept_regression_and_21_residuals() -> None:
    market_returns = [((index % 9) - 4) * 0.002 for index in range(21)]
    residuals = [((index % 4) - 1.5) * 0.0007 for index in range(21)]
    asset_returns = [0.0003 + 0.8 * market + residual for market, residual in zip(market_returns, residuals)]
    market = _daily_close_frame(market_returns, initial_close=200.0)
    symbol = _daily_close_frame(asset_returns, initial_close=50.0)

    actual = _idiosyncratic_volatility(symbol, _close_by_session(market))

    asset_log_returns = [log(1.0 + value) for value in asset_returns]
    market_log_returns = [log(1.0 + value) for value in market_returns]
    assert actual == pytest.approx(stdev(_hand_ols_residuals(asset_log_returns, market_log_returns)))


def test_max_daily_return_uses_the_last_21_returns() -> None:
    closes = [100.0, *[100.0 + index for index in range(1, 22)]]

    assert _max_daily_return(closes, 21) == pytest.approx(max(closes[index] / closes[index - 1] - 1 for index in range(1, 22)))
    assert _max_daily_return(closes[:21], 21) is None


def _monthly_close_frame(months: list[int], returns: list[float], *, initial_close: float) -> pl.DataFrame:
    closes = [initial_close]
    for value in returns:
        closes.append(closes[-1] * (1.0 + value))
    timestamps = [_datetime_for_month(month) for month in months]
    return pl.DataFrame({"event_time": timestamps, "close": closes})


def _daily_close_frame(returns: list[float], *, initial_close: float) -> pl.DataFrame:
    closes = [initial_close]
    for value in returns:
        closes.append(closes[-1] * (1.0 + value))
    start = date(2024, 1, 1)
    timestamps = [datetime.combine(start + timedelta(days=index), datetime.min.time(), UTC) for index in range(len(closes))]
    return pl.DataFrame({"event_time": timestamps, "close": closes})


def _datetime_for_month(month_index: int) -> datetime:
    year, month_offset = divmod(month_index, 12)
    return datetime(year, month_offset + 1, 15, 20, 0, tzinfo=UTC)


def _hand_ols_residuals(asset: list[float], market: list[float]) -> list[float]:
    asset_mean, market_mean = mean(asset), mean(market)
    beta = sum((left - asset_mean) * (right - market_mean) for left, right in zip(asset, market)) / sum(
        (right - market_mean) ** 2 for right in market
    )
    intercept = asset_mean - beta * market_mean
    return [left - (intercept + beta * right) for left, right in zip(asset, market)]
