"""Published accounting and return characteristics layered on px-v1."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import date, datetime
from itertools import pairwise
from math import isfinite, log
from statistics import mean, stdev
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from ws.features.px_v1 import PX_V1_FEATURES, _percentile_ranks, compute_px_v1
from ws.fundamentals.query import FundamentalsIndex

PXF_V1_ADDED_FEATURES = (
    "log_me",
    "bm",
    "ep",
    "cfp",
    "gp_at",
    "roe",
    "asset_growth",
    "ind_mom",
    "resid_mom",
    "ivol_21d",
    "max_ret_21d",
)
PXF_V1_FEATURES = (*PX_V1_FEATURES, *PXF_V1_ADDED_FEATURES)
PXF_V1_RAW_FEATURES = (*PXF_V1_FEATURES, "adv_dollars_20d_raw")
PXF_V1_LOOKBACK_SESSIONS = 850
_NEW_YORK = ZoneInfo("America/New_York")


def compute_pxf_v1(
    bars: pl.DataFrame,
    spy_bars: pl.DataFrame,
    fundamentals: FundamentalsIndex,
    *,
    as_of: datetime,
    ticker_to_cik: dict[str, str],
    industry_by_cik: dict[str, str],
) -> pl.DataFrame:
    """Compute pxf-v1 rows from point-in-time prices and SEC facts.

    SEC visibility is checked against ``as_of`` inside the pre-indexed facts store.
    All raw pxf additions are ranked across the supplied symbol cross-section here;
    px-v1's existing ranked columns are reused unchanged.
    """
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    base = compute_px_v1(bars, spy_bars)
    if base.is_empty():
        return _empty_frame()
    groups = _groups(bars)
    spy_group = _groups(spy_bars).get("SPY")
    spy_closes = _close_by_session(spy_group) if spy_group is not None else {}
    spy_monthly_returns = _monthly_returns(spy_group) if spy_group is not None else {}
    decision_month = _month_index(as_of.astimezone(_NEW_YORK).date())
    cik_by_symbol = {symbol.strip().upper(): str(cik).zfill(10) for symbol, cik in ticker_to_cik.items()}
    ciks = sorted({cik_by_symbol[symbol] for symbol in groups if symbol in cik_by_symbol})
    company_values = {cik: fundamentals.company_values(cik, as_of) for cik in ciks}

    rows: dict[str, dict[str, float | None]] = {}
    raw_126d: dict[str, float | None] = {}
    for symbol, group in groups.items():
        closes = [float(value) for value in group.get_column("close").to_list()]
        latest_close = closes[-1] if closes else None
        raw_126d[symbol] = _simple_return(closes, 126)
        cik = cik_by_symbol.get(symbol)
        selected = company_values.get(cik, {}) if cik is not None else {}
        values = _fundamental_features(selected, latest_close)
        rows[symbol] = {
            **values,
            "resid_mom": _residual_momentum(group, spy_monthly_returns, decision_month),
            "ivol_21d": _idiosyncratic_volatility(group, spy_closes),
            "max_ret_21d": _max_daily_return(closes, 21),
        }

    industry_by_symbol: dict[str, str] = {}
    for symbol in groups:
        cik = cik_by_symbol.get(symbol)
        industry_by_symbol[symbol] = industry_by_cik.get(cik, "Other") if cik is not None else "Other"
    industry_momentum = _industry_momentum(raw_126d, industry_by_symbol)
    for symbol in groups:
        rows[symbol]["ind_mom"] = industry_momentum.get(symbol)

    raw_frame = pl.DataFrame(
        [
            {"symbol": symbol, **rows[symbol]}
            for symbol in sorted(rows)
        ],
        schema={"symbol": pl.String, **{name: pl.Float64 for name in PXF_V1_ADDED_FEATURES}},
    )
    ranked = base.select(["symbol", *PX_V1_FEATURES, "adv_dollars_20d_raw"]).join(raw_frame, on="symbol", how="left")
    ranked = ranked.select(["symbol", *PX_V1_FEATURES, *PXF_V1_ADDED_FEATURES, "adv_dollars_20d_raw"])
    for name in PXF_V1_ADDED_FEATURES:
        ranked = ranked.with_columns(
            pl.Series(name, _percentile_ranks(ranked.get_column(name).to_list()), dtype=pl.Float64)
        )
    return ranked.sort("symbol")


def _fundamental_features(
    selected: dict[str, float | date | bool | None], latest_close: float | None
) -> dict[str, float | None]:
    shares = _number(selected.get("shares"))
    market_equity = shares * latest_close if shares is not None and latest_close is not None else None
    if market_equity is not None and (market_equity <= 0 or not isfinite(market_equity)):
        market_equity = None
    book_equity = _number(selected.get("book_equity"))
    net_income = _number(selected.get("net_income"))
    operating_cash_flow = _number(selected.get("operating_cash_flow"))
    revenue = _number(selected.get("revenue"))
    cost_of_revenue = _number(selected.get("cost_of_revenue"))
    gross_profit = _number(selected.get("gross_profit"))
    if gross_profit is None and revenue is not None and cost_of_revenue is not None:
        gross_profit = revenue - cost_of_revenue
    assets = _number(selected.get("assets"))
    assets_prior = _number(selected.get("assets_prior"))
    stale = bool(selected.get("stale", True))
    if stale:
        market_equity = None
        book_equity = net_income = operating_cash_flow = revenue = cost_of_revenue = gross_profit = None
        assets = assets_prior = None
    bm = _divide(book_equity, market_equity) if book_equity is not None and book_equity > 0 else None
    roe = _divide(net_income, book_equity) if book_equity is not None and book_equity > 0 else None
    return {
        "log_me": log(market_equity) if market_equity is not None else None,
        "bm": bm,
        "ep": _divide(net_income, market_equity),
        "cfp": _divide(operating_cash_flow, market_equity),
        "gp_at": _divide(gross_profit, assets) if assets is not None and assets > 0 else None,
        "roe": roe,
        "asset_growth": (
            _divide(assets, assets_prior) - 1.0
            if assets is not None and assets_prior is not None and assets_prior > 0
            else None
        ),
    }


def _residual_momentum(
    symbol_bars: pl.DataFrame,
    spy_returns: dict[int, float],
    decision_month: int,
) -> float | None:
    if not spy_returns:
        return None
    symbol_returns = _monthly_returns(symbol_bars)
    needed_months = list(range(decision_month - 36, decision_month))
    if any(month not in symbol_returns or month not in spy_returns for month in needed_months):
        return None
    pairs = [(symbol_returns[month], spy_returns[month]) for month in needed_months]
    residuals = _ols_residuals([item[0] for item in pairs], [item[1] for item in pairs])
    signal_months = list(range(decision_month - 12, decision_month - 1))
    selected_residuals = [residuals[month - needed_months[0]] for month in signal_months]
    deviation = stdev(selected_residuals) if len(selected_residuals) > 1 else 0.0
    if deviation == 0 or not isfinite(deviation):
        return None
    value = sum(selected_residuals) / deviation
    return value if isfinite(value) else None


def _idiosyncratic_volatility(
    symbol_bars: pl.DataFrame,
    market: dict[date, float],
) -> float | None:
    if not market:
        return None
    symbol = _close_by_session(symbol_bars)
    market_sessions = sorted(market)[-22:]
    if len(market_sessions) < 22 or any(session not in symbol for session in market_sessions):
        return None
    asset_returns: list[float] = []
    market_returns: list[float] = []
    for previous, current in pairwise(market_sessions):
        asset_returns.append(log(symbol[current] / symbol[previous]))
        market_returns.append(log(market[current] / market[previous]))
    residuals = _ols_residuals(asset_returns, market_returns)
    deviation = stdev(residuals) if len(residuals) > 1 else 0.0
    return deviation if isfinite(deviation) else None


def _max_daily_return(closes: Sequence[float], window: int) -> float | None:
    if len(closes) < window + 1:
        return None
    values = [closes[index] / closes[index - 1] - 1.0 for index in range(len(closes) - window, len(closes))]
    return max(values) if values and all(isfinite(value) for value in values) else None


def _industry_momentum(
    returns_126d: dict[str, float | None],
    industry_by_symbol: dict[str, str],
) -> dict[str, float | None]:
    """Average same-industry 126-session returns when at least three are available."""
    members_by_industry: dict[str, list[str]] = defaultdict(list)
    for symbol, industry in industry_by_symbol.items():
        members_by_industry[industry].append(symbol)
    result: dict[str, float | None] = {}
    for symbols in members_by_industry.values():
        members = [returns_126d[symbol] for symbol in symbols if returns_126d.get(symbol) is not None]
        industry_return = mean(members) if len(members) >= 3 else None
        for symbol in symbols:
            result[symbol] = industry_return
    return result


def _monthly_returns(bars: pl.DataFrame) -> dict[int, float]:
    closes_by_month: dict[int, tuple[date, float]] = {}
    for row in bars.sort("event_time").select("event_time", "close").iter_rows(named=True):
        session_date = row["event_time"].astimezone(_NEW_YORK).date()
        month = _month_index(session_date)
        closes_by_month[month] = (session_date, float(row["close"]))
    result: dict[int, float] = {}
    months = sorted(closes_by_month)
    for previous, current in pairwise(months):
        if current != previous + 1:
            continue
        previous_close = closes_by_month[previous][1]
        current_close = closes_by_month[current][1]
        if previous_close > 0:
            result[current] = current_close / previous_close - 1.0
    return result


def _close_by_session(bars: pl.DataFrame | None) -> dict[date, float]:
    if bars is None or bars.is_empty():
        return {}
    result: dict[date, float] = {}
    for timestamp, close in bars.sort("event_time").select("event_time", "close").iter_rows():
        session_date = timestamp.astimezone(_NEW_YORK).date()
        result[session_date] = float(close)
    return result


def _ols_residuals(asset: Sequence[float], market: Sequence[float]) -> list[float]:
    if len(asset) != len(market) or len(asset) < 2:
        return []
    asset_mean, market_mean = mean(asset), mean(market)
    market_ss = sum((value - market_mean) ** 2 for value in market)
    beta = (
        sum((left - asset_mean) * (right - market_mean) for left, right in zip(asset, market, strict=True))
        / market_ss
        if market_ss > 0
        else 0.0
    )
    intercept = asset_mean - beta * market_mean
    return [left - (intercept + beta * right) for left, right in zip(asset, market, strict=True)]


def _simple_return(closes: Sequence[float], sessions: int) -> float | None:
    if len(closes) <= sessions or closes[-sessions - 1] <= 0:
        return None
    value = closes[-1] / closes[-sessions - 1] - 1.0
    return value if isfinite(value) else None


def _groups(bars: pl.DataFrame) -> dict[str, pl.DataFrame]:
    result: dict[str, pl.DataFrame] = {}
    if bars.is_empty():
        return result
    for raw_symbol, group in bars.group_by("symbol", maintain_order=False):
        symbol = raw_symbol[0] if isinstance(raw_symbol, tuple) else raw_symbol
        result[str(symbol).strip().upper()] = group.sort("event_time")
    return result


def _number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    value = numerator / denominator
    return value if isfinite(value) else None


def _month_index(value: date) -> int:
    return value.year * 12 + value.month - 1


def _empty_frame() -> pl.DataFrame:
    return pl.DataFrame(schema={"symbol": pl.String, **{name: pl.Float64 for name in PXF_V1_RAW_FEATURES}})
