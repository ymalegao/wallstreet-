"""Daily-bar portfolio simulator with prior-close signals and deterministic costs."""

from __future__ import annotations

from datetime import UTC, datetime
from math import isfinite

import exchange_calendars as xcals
import polars as pl

from ws.broker.costs import CostModel

_XNYS = xcals.get_calendar("XNYS")
INPUT_COLUMNS = {
    "symbol",
    "event_time",
    "open",
    "close",
    "target_weight",
    "signal_available_at",
    "adv_dollars",
    "annual_volatility",
}


def _aware_utc(value: object, column: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{column} must contain timezone-aware datetimes")
    return value.astimezone(UTC)


def _session_open(event_time: datetime) -> datetime:
    session = _XNYS.date_to_session(event_time.date().isoformat(), direction="none")
    return _XNYS.session_open(session).to_pydatetime().astimezone(UTC)


def simulate_daily(
    panel: pl.DataFrame,
    *,
    initial_equity: float = 1_000.0,
    cost_model: CostModel | None = None,
) -> pl.DataFrame:
    """Simulate open executions and intraday P&L from a tidy daily panel.

    Each row contains the target and cost inputs computed from information available before that
    session's open. `event_time` identifies that session's close; the simulation uses the session
    open for execution, the previous close-to-open move for existing holdings, and the open-to-close
    move for newly selected holdings. The first session begins in cash.

    Required columns are symbol, event_time, open, close, target_weight, signal_available_at,
    adv_dollars, and annual_volatility. Include every held symbol on each session and set its target
    to zero to close it; a missing held-symbol bar fails closed. Signal timestamps after the session
    open are rejected to prevent look-ahead.
    """
    missing = INPUT_COLUMNS - set(panel.columns)
    if missing:
        raise ValueError(f"daily panel is missing columns: {sorted(missing)}")
    if not isfinite(initial_equity) or initial_equity <= 0:
        raise ValueError("initial_equity must be finite and positive")
    if panel.is_empty():
        return pl.DataFrame(
            schema={
                "event_time": pl.Datetime("us", "UTC"),
                "gross_return": pl.Float64(),
                "transaction_costs": pl.Float64(),
                "turnover": pl.Float64(),
                "net_return": pl.Float64(),
                "equity": pl.Float64(),
            }
        )

    model = cost_model or CostModel()
    frame = panel.sort(["event_time", "symbol"])
    if frame.select(pl.struct(["symbol", "event_time"]).n_unique()).item() != frame.height:
        raise ValueError("daily panel has duplicate symbol/event_time rows")

    equity = initial_equity
    holdings: dict[str, float] = {}
    previous_close: dict[str, float] = {}
    previous_event: datetime | None = None
    output: list[dict[str, float | datetime]] = []

    for event_time_raw, group in frame.group_by("event_time", maintain_order=True):
        group_event_time = event_time_raw[0] if isinstance(event_time_raw, tuple) else event_time_raw
        event_time = _aware_utc(group_event_time, "event_time")
        session_open = _session_open(event_time)
        prior_equity = equity
        open_prices: dict[str, float] = {}
        close_prices: dict[str, float] = {}
        targets: dict[str, float] = {}
        adv: dict[str, float] = {}
        volatility: dict[str, float] = {}

        for row in group.iter_rows(named=True):
            symbol = str(row["symbol"]).strip().upper()
            if not symbol:
                raise ValueError("symbol cannot be empty")
            if symbol in open_prices:
                raise ValueError(f"duplicate normalized symbol {symbol} on {event_time.date()}")
            opening, closing = float(row["open"]), float(row["close"])
            target = float(row["target_weight"])
            if not all(isfinite(value) for value in (opening, closing, target)) or opening <= 0 or closing <= 0:
                raise ValueError(f"invalid prices or target weight for {symbol} at {event_time.isoformat()}")
            if target < 0:
                raise ValueError(f"negative target weight for long-only simulation: {symbol}")
            signal_available = _aware_utc(row["signal_available_at"], "signal_available_at")
            if signal_available > session_open:
                raise ValueError(f"signal for {symbol} was unavailable at session open")
            open_prices[symbol] = opening
            close_prices[symbol] = closing
            targets[symbol] = target
            adv[symbol] = float(row["adv_dollars"])
            volatility[symbol] = float(row["annual_volatility"])
            if not isfinite(adv[symbol]) or adv[symbol] <= 0:
                raise ValueError(f"adv_dollars must be finite and positive for {symbol}")
            if not isfinite(volatility[symbol]) or volatility[symbol] < 0:
                raise ValueError(f"annual_volatility must be finite and non-negative for {symbol}")

        gross_target = sum(targets.values())
        if gross_target > 1.0 + 1e-9:
            raise ValueError(f"long-only target weights exceed 100% gross on {event_time.date()}")

        overnight_portfolio_return = 0.0
        overnight_asset_returns: dict[str, float] = {}
        if previous_event is not None:
            if event_time <= previous_event:
                raise ValueError("event_time values must be strictly increasing")
            for symbol, weight in holdings.items():
                if symbol not in open_prices or symbol not in previous_close:
                    raise ValueError(f"held symbol {symbol} is missing from an adjacent daily session")
                asset_return = open_prices[symbol] / previous_close[symbol] - 1.0
                overnight_asset_returns[symbol] = asset_return
                overnight_portfolio_return += weight * asset_return
        pre_trade_equity = equity * (1.0 + overnight_portfolio_return)
        if pre_trade_equity <= 0:
            raise ValueError(f"portfolio equity became non-positive before trade on {event_time.date()}")

        pre_trade_weights = {
            symbol: weight * (1.0 + overnight_asset_returns.get(symbol, 0.0)) / (1.0 + overnight_portfolio_return)
            for symbol, weight in holdings.items()
        }
        all_symbols = set(pre_trade_weights) | set(targets)
        turnover_by_symbol = {
            symbol: abs(targets.get(symbol, 0.0) - pre_trade_weights.get(symbol, 0.0)) for symbol in all_symbols
        }
        turnover = sum(turnover_by_symbol.values())
        transaction_costs = 0.0
        for symbol, delta in turnover_by_symbol.items():
            if delta == 0:
                continue
            if symbol not in adv or symbol not in volatility:
                raise ValueError(f"cost inputs are missing for rebalance of {symbol}")
            notional = pre_trade_equity * delta
            transaction_costs += model.dollars(
                order_notional=notional,
                adv_dollars=adv[symbol],
                annual_volatility=volatility[symbol],
            )
        post_trade_equity = pre_trade_equity - transaction_costs
        if post_trade_equity <= 0:
            raise ValueError(f"transaction costs exhausted portfolio equity on {event_time.date()}")

        intraday_return = sum(
            target * (close_prices[symbol] / open_prices[symbol] - 1.0) for symbol, target in targets.items()
        )
        gross_equity = pre_trade_equity * (1.0 + intraday_return)
        equity = post_trade_equity * (1.0 + intraday_return)
        if equity <= 0:
            raise ValueError(f"portfolio equity became non-positive on {event_time.date()}")
        holdings = {
            symbol: target * (close_prices[symbol] / open_prices[symbol]) / (1.0 + intraday_return)
            for symbol, target in targets.items()
            if target > 0
        }
        previous_close = close_prices
        previous_event = event_time
        output.append(
            {
                "event_time": event_time,
                "gross_return": gross_equity / prior_equity - 1.0,
                "transaction_costs": transaction_costs,
                "turnover": turnover,
                "net_return": equity / prior_equity - 1.0,
                "equity": equity,
            }
        )

    return pl.DataFrame(output).with_columns(pl.col("event_time").cast(pl.Datetime("us", "UTC")))
