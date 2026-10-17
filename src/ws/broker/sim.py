"""Daily-bar simulated broker with next-session-open fills."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from math import isfinite
from typing import Any

import exchange_calendars as xcals
import polars as pl

from ws.broker.base import BrokerState
from ws.broker.costs import CostModel
from ws.pipeline_v0.messages import Order
from ws.pipeline_v0.metrics import adv_dollars_20d, realized_vol_21d
from ws.store.history import BarHistory

_XNYS = xcals.get_calendar("XNYS")
_NEW_YORK = "America/New_York"


@dataclass(frozen=True)
class _PendingOrder:
    order: Order
    placed_at: datetime


class SimBroker:
    """Fill pending orders at the next eligible XNYS open using preloaded bar history."""

    def __init__(
        self,
        history: BarHistory,
        cost_model: CostModel,
        initial_cash: float = 1000.0,
    ) -> None:
        if not isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError("initial_cash must be finite and positive")
        self.history = history
        self.cost_model = cost_model
        self.cash = initial_cash
        self.peak_equity = initial_cash
        self.positions: dict[str, float] = {}
        self._last_prices: dict[str, float] = {}
        self._pending: list[_PendingOrder] = []
        self.fills: list[dict[str, str | float]] = []
        self.cancelled: list[dict[str, str | float]] = []

    def submit(self, orders: Sequence[Order], placed_at: datetime) -> None:
        """Queue orders for the first session open strictly after placed_at."""
        if placed_at.tzinfo is None or placed_at.utcoffset() is None:
            raise ValueError("placed_at must be timezone-aware")
        self._pending.extend(_PendingOrder(order=order, placed_at=placed_at.astimezone(UTC)) for order in orders)

    def state(self, as_of: datetime) -> BrokerState:
        """Settle due orders, mark shares to the latest visible close, and report account state."""
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
        cutoff = as_of.astimezone(UTC)
        visible = self.history.visible(cutoff)
        due: dict[date, list[_PendingOrder]] = defaultdict(list)
        still_pending: list[_PendingOrder] = []
        for pending in self._pending:
            session_date, session_open = self._next_open(pending.placed_at, cutoff)
            if session_date is None or session_open is None:
                still_pending.append(pending)
            else:
                due[session_date].append(pending)
        self._pending = still_pending

        for session_date in sorted(due):
            session_orders = due[session_date]
            self._settle_session(session_date, session_orders, visible)

        if not visible.is_empty():
            ordered = visible.sort(["event_time", "symbol"])
            for row in ordered.iter_rows(named=True):
                self._last_prices[row["symbol"]] = float(row["close"])
        market_values = {
            symbol: shares * self._last_prices[symbol]
            for symbol, shares in sorted(self.positions.items())
            if shares > 1e-12 and symbol in self._last_prices
        }
        self.positions = {symbol: shares for symbol, shares in self.positions.items() if shares > 1e-12}
        equity = self.cash + sum(market_values.values())
        if not isfinite(equity) or equity <= 0:
            raise ValueError("simulated account equity became non-positive")
        self.peak_equity = max(self.peak_equity, equity)
        return BrokerState(
            cash=self.cash,
            positions=dict(self.positions),
            market_values=market_values,
            equity=equity,
            peak_equity=self.peak_equity,
        )

    def _next_open(self, placed_at: datetime, as_of: datetime) -> tuple[date | None, datetime | None]:
        """Find the first exchange open after an order time and no later than as_of."""
        if as_of <= placed_at:
            return None, None
        start = placed_at.date().isoformat()
        end = as_of.date().isoformat()
        sessions = _XNYS.sessions_in_range(start, end)
        for session in sessions:
            session_open = _XNYS.session_open(session).to_pydatetime().astimezone(UTC)
            if placed_at < session_open <= as_of:
                return session.date(), session_open
        return None, None

    def _settle_session(self, session_date: date, pending: list[_PendingOrder], visible: pl.DataFrame) -> None:
        """Fill one session's sells before buys and cancel orders with no daily bar."""
        session_bars = visible.filter(
            pl.col("event_time").dt.convert_time_zone(_NEW_YORK).dt.date() == session_date
        )
        by_symbol = {row["symbol"]: row for row in session_bars.iter_rows(named=True)}
        session = _XNYS.date_to_session(session_date.isoformat(), direction="none")
        fill_open = _XNYS.session_open(session).to_pydatetime().astimezone(UTC)
        known_at_open = self.history.visible(fill_open)
        adv_rows = adv_dollars_20d(known_at_open)
        vol_rows = realized_vol_21d(known_at_open)
        adv = {row["symbol"]: row["adv_dollars_20d"] for row in adv_rows.iter_rows(named=True)}
        volatility = {row["symbol"]: row["vol_21d"] for row in vol_rows.iter_rows(named=True)}

        viable: list[_PendingOrder] = []
        for pending_order in pending:
            symbol = pending_order.order.symbol
            if symbol not in by_symbol:
                self.cancelled.append(
                    {
                        "session": session_date.isoformat(),
                        "symbol": symbol,
                        "side": pending_order.order.side,
                        "notional": pending_order.order.notional,
                        "reason": "missing_session_bar",
                    }
                )
            else:
                viable.append(pending_order)

        sells = sorted((item for item in viable if item.order.side == "sell"), key=lambda item: item.order.symbol)
        buys = sorted((item for item in viable if item.order.side == "buy"), key=lambda item: item.order.symbol)
        for pending_order in sells:
            self._fill_sell(session_date, pending_order.order, by_symbol[pending_order.order.symbol], adv, volatility)
        self._fill_buys(session_date, buys, by_symbol, adv, volatility)

    def _cost(self, symbol: str, notional: float, adv: dict[str, Any], volatility: dict[str, Any]) -> float:
        """Calculate transaction cost from only bars visible before the fill open."""
        avg_dollars = adv.get(symbol)
        annual_vol = volatility.get(symbol)
        if avg_dollars is None or annual_vol is None or avg_dollars <= 0:
            raise ValueError(f"insufficient pre-fill cost history for {symbol}")
        return self.cost_model.dollars(
            order_notional=notional,
            adv_dollars=float(avg_dollars),
            annual_volatility=float(annual_vol),
        )

    def _fill_sell(
        self,
        session_date: date,
        order: Order,
        bar: dict[str, Any],
        adv: dict[str, Any],
        volatility: dict[str, Any],
    ) -> None:
        """Fill a sell up to the shares currently held."""
        open_price = float(bar["open"])
        shares = min(self.positions.get(order.symbol, 0.0), order.notional / open_price)
        if shares <= 0:
            self.cancelled.append(
                {
                    "session": session_date.isoformat(),
                    "symbol": order.symbol,
                    "side": order.side,
                    "notional": order.notional,
                    "reason": "no_shares_to_sell",
                }
            )
            return
        notional = shares * open_price
        cost = self._cost(order.symbol, notional, adv, volatility)
        self.positions[order.symbol] -= shares
        self.cash += notional - cost
        self.fills.append(
            {
                "session": session_date.isoformat(),
                "symbol": order.symbol,
                "side": "sell",
                "shares": shares,
                "price": open_price,
                "cost": cost,
            }
        )

    def _fill_buys(
        self,
        session_date: date,
        orders: list[_PendingOrder],
        bars: dict[str, dict[str, Any]],
        adv: dict[str, Any],
        volatility: dict[str, Any],
    ) -> None:
        """Scale all buys proportionally when cash cannot cover their notional and costs."""
        if not orders:
            return

        def total_required(scale: float) -> float:
            required = 0.0
            for pending_order in orders:
                order = pending_order.order
                notional = order.notional * scale
                required += notional + self._cost(order.symbol, notional, adv, volatility)
            return required

        scale = 1.0
        if total_required(scale) > self.cash:
            low, high = 0.0, 1.0
            for _ in range(64):
                midpoint = (low + high) / 2.0
                if total_required(midpoint) <= self.cash:
                    low = midpoint
                else:
                    high = midpoint
            scale = low

        for pending_order in orders:
            order = pending_order.order
            notional = order.notional * scale
            if notional <= 1e-12:
                self.cancelled.append(
                    {
                        "session": session_date.isoformat(),
                        "symbol": order.symbol,
                        "side": order.side,
                        "notional": order.notional,
                        "reason": "insufficient_cash",
                    }
                )
                continue
            open_price = float(bars[order.symbol]["open"])
            cost = self._cost(order.symbol, notional, adv, volatility)
            shares = notional / open_price
            self.cash -= notional + cost
            if self.cash < 1e-9:
                self.cash = max(0.0, self.cash)
            self.positions[order.symbol] = self.positions.get(order.symbol, 0.0) + shares
            self.fills.append(
                {
                    "session": session_date.isoformat(),
                    "symbol": order.symbol,
                    "side": "buy",
                    "shares": shares,
                    "price": open_price,
                    "cost": cost,
                }
            )
