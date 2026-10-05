"""Deterministic long-only research replay on regular-hours bars; no broker connection.

Signals are formed at decision cycles and fill at a later bar open. Intrabar stop fills use
min(open, stop), then adverse costs. Adjusted prices are a total-return proxy, not broker fills.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import polars as pl

from ws import calendar as cal
from ws.labels import regular_hours
from ws.timeutil import ET

SECTORS = {
    "AAPL": "technology",
    "AMD": "technology",
    "AVGO": "technology",
    "DELL": "technology",
    "INTC": "technology",
    "LITE": "technology",
    "LRCX": "technology",
    "MRVL": "technology",
    "MSFT": "technology",
    "MU": "technology",
    "NBIS": "technology",
    "NVDA": "technology",
    "ORCL": "technology",
    "PLTR": "technology",
    "STX": "technology",
    "TSM": "technology",
    "WDC": "technology",
    "TSLA": "consumer_discretionary",
    "AMZN": "consumer_discretionary",
    "GOOG": "communication_services",
    "GOOGL": "communication_services",
    "META": "communication_services",
    "BE": "industrials",
    "JPM": "financials",
    "XOM": "energy",
    "PFE": "healthcare",
    "WMT": "consumer_staples",
}


@dataclass(frozen=True)
class Rules:
    capital: float = 500.0
    max_positions: int = 5
    max_per_sector: int = 2
    max_weight: float = 0.20
    cost_bps_per_side: float = 15.0
    day_loss_limit: float = 0.04
    drawdown_limit: float = 0.15


@dataclass
class Position:
    ticker: str
    quantity: float
    entry_ts: datetime
    entry_px: float
    exit_ts: datetime
    stop: float
    signal: float


class Governor:
    """Pure portfolio rules shared by all strategies and controls."""

    def __init__(self, rules: Rules) -> None:
        self.rules = rules
        self.cash = rules.capital
        self.unsettled: list[tuple[date, float]] = []
        self.positions: dict[str, Position] = {}
        self.halted = False

    def settle(self, session: date) -> None:
        self.cash += sum(amount for available, amount in self.unsettled if available <= session)
        self.unsettled = [(available, amount) for available, amount in self.unsettled if available > session]

    def equity(self, prices: dict[str, float]) -> float:
        return (
            self.cash
            + sum(v for _, v in self.unsettled)
            + sum(p.quantity * prices[t] for t, p in self.positions.items())
        )

    def allocation(self, ticker: str, equity: float) -> tuple[float, str]:
        r = self.rules
        if self.halted:
            return 0.0, "halted"
        if ticker in self.positions:
            return 0.0, "already held"
        if len(self.positions) >= r.max_positions:
            return 0.0, "position cap"
        sector = SECTORS.get(ticker)
        if sector is not None and sum(SECTORS.get(t) == sector for t in self.positions) >= r.max_per_sector:
            return 0.0, "sector cap"
        budget = min(r.max_weight * equity, self.cash / (r.max_positions - len(self.positions)))
        return (max(0.0, budget), "") if budget >= 1 else (0.0, "settled cash")

    def sell(self, ticker: str, ts: datetime, price: float, reason: str) -> dict[str, Any]:
        p = self.positions.pop(ticker)
        execution = price * (1 - self.rules.cost_bps_per_side / 10000)
        proceeds = p.quantity * execution
        self.unsettled.append((cal.next_session(ts.astimezone(ET).date()), proceeds))
        return {
            "ticker": ticker,
            "entry_ts": p.entry_ts.isoformat(),
            "exit_ts": ts.isoformat(),
            "entry_px": p.entry_px,
            "exit_px": execution,
            "quantity": p.quantity,
            "pnl": proceeds - p.quantity * p.entry_px,
            "net_return": execution / p.entry_px - 1,
            "exit_reason": reason,
            "signal": p.signal,
        }


def performance(daily_equity: list[float], initial: float) -> dict[str, Any]:
    x = np.array([initial, *daily_equity], dtype=float)
    returns = x[1:] / x[:-1] - 1
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    return {
        "total_return": float(x[-1] / initial - 1),
        "max_drawdown": float(np.min(x / np.maximum.accumulate(x) - 1)),
        "annualized_sharpe": float(np.mean(returns) / std * math.sqrt(252)) if std > 0 else None,
        "sessions": len(daily_equity),
        "final_equity": float(x[-1]),
    }


def run_replay(candidates: list[dict[str, Any]], bars: pl.DataFrame, rules: Rules | None = None) -> dict[str, Any]:
    """Candidates include precomputed entry/exit times, an as-of ATR fraction and a signal."""
    rules = rules or Rules()
    rth = regular_hours(bars).sort("ts", "symbol")
    scheduled: dict[datetime, list[dict[str, Any]]] = {}
    for c in candidates:
        if c["entry_ts"] <= c["cycle_ts"]:
            raise ValueError("Entry must be strictly after the information cycle")
        if c["feature_ts"] > c["cycle_ts"]:
            raise ValueError("Feature timestamp after decision")
        scheduled.setdefault(c["entry_ts"], []).append(c)
    governor = Governor(rules)
    trades = []
    rejected: Counter[str] = Counter()
    last: dict[str, float] = {}
    curve: list[dict[str, Any]] = []
    prior_session = None
    day_start = rules.capital
    peak = rules.capital
    day_halted = False
    flatten = False
    for (ts,), group in rth.group_by("ts", maintain_order=True):
        current = {r["symbol"]: r for r in group.iter_rows(named=True)}
        session = ts.astimezone(ET).date()
        if session != prior_session:
            governor.settle(session)
            day_start = curve[-1]["equity"] if curve else rules.capital
            day_halted = False
            prior_session = session
        for ticker, b in current.items():
            last[ticker] = b["open"]
        # Scheduled exits/kill switches precede entries; sale proceeds settle next session.
        for ticker, p in list(governor.positions.items()):
            if ticker in current and (flatten or ts >= p.exit_ts):
                reason = "kill switch" if flatten else "time exit"
                trades.append(governor.sell(ticker, ts, current[ticker]["open"], reason))
        flatten = flatten and bool(governor.positions)
        for c in sorted(scheduled.get(ts, []), key=lambda x: (-x["signal"], x["ticker"])):
            ticker = c["ticker"]
            if day_halted or flatten:
                rejected["day halt"] += 1
                continue
            if ticker not in current:
                rejected["missing entry bar"] += 1
                continue
            budget, reason = governor.allocation(ticker, governor.equity(last))
            if not budget:
                rejected[reason] += 1
                continue
            price = current[ticker]["open"] * (1 + rules.cost_bps_per_side / 10000)
            qty = math.floor(budget / price * 1e6) / 1e6
            if qty <= 0:
                rejected["zero size"] += 1
                continue
            governor.cash -= qty * price
            governor.positions[ticker] = Position(
                ticker, qty, ts, price, c["exit_ts"], price * (1 - 2 * c["atr_fraction"]), c["signal"]
            )
        # Stops operate throughout each bar, including the entry bar. Gap-through fills at open.
        for ticker, p in list(governor.positions.items()):
            if ticker in current and current[ticker]["low"] <= p.stop:
                b = current[ticker]
                stop_ts = ts if b["open"] <= p.stop else ts + timedelta(minutes=15)
                trades.append(governor.sell(ticker, stop_ts, min(b["open"], p.stop), "ATR stop"))
        for ticker, b in current.items():
            last[ticker] = b["close"]
        equity = governor.equity(last)
        peak = max(peak, equity)
        if equity <= day_start * (1 - rules.day_loss_limit):
            day_halted = True
            flatten = True
        if equity <= peak * (1 - rules.drawdown_limit):
            governor.halted = True
            flatten = True
        curve.append(
            {
                "ts": (ts + timedelta(minutes=15)).isoformat(),
                "session": str(session),
                "equity": equity,
                "settled_cash": governor.cash,
                "positions": len(governor.positions),
            }
        )
        if governor.cash < -0.000001 or len(governor.positions) > rules.max_positions:
            raise AssertionError("Governor invariant violated")
    unprocessed = sum(
        len(items)
        for ts, items in scheduled.items()
        if ts not in {datetime.fromisoformat(r["ts"]) - timedelta(minutes=15) for r in curve}
    )
    if unprocessed:
        rejected["missing simulation bar"] += unprocessed
    daily = {row["session"]: row["equity"] for row in curve}
    return {
        "metrics": {
            **performance(list(daily.values()), rules.capital),
            "closed_trades": len(trades),
            "open_positions": len(governor.positions),
            "halted": governor.halted,
            "win_rate": sum(t["pnl"] > 0 for t in trades) / len(trades) if trades else None,
        },
        "rejected": dict(rejected),
        "trades": trades,
        "daily_equity": daily,
        "curve": curve,
    }
