"""Transparent daily-bar transaction-cost model for backtests."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt


@dataclass(frozen=True)
class CostModel:
    """Commission + half spread + square-root market impact, all as decimal costs."""

    commission_bps: float = 0.0
    half_spread_bps: float = 2.5
    impact_coefficient: float = 0.1

    def __post_init__(self) -> None:
        for name in ("commission_bps", "half_spread_bps", "impact_coefficient"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    def rate(self, *, order_notional: float, adv_dollars: float, annual_volatility: float) -> float:
        """Return the one-way cost fraction for a trade at its reference price."""
        if not isfinite(order_notional) or order_notional < 0:
            raise ValueError("order_notional must be finite and non-negative")
        if not isfinite(adv_dollars) or adv_dollars <= 0:
            raise ValueError("adv_dollars must be finite and positive")
        if not isfinite(annual_volatility) or annual_volatility < 0:
            raise ValueError("annual_volatility must be finite and non-negative")
        impact = self.impact_coefficient * annual_volatility * sqrt(order_notional / adv_dollars)
        return (self.commission_bps + self.half_spread_bps) / 10_000 + impact

    def dollars(self, *, order_notional: float, adv_dollars: float, annual_volatility: float) -> float:
        return order_notional * self.rate(
            order_notional=order_notional,
            adv_dollars=adv_dollars,
            annual_volatility=annual_volatility,
        )
