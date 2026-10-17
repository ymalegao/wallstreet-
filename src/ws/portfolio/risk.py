"""Pure, deterministic v0 risk constraints for a long-only portfolio."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType


@dataclass(frozen=True)
class RiskConfig:
    max_name_weight: float = 0.05
    max_gross_weight: float = 1.0
    max_sector_weight: float = 0.30
    target_annual_volatility: float = 0.15
    drawdown_halt: float = 0.15
    max_adv_fraction: float = 0.01

    def __post_init__(self) -> None:
        for name in (
            "max_name_weight",
            "max_gross_weight",
            "max_sector_weight",
            "target_annual_volatility",
            "drawdown_halt",
            "max_adv_fraction",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_name_weight > 1 or self.max_gross_weight > 1 or self.max_sector_weight > 1:
            raise ValueError("long-only exposure caps cannot exceed 1")
        if self.max_adv_fraction > 1:
            raise ValueError("max_adv_fraction cannot exceed 1")
        if self.drawdown_halt > 1:
            raise ValueError("drawdown_halt cannot exceed 1")


@dataclass(frozen=True)
class RiskDecision:
    weights: Mapping[str, float]
    gross_exposure: float
    constraints: tuple[str, ...]


class RiskManager:
    """Clamp requested weights and report every constraint that changed them."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    def apply(
        self,
        requested: Mapping[str, float],
        *,
        sectors: Mapping[str, str],
        annual_volatility: float | None = None,
        drawdown: float = 0.0,
        current_weights: Mapping[str, float] | None = None,
        equity: float | None = None,
        adv_dollars: Mapping[str, float] | None = None,
    ) -> RiskDecision:
        if not isfinite(drawdown) or not 0 <= drawdown <= 1:
            raise ValueError("drawdown must be in [0, 1]")
        if annual_volatility is not None and (not isfinite(annual_volatility) or annual_volatility <= 0):
            raise ValueError("annual_volatility must be finite and positive when supplied")
        if adv_dollars is not None and (equity is None or not isfinite(equity) or equity <= 0):
            raise ValueError("positive equity is required when applying liquidity caps")
        equity_value = float(equity) if equity is not None else 0.0

        constraints: set[str] = set()
        weights: dict[str, float] = {}
        for raw_symbol, raw_weight in requested.items():
            symbol = raw_symbol.strip().upper()
            if not symbol:
                raise ValueError("symbol cannot be empty")
            if symbol in weights:
                raise ValueError(f"duplicate normalized symbol {symbol}")
            weight = float(raw_weight)
            if not isfinite(weight):
                raise ValueError(f"non-finite target weight for {symbol}")
            if weight < 0:
                constraints.add("long_only")
                weight = 0.0
            if weight > self.config.max_name_weight:
                constraints.add("max_name_weight")
                weight = self.config.max_name_weight
            if drawdown >= self.config.drawdown_halt:
                current = max(0.0, float((current_weights or {}).get(symbol, 0.0)))
                if weight > current:
                    constraints.add("drawdown_no_new_buys")
                    weight = current
            if adv_dollars is not None:
                adv = adv_dollars.get(symbol)
                if adv is None or not isfinite(float(adv)) or float(adv) <= 0:
                    if weight > 0:
                        constraints.add("missing_adv")
                    weight = 0.0
                else:
                    liquidity_cap = self.config.max_adv_fraction * float(adv) / equity_value
                    if weight > liquidity_cap:
                        constraints.add("liquidity_cap")
                        weight = liquidity_cap
            weights[symbol] = weight

        # Scale each sector independently so sector limits survive later gross/volatility scaling.
        sector_members: dict[str, list[str]] = {}
        for symbol in weights:
            sector_members.setdefault(sectors.get(symbol, "UNKNOWN"), []).append(symbol)
        for members in sector_members.values():
            total = sum(weights[symbol] for symbol in members)
            if total > self.config.max_sector_weight:
                scale = self.config.max_sector_weight / total
                for symbol in members:
                    weights[symbol] *= scale
                constraints.add("max_sector_weight")

        total_weight = sum(weights.values())
        gross_scale = min(1.0, self.config.max_gross_weight / total_weight) if total_weight else 1.0
        if gross_scale < 1.0:
            constraints.add("max_gross_weight")
        volatility_scale = 1.0
        if annual_volatility is not None:
            volatility_scale = min(1.0, self.config.target_annual_volatility / annual_volatility)
            if volatility_scale < 1.0:
                constraints.add("volatility_target")
        scale = min(gross_scale, volatility_scale)
        if scale < 1.0:
            weights = {symbol: weight * scale for symbol, weight in weights.items()}

        clean = {symbol: weight for symbol, weight in sorted(weights.items()) if weight > 1e-12}
        return RiskDecision(
            weights=MappingProxyType(clean),
            gross_exposure=sum(clean.values()),
            constraints=tuple(sorted(constraints)),
        )
