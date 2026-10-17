"""Convert cross-sectional scores into simple long-only target weights."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType


@dataclass(frozen=True)
class PortfolioTargets:
    weights: Mapping[str, float]
    ranked_symbols: tuple[str, ...]


class PortfolioConstructor:
    """Equal-weight the top positive scores; risk constraints are applied separately."""

    def __init__(self, holdings: int = 25) -> None:
        if holdings < 1:
            raise ValueError("holdings must be at least one")
        self.holdings = holdings

    def construct(self, scores: Mapping[str, float]) -> PortfolioTargets:
        eligible = []
        seen: set[str] = set()
        for symbol, raw_score in scores.items():
            score = float(raw_score)
            if not symbol.strip():
                raise ValueError("symbol cannot be empty")
            normalized = symbol.strip().upper()
            if normalized in seen:
                raise ValueError(f"duplicate normalized symbol {normalized}")
            seen.add(normalized)
            if not isfinite(score):
                continue
            if score > 0:
                eligible.append((normalized, score))
        eligible.sort(key=lambda item: (-item[1], item[0]))
        selected = eligible[: self.holdings]
        ranked = tuple(symbol for symbol, _ in selected)
        if not ranked:
            return PortfolioTargets(weights=MappingProxyType({}), ranked_symbols=())
        weight = 1.0 / len(ranked)
        return PortfolioTargets(
            weights=MappingProxyType({symbol: weight for symbol in ranked}), ranked_symbols=ranked
        )
