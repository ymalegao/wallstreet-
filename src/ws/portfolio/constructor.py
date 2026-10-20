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
    """Equal-weight the highest-ranked scores, retaining held names inside a buffer."""

    def __init__(self, holdings: int = 25, buffer_rank: int | None = 35) -> None:
        if holdings < 1:
            raise ValueError("holdings must be at least one")
        if buffer_rank is not None and buffer_rank < holdings:
            raise ValueError("buffer_rank must be at least holdings")
        self.holdings = holdings
        self.buffer_rank = buffer_rank

    def construct(
        self, scores: Mapping[str, float], *, held_symbols: set[str] | frozenset[str] = frozenset()
    ) -> PortfolioTargets:
        """Rank finite scores and retain existing holdings within ``buffer_rank``.

        ``buffer_rank=None`` disables retention and selects the highest-ranked
        ``holdings`` names. Scores are rankings, so negative finite scores remain
        eligible when fewer than ``holdings`` scores are positive.
        """
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
            eligible.append((normalized, score))
        eligible.sort(key=lambda item: (-item[1], item[0]))
        ranked_all = [symbol for symbol, _ in eligible]
        if self.buffer_rank is None:
            ranked = tuple(ranked_all[: self.holdings])
        else:
            normalized_held = {symbol.strip().upper() for symbol in held_symbols if symbol.strip()}
            buffered = set(ranked_all[: self.buffer_rank])
            retained = [symbol for symbol in ranked_all if symbol in buffered and symbol in normalized_held]
            selected = retained[: self.holdings]
            selected_set = set(selected)
            for symbol in ranked_all:
                if len(selected) >= self.holdings:
                    break
                if symbol in normalized_held or symbol in selected_set:
                    continue
                selected.append(symbol)
                selected_set.add(symbol)
            ranked = tuple(symbol for symbol in ranked_all if symbol in selected_set)
        if not ranked:
            return PortfolioTargets(weights=MappingProxyType({}), ranked_symbols=())
        weight = 1.0 / len(ranked)
        return PortfolioTargets(
            weights=MappingProxyType({symbol: weight for symbol in ranked}), ranked_symbols=ranked
        )
