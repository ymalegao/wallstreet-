"""Typed messages exchanged by the daily momentum pipeline."""

from __future__ import annotations

from datetime import date
from typing import Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ws.core.messages import Message


class MarketSnapshot(Message):
    """Point-in-time bars and the dated universe visible to one decision tick."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    universe: tuple[str, ...]
    universe_id: str
    bars: pl.DataFrame
    last_session: date
    benchmark_bars: pl.DataFrame = Field(default_factory=pl.DataFrame)


class PortfolioState(Message):
    """Cash, share positions, marked values, equity, and its running peak."""

    cash: float
    positions: dict[str, float]
    market_values: dict[str, float]
    equity: float
    peak_equity: float


class Features(Message):
    """Cross-sectional momentum, volatility, and liquidity features."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    frame: pl.DataFrame
    feature_set: str


class Insights(Message):
    """Per-symbol alpha scores emitted by the configured predictor."""

    scores: dict[str, float]
    model_id: str


class PortfolioTargets(Message):
    """Desired long-only portfolio weights and their ranking."""

    weights: dict[str, float]
    ranked_symbols: tuple[str, ...]


class RiskAdjustedTargets(Message):
    """Portfolio weights after deterministic risk constraints."""

    weights: dict[str, float]
    gross_exposure: float
    constraints: tuple[str, ...]
    risk_reductions: dict[str, float] = Field(default_factory=dict)


class Order(BaseModel):
    """Dollar-notional order priced from the latest close known at decision time."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    side: Literal["buy", "sell"]
    notional: float = Field(gt=0)
    reference_price: float = Field(gt=0)

    @field_validator("symbol")
    @classmethod
    def _uppercase_symbol(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("symbol cannot be empty")
        return normalized


class Orders(Message):
    """Orders produced for submission and counts of submitted or skipped changes."""

    orders: tuple[Order, ...]
    submitted: int
    skipped_small: int
