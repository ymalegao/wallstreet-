"""Shared broker state and protocol for simulation and future paper adapters."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from ws.pipeline_v0.messages import Order


class BrokerState(BaseModel):
    """Portfolio cash, shares, close-marked values, equity, and peak equity."""

    model_config = ConfigDict(frozen=True)

    cash: float
    positions: dict[str, float]
    market_values: dict[str, float]
    equity: float
    peak_equity: float


class Broker(Protocol):
    """Minimal state and order interface consumed by pipeline nodes."""

    def state(self, as_of: datetime) -> BrokerState:
        """Settle eligible orders and return the account state visible at as_of."""
        ...

    def submit(self, orders: Sequence[Order], placed_at: datetime) -> None:
        """Queue orders for execution according to this broker's fill model."""
        ...
