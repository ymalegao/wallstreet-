"""Normalized record types shared by every source."""

from __future__ import annotations

from datetime import UTC, datetime
from pydantic import BaseModel, field_validator


class Bar(BaseModel):
    """Minimal raw OHLCV record consumed by the point-in-time bar converter."""

    model_config = {"frozen": True}

    symbol: str
    ts: datetime  # bar START time, UTC
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None
    trade_count: int | None = None

    @field_validator("ts")
    @classmethod
    def _bar_time(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("bar timestamps must be timezone-aware")
        return v.astimezone(UTC)
