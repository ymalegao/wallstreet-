"""Normalized record types shared by every source."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator


class TsOrigin(StrEnum):
    """Where ``first_seen_ts`` came from. Matters for how much we trust it in a backtest."""

    OBSERVED = "observed"  # we received it live; first_seen_ts is our own receipt time
    VENDOR = "vendor"  # backfilled; first_seen_ts is the vendor's publish/acceptance time


class Event(BaseModel):
    """One news item or filing, normalized. Immutable once written."""

    model_config = {"frozen": True}

    event_id: str  # "{source}:{source_id}", globally unique
    source: str  # alpaca_news | edgar | finnhub | fnspid
    source_id: str
    first_seen_ts: datetime  # earliest time the information was public / seen by us (UTC)
    ts_origin: TsOrigin
    published_ts: datetime  # vendor's own publish timestamp (UTC)
    updated_ts: datetime | None = None  # vendor's last revision time, if reported
    ingested_at: datetime  # when this record was written to our store (UTC)
    tickers: list[str] = Field(default_factory=list)
    headline: str
    body: str = ""
    url: str = ""
    kind: str = "news"  # news | 8-K | 4 | ...
    meta: dict[str, str] = Field(default_factory=dict)  # source-specific extras (e.g. 8-K items)

    @field_validator("first_seen_ts", "published_ts", "updated_ts", "ingested_at")
    @classmethod
    def _aware_utc(cls, v: datetime | None) -> datetime | None:
        if v is not None and (v.tzinfo is None or v.utcoffset() is None):
            raise ValueError("timestamps must be timezone-aware")
        return v.astimezone(UTC) if v is not None else None

    @field_validator("tickers")
    @classmethod
    def _upper_tickers(cls, v: list[str]) -> list[str]:
        return sorted({t.strip().upper() for t in v if t and t.strip()})

    @model_validator(mode="after")
    def _causality(self) -> Event:
        if self.first_seen_ts > self.ingested_at:
            raise ValueError(f"{self.event_id}: first_seen_ts after ingested_at (clock or parse bug)")
        return self

    @property
    def revised(self) -> bool:
        """True if the vendor revised the item after publication: the stored text may contain later info."""
        return self.updated_ts is not None and self.updated_ts > self.published_ts


class Bar(BaseModel):
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
