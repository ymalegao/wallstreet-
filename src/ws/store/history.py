"""Preloaded daily-bar history with clock-filtered point-in-time access."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import polars as pl

from ws.core.clock import LiveClock, Tick
from ws.store.pit_bars import PIT_BAR_SCHEMA, PITBarStore


class BarHistory:
    """Hold run history in memory and reveal only bars available by a requested time."""

    def __init__(self, frame: pl.DataFrame) -> None:
        self._frame = frame.sort(["available_at", "symbol"])

    @classmethod
    def load(cls, store: PITBarStore, symbols: Sequence[str], through: datetime) -> BarHistory:
        """Load one immutable run window through the supplied timezone-aware cutoff."""
        if through.tzinfo is None or through.utcoffset() is None:
            raise ValueError("through must be timezone-aware")
        tick = Tick(id=through.isoformat(), at=through, kind="event")
        frame = store.read(LiveClock(tick), symbols=list(symbols))
        return cls(frame.sort(["available_at", "symbol"]))

    def visible(
        self,
        now: datetime,
        *,
        symbols: Sequence[str] | None = None,
        lookback_sessions: int | None = None,
    ) -> pl.DataFrame:
        """Return rows available by now, optionally limited to recent visible sessions."""
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        if lookback_sessions is not None and lookback_sessions < 1:
            raise ValueError("lookback_sessions must be positive")
        frame = self._frame.filter(pl.col("available_at") <= now)
        if symbols is not None:
            frame = frame.filter(pl.col("symbol").is_in(list(symbols)))
        if lookback_sessions is not None and not frame.is_empty():
            sessions = frame.get_column("event_time").unique().sort().tail(lookback_sessions).to_list()
            frame = frame.filter(pl.col("event_time").is_in(sessions))
        if frame.is_empty():
            return pl.DataFrame(schema=PIT_BAR_SCHEMA)
        return frame.sort(["event_time", "symbol"]).select(list(PIT_BAR_SCHEMA))
