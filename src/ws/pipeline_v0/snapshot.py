"""Coordinator-side providers for immutable per-tick inputs to the daily pipeline."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

from ws.core.clock import Tick
from ws.pipeline_v0.messages import MarketSnapshot
from ws.store.history import BarHistory

_NEW_YORK = ZoneInfo("America/New_York")


class SnapshotProvider:
    """Build a point-in-time market snapshot outside the computational node graph."""

    def __init__(
        self,
        history: BarHistory,
        universe_path: Path,
        lookback_sessions: int = 300,
        benchmark_symbols: tuple[str, ...] = ("SPY",),
    ) -> None:
        if lookback_sessions < 1:
            raise ValueError("lookback_sessions must be positive")
        universe_data = json.loads(universe_path.read_text(encoding="utf-8"))
        symbols = tuple(sorted({str(symbol).strip().upper() for symbol in universe_data["symbols"]}))
        if not symbols or any(not symbol for symbol in symbols):
            raise ValueError(f"universe at {universe_path} has no valid symbols")
        self.history = history
        self.symbols = symbols
        self.benchmark_symbols = tuple(sorted({symbol.strip().upper() for symbol in benchmark_symbols}))
        self.universe_id = f"sp100-proxy@{date.fromisoformat(universe_data['as_of']).isoformat()}"
        self.lookback_sessions = lookback_sessions

    def snapshot(self, tick: Tick) -> MarketSnapshot:
        """Return the symbols and bars visible to the coordinator at this tick."""
        bars = self.history.visible(
            tick.at,
            symbols=self.symbols,
            lookback_sessions=self.lookback_sessions,
        )
        benchmark_bars = self.history.visible(
            tick.at,
            symbols=self.benchmark_symbols,
            lookback_sessions=self.lookback_sessions,
        )
        tick_session = tick.at.astimezone(_NEW_YORK).date()
        visible_sessions = bars.get_column("event_time").dt.convert_time_zone("America/New_York").dt.date()
        if tick_session not in visible_sessions.to_list():
            raise ValueError(f"no bar is visible for the tick's session {tick_session}")
        last_session = max(value.astimezone(_NEW_YORK).date() for value in bars.get_column("event_time").to_list())
        return MarketSnapshot(
            tick=tick,
            producer="snapshot_provider",
            universe=self.symbols,
            universe_id=self.universe_id,
            bars=bars,
            last_session=last_session,
            benchmark_bars=benchmark_bars,
        )
