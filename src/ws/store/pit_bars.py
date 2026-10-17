"""Point-in-time daily bars with clock-bound visibility and append-safe Parquet files."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ws.core.clock import Clock
from ws.bars import Bar
from ws.store.atomic import atomic_path, file_lock

NEW_YORK = ZoneInfo("America/New_York")
PIT_BAR_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.String(),
    "event_time": pl.Datetime("us", "UTC"),
    "available_at": pl.Datetime("us", "UTC"),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Float64(),
    "vwap": pl.Float64(),
    "adj_factor": pl.Float64(),
}


class PITBar(BaseModel):
    """Daily OHLCV bar with exchange close time and conservative availability time."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    event_time: datetime
    available_at: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None
    adj_factor: float | None = None

    @field_validator("event_time", "available_at")
    @classmethod
    def _utc_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("PIT bar timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _availability_after_event(self) -> PITBar:
        if self.available_at < self.event_time:
            raise ValueError("available_at must be at or after event_time")
        return self


def daily_bar_to_pit(bar: Bar, vendor_delay: timedelta = timedelta(minutes=15)) -> PITBar:
    """Map Alpaca's daily bar date to the XNYS close, then add a conservative delivery lag.

    Alpaca daily timestamps identify the session date rather than the instant the close became
    observable. Raw prices are retained; adjusted historical bars can embed later corporate actions.
    """
    if vendor_delay < timedelta(0):
        raise ValueError("vendor_delay cannot be negative")
    session_date = bar.ts.astimezone(NEW_YORK).date()
    close_local = datetime.combine(session_date, time(16), NEW_YORK)
    try:
        import exchange_calendars as xcals

        session = xcals.get_calendar("XNYS").date_to_session(session_date.isoformat(), direction="none")
        event_time = xcals.get_calendar("XNYS").session_close(session).to_pydatetime().astimezone(UTC)
    except (ImportError, ValueError, KeyError):
        # Retain a sensible timestamp for fixtures and dates outside the calendar package range.
        event_time = close_local.astimezone(UTC)
    return PITBar(
        symbol=bar.symbol,
        event_time=event_time,
        available_at=event_time + vendor_delay,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        volume=bar.volume,
        vwap=bar.vwap,
        adj_factor=None,
    )


class PITBarStore:
    """Year/month-partitioned Parquet store whose public query API accepts a Clock."""

    def __init__(self, root: Path) -> None:
        self.dir = root

    def write(self, bars: Iterable[PITBar]) -> int:
        grouped: dict[tuple[int, int], list[dict[str, object]]] = {}
        for bar in bars:
            grouped.setdefault((bar.event_time.year, bar.event_time.month), []).append(bar.model_dump())
        if not grouped:
            return 0
        inserted = 0
        with file_lock(self.dir / ".writer.lock"):
            for (year, month), rows in grouped.items():
                part_dir = self.dir / f"year={year:04d}" / f"month={month:02d}"
                part_dir.mkdir(parents=True, exist_ok=True)
                path = part_dir / "part-00000.parquet"
                new = pl.DataFrame(rows, schema=PIT_BAR_SCHEMA)
                old = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=PIT_BAR_SCHEMA)
                combined = pl.concat([old, new], how="vertical").unique(
                    subset=["symbol", "event_time"], keep="first"
                ).sort(["event_time", "symbol"])
                added = combined.height - old.height
                inserted += added
                if added == 0:
                    continue
                with atomic_path(path) as tmp:
                    combined.write_parquet(tmp)
        return inserted

    def read(
        self,
        clock: Clock,
        symbols: list[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        """Read only records available at the clock's current tick.

        ``start`` and ``end`` constrain event time; they cannot move visibility beyond the clock.
        """
        now = clock.now()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("clock returned a naive time")
        for name, value in (("start", start), ("end", end)):
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError(f"{name} must be timezone-aware")
        files = sorted(self.dir.glob("year=*/month=*/part-*.parquet")) if self.dir.exists() else []
        if not files:
            return pl.DataFrame(schema=PIT_BAR_SCHEMA)
        frame = pl.scan_parquet([str(path) for path in files]).filter(pl.col("available_at") <= now)
        if symbols is not None:
            frame = frame.filter(pl.col("symbol").is_in(symbols))
        if start is not None:
            frame = frame.filter(pl.col("event_time") >= start.astimezone(UTC))
        if end is not None:
            frame = frame.filter(pl.col("event_time") < end.astimezone(UTC))
        return frame.sort(["event_time", "symbol"]).collect().select(list(PIT_BAR_SCHEMA))

