"""Simulation and wall-clock implementations used by pipeline nodes."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, field_validator

TickKind = Literal["open", "close", "intraday", "event"]


class Tick(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    at: datetime
    kind: TickKind = "event"

    @field_validator("at")
    @classmethod
    def _aware_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("tick time must be timezone-aware")
        return value

    @classmethod
    def from_datetime(cls, at: datetime, kind: TickKind = "event") -> Tick:
        if at.tzinfo is None or at.utcoffset() is None:
            raise ValueError("tick time must be timezone-aware")
        return cls(id=at.isoformat(), at=at, kind=kind)


class Clock(Protocol):
    def now(self) -> datetime: ...

    def tick(self) -> Tick: ...


class SimClock:
    """Monotonic clock over a fixed, ordered sequence of ticks."""

    def __init__(self, ticks: Iterable[Tick]) -> None:
        self._ticks = tuple(ticks)
        if not self._ticks:
            raise ValueError("SimClock requires at least one tick")
        if any(right.at < left.at for left, right in zip(self._ticks, self._ticks[1:], strict=False)):
            raise ValueError("simulation ticks must be ordered")
        self._index = 0

    def now(self) -> datetime:
        return self.tick().at

    def tick(self) -> Tick:
        return self._ticks[self._index]

    def advance(self) -> Tick:
        if self._index + 1 >= len(self._ticks):
            raise StopIteration("simulation clock reached its final tick")
        self._index += 1
        return self.tick()

    def __iter__(self) -> Iterator[Tick]:
        yield self.tick()
        while self._index + 1 < len(self._ticks):
            yield self.advance()


class LiveClock:
    """Clock for an already scheduled live decision tick.

    The scheduler supplies the exchange-aligned tick; this class never invents
    a tick from unaligned wall time.
    """

    def __init__(self, tick: Tick) -> None:
        self._tick = tick

    def now(self) -> datetime:
        # Freeze visibility at the scheduled decision time even as execution runs.
        return self._tick.at

    def tick(self) -> Tick:
        return self._tick
