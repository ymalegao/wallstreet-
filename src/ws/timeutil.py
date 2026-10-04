"""Timestamp handling. Every timestamp inside the system is a tz-aware UTC datetime."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def utcnow() -> datetime:
    return datetime.now(UTC)


def to_utc(ts: datetime) -> datetime:
    """Convert an aware datetime to UTC. Naive datetimes are rejected: guessing their zone is how leaks happen."""
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError(f"naive datetime not allowed: {ts!r}")
    return ts.astimezone(UTC)


def parse_rfc3339(s: str) -> datetime:
    """Parse an RFC3339 / ISO-8601 string that carries an explicit offset ('Z' or '+hh:mm')."""
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return to_utc(dt)


def parse_epoch(seconds: int | float) -> datetime:
    return datetime.fromtimestamp(float(seconds), UTC)


def parse_naive_as(s: str, tz: ZoneInfo) -> datetime:
    """Parse a timestamp, ignoring any offset in the string, and interpret the wall-clock time in `tz`.

    Exists for sources whose offset suffix is known to be wrong (see edgar.py).
    """
    dt = datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    return dt.replace(tzinfo=tz).astimezone(UTC)
