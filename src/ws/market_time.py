"""Exchange-local calendar dates for daily market-model inputs."""

from __future__ import annotations

from datetime import date, datetime, time

from ws.timeutil import ET


def daily_model_timestamp(ts: datetime) -> datetime:
    """Encode a daily bar as naive exchange-session midnight for Kronos time features.

    Alpaca's daily bar timestamp is a UTC market-open instant, while generated daily dates
    naturally arrive at midnight. Passing both into Kronos as-is shifts historical hour
    embeddings to 04:00/05:00 and silently misaligns its time features.
    """
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError("A timezone-aware daily bar timestamp is required")
    return datetime.combine(ts.astimezone(ET).date(), time.min)


def session_model_timestamp(session: date) -> datetime:
    return datetime.combine(session, time.min)
