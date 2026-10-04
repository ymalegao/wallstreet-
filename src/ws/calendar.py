"""NYSE trading calendar and the bot's decision cycles.

The system decides at two fixed cycles per session (design spec §1):
  * morning:   09:45 ET (close of the first 15-minute bar)
  * afternoon: 30 minutes before the session close (15:30 ET normally, 12:30 ET on half days)

An event can only influence the first cycle at or after ``first_seen_ts + latency``. Labels,
backtests and the live scheduler all use these functions, so research and trading cannot disagree
about when an event was actionable.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd

from ws.timeutil import ET, to_utc

MORNING_CYCLE_ET = time(9, 45)
AFTERNOON_OFFSET_FROM_CLOSE = timedelta(minutes=30)
DEFAULT_LATENCY = timedelta(minutes=5)


@lru_cache(maxsize=1)
def nyse() -> xcals.ExchangeCalendar:
    return xcals.get_calendar("XNYS")


def is_session(d: date) -> bool:
    return bool(nyse().is_session(pd.Timestamp(d)))


def session_close(d: date) -> datetime:
    return to_utc(nyse().session_close(pd.Timestamp(d)).to_pydatetime())


def session_open(d: date) -> datetime:
    return to_utc(nyse().session_open(pd.Timestamp(d)).to_pydatetime())


def cycles_for_session(d: date) -> tuple[datetime, datetime]:
    """(morning, afternoon) decision cycle timestamps in UTC for a session date."""
    if not is_session(d):
        raise ValueError(f"{d} is not an NYSE session")
    morning = to_utc(datetime.combine(d, MORNING_CYCLE_ET, tzinfo=ET))
    afternoon = session_close(d) - AFTERNOON_OFFSET_FROM_CLOSE
    return morning, afternoon


def next_session(d: date) -> date:
    """First session strictly after date ``d`` (``d`` itself need not be a session)."""
    cal = nyse()
    ts = pd.Timestamp(d)
    nxt = cal.date_to_session(ts, direction="next")
    if nxt == ts:
        nxt = cal.next_session(ts)
    return nxt.date()


def add_sessions(d: date, n: int) -> date:
    """The session ``n`` sessions after session ``d`` (n >= 0)."""
    if not is_session(d):
        raise ValueError(f"{d} is not an NYSE session")
    return nyse().session_offset(pd.Timestamp(d), n).date()


def next_decision_cycle(first_seen_ts: datetime, latency: timedelta = DEFAULT_LATENCY) -> datetime:
    """First decision cycle at or after ``first_seen_ts + latency``."""
    ready = to_utc(first_seen_ts) + latency
    d = ready.astimezone(ET).date()
    if not is_session(d):
        d = next_session(d)
    while True:
        for cycle in cycles_for_session(d):
            if cycle >= ready:
                return cycle
        d = next_session(d)
