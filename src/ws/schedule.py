"""Exchange-calendar decision ticks for daily research runs."""

from __future__ import annotations

from datetime import date, timedelta

import exchange_calendars as xcals

from ws.core.clock import Tick

_XNYS = xcals.get_calendar("XNYS")


def daily_ticks(start: date, end: date) -> list[Tick]:
    """Return one close tick per XNYS session in the inclusive date range."""
    if end < start:
        raise ValueError("end must be on or after start")
    sessions = _XNYS.sessions_in_range(start.isoformat(), end.isoformat())
    ticks: list[Tick] = []
    for session in sessions:
        close = _XNYS.session_close(session).to_pydatetime()
        at = close + timedelta(minutes=30)
        ticks.append(Tick(id=at.isoformat(), at=at, kind="close"))
    return ticks
