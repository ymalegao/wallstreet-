from datetime import date, datetime, timedelta

import pytest

from ws import calendar as cal
from ws.timeutil import ET, UTC


def et(*args: int) -> datetime:
    return datetime(*args, tzinfo=ET)


def test_cycles_regular_session():
    m, a = cal.cycles_for_session(date(2025, 3, 10))  # Monday, EDT
    assert m == et(2025, 3, 10, 9, 45).astimezone(UTC)
    assert a == et(2025, 3, 10, 15, 30).astimezone(UTC)


def test_half_day_afternoon_cycle_is_30min_before_early_close():
    _, a = cal.cycles_for_session(date(2024, 11, 29))  # day after Thanksgiving, 13:00 close
    assert a.astimezone(ET).time().hour == 12 and a.astimezone(ET).time().minute == 30


def test_non_session_rejected():
    with pytest.raises(ValueError):
        cal.cycles_for_session(date(2025, 12, 25))


@pytest.mark.parametrize(
    ("seen", "expected"),
    [
        (et(2025, 3, 10, 8, 0), et(2025, 3, 10, 9, 45)),  # pre-market -> morning cycle
        (et(2025, 3, 10, 9, 40), et(2025, 3, 10, 9, 45)),  # ready exactly at 09:45 -> same cycle
        (et(2025, 3, 10, 9, 41), et(2025, 3, 10, 15, 30)),  # ready 09:46 -> afternoon
        (et(2025, 3, 10, 15, 26), et(2025, 3, 11, 9, 45)),  # too late for afternoon -> next morning
        (et(2025, 3, 8, 12, 0), et(2025, 3, 10, 9, 45)),  # Saturday -> Monday morning
        (et(2025, 4, 17, 18, 0), et(2025, 4, 21, 9, 45)),  # Thursday evening before Good Friday
    ],
)
def test_next_decision_cycle(seen: datetime, expected: datetime):
    assert cal.next_decision_cycle(seen) == expected.astimezone(UTC)


def test_cycle_never_before_ready_time():
    t = et(2024, 1, 2, 0, 0)
    for _ in range(500):
        c = cal.next_decision_cycle(t)
        assert c >= t + cal.DEFAULT_LATENCY
        t += timedelta(minutes=37)


def test_add_sessions_skips_weekend_and_holiday():
    assert cal.add_sessions(date(2025, 4, 17), 1) == date(2025, 4, 21)  # Good Friday closed
