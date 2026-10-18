"""Expanding one-year test folds with purging and embargo."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import exchange_calendars as xcals
import polars as pl

_XNYS = xcals.get_calendar("XNYS")


@dataclass(frozen=True)
class WalkForwardFold:
    fold: str
    train: pl.DataFrame
    valid: pl.DataFrame
    test: pl.DataFrame
    train_range: tuple[date, date]
    valid_range: tuple[date, date]
    test_range: tuple[date, date]


def build_folds(table: pl.DataFrame, *, horizon: int = 21) -> list[WalkForwardFold]:
    """Split a pre-holdout labeled table into expanding calendar-year folds."""
    required = {"date", "label_exit_date", "label", "raw_forward_return"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"training table is missing split columns: {sorted(missing)}")
    dates = sorted(table.get_column("date").unique().to_list())
    if not dates:
        return []
    first_date, last_date = dates[0], dates[-1]
    folds: list[WalkForwardFold] = []
    for year in range(first_date.year + 1, last_date.year + 1):
        test_start = date(year, 1, 1)
        test_end = date(year + 1, 1, 1)
        if (test_start - first_date).days < 3 * 365:
            continue
        test = table.filter((pl.col("date") >= test_start) & (pl.col("date") < test_end))
        if test.is_empty():
            continue
        valid_start = _one_year_before(test_start)
        validation_candidates = table.filter(
            (pl.col("date") >= valid_start)
            & (pl.col("date") < test_start)
            & (pl.col("label_exit_date") < test_start)
        )
        if validation_candidates.is_empty():
            continue
        actual_valid_start = min(validation_candidates.get_column("date").to_list())
        train = table.filter(
            (pl.col("date") < actual_valid_start)
            & (pl.col("label_exit_date") < actual_valid_start)
            & (~pl.col("date").is_in(_embargo_dates(actual_valid_start, horizon, dates[0])))
        )
        valid = validation_candidates
        if train.is_empty() or valid.is_empty():
            continue
        # Explicit test-start embargo, in addition to label-window purging.
        test_embargo = set(_embargo_dates(test_start, horizon, dates[0]))
        train = train.filter(~pl.col("date").is_in(test_embargo))
        if train.is_empty():
            continue
        folds.append(
            WalkForwardFold(
                fold=str(year),
                train=train,
                valid=valid,
                test=test,
                train_range=(min(train["date"].to_list()), max(train["date"].to_list())),
                valid_range=(min(valid["date"].to_list()), max(valid["date"].to_list())),
                test_range=(min(test["date"].to_list()), max(test["date"].to_list())),
            )
        )
    return folds


def _one_year_before(value: date) -> date:
    try:
        approximate = value.replace(year=value.year - 1)
    except ValueError:
        approximate = value.replace(year=value.year - 1, day=28)
    sessions = _XNYS.sessions_in_range(approximate.isoformat(), (approximate + timedelta(days=8)).isoformat())
    return sessions[0].date()


def _embargo_dates(boundary: date, horizon: int, first_date: date) -> list[date]:
    sessions = _XNYS.sessions_in_range(first_date.isoformat(), boundary.isoformat())
    previous = [session.date() for session in sessions if session.date() < boundary]
    return previous[-horizon:]
