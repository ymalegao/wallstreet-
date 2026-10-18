"""Cross-sectional prediction and portfolio summary statistics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import date
from math import isfinite, sqrt
from statistics import mean, stdev
from typing import TypedDict

import exchange_calendars as xcals
import polars as pl

_XNYS = xcals.get_calendar("XNYS")


class RankIC(TypedDict):
    date: date
    rank_ic: float


def rank_ic_by_date(
    frame: pl.DataFrame,
    predictions: Iterable[float | None],
    *,
    target: str = "raw_forward_return",
    rebalance_only: bool = True,
) -> list[RankIC]:
    """Calculate Spearman rank IC for each date with at least two valid pairs."""
    values = list(predictions)
    if len(values) != frame.height:
        raise ValueError("prediction count must match frame rows")
    grouped: dict[date, list[tuple[float, float]]] = defaultdict(list)
    for row, prediction in zip(frame.iter_rows(named=True), values, strict=True):
        session_date = row["date"]
        actual = row.get(target)
        if (
            actual is None
            or prediction is None
            or not isfinite(float(actual))
            or not isfinite(float(prediction))
        ):
            continue
        if rebalance_only and not is_month_end(session_date):
            continue
        grouped[session_date].append((float(prediction), float(actual)))
    output: list[RankIC] = []
    for session_date in sorted(grouped):
        pairs = grouped[session_date]
        if len(pairs) < 2:
            continue
        prediction_ranks = _rank([pair[0] for pair in pairs])
        actual_ranks = _rank([pair[1] for pair in pairs])
        ic = _correlation(prediction_ranks, actual_ranks)
        if ic is not None:
            output.append({"date": session_date, "rank_ic": ic})
    return output


def summarize_series(values: Iterable[float]) -> dict[str, float | int | None]:
    data = [float(value) for value in values if isfinite(float(value))]
    if not data:
        return {"count": 0, "mean": None, "std": None, "t_stat": None, "positive_fraction": None}
    average = mean(data)
    deviation = stdev(data) if len(data) > 1 else 0.0
    standard_error = deviation / sqrt(len(data)) if len(data) > 1 else 0.0
    return {
        "count": len(data),
        "mean": average,
        "std": deviation,
        "t_stat": average / standard_error if standard_error else None,
        "positive_fraction": sum(value > 0 for value in data) / len(data),
    }


def is_month_end(session_date: date) -> bool:
    """Identify the last XNYS session in a calendar month."""
    session = _XNYS.date_to_session(session_date.isoformat(), direction="none")
    next_session = _XNYS.next_session(session)
    next_date = next_session.date()
    return (next_date.month, next_date.year) != (session_date.month, session_date.year)


def _rank(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for source_index, _ in ordered[index:end]:
            ranks[source_index] = average_rank
        index = end
    return ranks


def _correlation(left: list[float], right: list[float]) -> float | None:
    left_mean, right_mean = mean(left), mean(right)
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    if left_ss <= 0 or right_ss <= 0:
        return None
    covariance = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    return covariance / sqrt(left_ss * right_ss)
