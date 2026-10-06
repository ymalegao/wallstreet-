from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

_SPEC = importlib.util.spec_from_file_location(
    "jev_memorization_probe", Path(__file__).parents[1] / "scripts/jev_memorization_probe.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_PROBE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROBE)
build_questions = _PROBE.build_questions
holm_adjust = _PROBE.holm_adjust
one_sided_binomial_p = _PROBE.one_sided_binomial_p
summarize = _PROBE.summarize
wilson_interval = _PROBE.wilson_interval


def test_build_questions_balances_each_month_and_pairs_identity_control() -> None:
    rows = []
    closes = {
        "UP1": [10, 12, 11],
        "UP2": [10, 11, 12],
        "DN1": [10, 9, 8],
        "DN2": [10, 8, 9],
    }
    months = ["2022-12", "2023-01", "2023-02"]
    for ticker, prices in closes.items():
        for month, price in zip(months, prices, strict=True):
            year, number = map(int, month.split("-"))
            rows.append(
                {
                    "symbol": ticker,
                    "ts": datetime(year, number, 28, tzinfo=UTC),
                    "close": price,
                }
            )
    questions = build_questions(pl.DataFrame(rows), {"UP1", "UP2", "DN1", "DN2"}, start="2023-01", end="2023-02")
    assert len(questions) == 8
    for month in ["2023-01", "2023-02"]:
        subset = [row for row in questions if row["month"] == month]
        assert [row["answer"] for row in subset].count("higher") == 2
        assert [row["answer"] for row in subset].count("lower") == 2
        assert all(row["named_question"] != row["masked_question"] for row in subset)
        assert all(set(row["options"]) == {"higher", "lower"} for row in subset)


def test_exact_binomial_wilson_and_holm_helpers() -> None:
    assert one_sided_binomial_p(8, 10) == 56 / 1024
    low, high = wilson_interval(5, 10)
    assert 0.2 < low < 0.5 < high < 0.8
    assert holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_cutoff_requires_holm_significance_and_complete_scores() -> None:
    questions = [
        {"id": f"2023-01:{index}", "month": "2023-01", "answer": "higher" if index < 8 else "lower"}
        for index in range(10)
    ]
    scores = [
        {
            "id": row["id"],
            "named_prediction": row["answer"] if index < 8 else "higher",
            "masked_prediction": "higher" if index % 2 == 0 else "lower",
        }
        for index, row in enumerate(questions)
    ]
    result = summarize(questions, scores)
    assert result["detected_cutoff_month"] is None
    assert result["candidate_clean_window_start"] is None
    assert result["clean_window_status"] == "UNESTABLISHED"
