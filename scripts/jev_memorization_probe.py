"""Build and score a preregistered, local JEV historical-outcome memorization probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from ws.models.text import TextModel
from ws.store.atomic import atomic_text
from ws.store.event_store import BarStore

MODEL_REVISION = "b63f651ce8ed64481d3f5e73ecdb05f740042f01"
FIRST_MONTH = "2023-01"
LAST_MONTH = "2026-09"
FIRST_REFERENCE_MONTH = "2022-12"
SEED = 20261005
MAX_PER_CLASS_PER_MONTH = 32
CONFIG_PATH = Path("configs/jev_memorization.yaml")
QUESTION_PATH = Path("data/probe/jev-memorization-questions.jsonl")
MANIFEST_PATH = Path("data/probe/jev-memorization-manifest.json")
SCORES_PATH = Path("data/probe/jev-memorization-scores.jsonl")
REPORT_PATH = Path("docs/jev-memorization-probe.json")
REPORT_MD_PATH = Path("docs/jev-memorization-probe.md")


def months_inclusive(start: str, end: str) -> list[str]:
    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    months: list[str] = []
    year, month = sy, sm
    while (year, month) <= (ey, em):
        months.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def option_order(month: str, ticker: str) -> list[str]:
    options = ["higher", "lower"]
    seed = int(hashlib.sha256(f"{SEED}:{month}:{ticker}:options".encode()).hexdigest()[:16], 16)
    random.Random(seed).shuffle(options)
    return options


def build_questions(
    daily: pl.DataFrame,
    candidates: set[str],
    start: str = FIRST_MONTH,
    end: str = LAST_MONTH,
) -> list[dict[str, Any]]:
    required = months_inclusive(FIRST_REFERENCE_MONTH, end)
    monthly = (
        daily.filter(pl.col("symbol").is_in(candidates))
        .sort(["symbol", "ts"])
        .with_columns(month=pl.col("ts").dt.strftime("%Y-%m"))
        .group_by(["symbol", "month"], maintain_order=True)
        .agg(pl.col("close").last())
        .filter(pl.col("month").is_in(required))
        .sort(["symbol", "month"])
    )
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in monthly.iter_rows(named=True):
        grouped[row["symbol"]][row["month"]] = float(row["close"])
    complete_symbols = {ticker for ticker, closes in grouped.items() if all(month in closes for month in required)}
    grouped = {ticker: grouped[ticker] for ticker in complete_symbols}

    outcomes_by_month: dict[str, dict[str, bool]] = {}
    for month in months_inclusive(start, end):
        year, number = map(int, month.split("-"))
        previous = f"{year - 1:04d}-12" if number == 1 else f"{year:04d}-{number - 1:02d}"
        outcomes: dict[str, bool] = {}
        for ticker, closes in grouped.items():
            if previous in closes and month in closes:
                outcomes[ticker] = closes[month] > closes[previous]
        if not outcomes or len(set(outcomes.values())) != 2:
            raise ValueError(f"Month {month} lacks both higher and lower outcomes")
        outcomes_by_month[month] = outcomes

    questions: list[dict[str, Any]] = []
    for month, outcomes in outcomes_by_month.items():
        higher = sorted(ticker for ticker, up in outcomes.items() if up)
        lower = sorted(ticker for ticker, up in outcomes.items() if not up)
        count = min(len(higher), len(lower), MAX_PER_CLASS_PER_MONTH)
        year, number = map(int, month.split("-"))
        month_name = date(year, number, 1).strftime("%B %Y")
        month_seed = int(hashlib.sha256(f"{SEED}:{month}:sample".encode()).hexdigest()[:16], 16)
        rng = random.Random(month_seed)
        chosen = [(ticker, True) for ticker in rng.sample(higher, count)]
        chosen.extend((ticker, False) for ticker in rng.sample(lower, count))
        rng.shuffle(chosen)
        for ticker, up in chosen:
            options = option_order(month, ticker)
            answer = "higher" if up else "lower"
            named_question = (
                f"Over {month_name}, did the split/dividend-adjusted closing price of ticker {ticker} "
                "finish higher or lower than at the end of the preceding month? Choose the matching description."
            )
            masked_question = (
                f"Over {month_name}, did the split/dividend-adjusted closing price of the stock whose identity "
                "is withheld finish higher or lower than at the end of the preceding month? "
                "Choose the matching description."
            )
            questions.append(
                {
                    "id": f"{month}:{ticker}",
                    "month": month,
                    "ticker": ticker,
                    "named_question": named_question,
                    "masked_question": masked_question,
                    "options": options,
                    "answer": answer,
                    "label_source": "cached 1Day_all month-end close comparison",
                }
            )
    return questions


def canonical_jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)


def build(args: argparse.Namespace) -> None:
    universe_data = json.loads(args.universe.read_text())
    candidates = set(universe_data["symbols"]) - {"SPY", "QQQ"}
    daily = BarStore(args.data_dir, "1Day_all").read()
    questions = build_questions(daily, candidates, args.start, args.end)
    payload = canonical_jsonl(questions)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    atomic_text(args.questions, payload)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"higher": 0, "lower": 0})
    for row in questions:
        counts[row["month"]][row["answer"]] += 1
    manifest = {
        "protocol_version": 1,
        "model": "autotrust/JEV-9B",
        "model_revision": MODEL_REVISION,
        "start": args.start,
        "end": args.end,
        "source": str(args.data_dir / "bars/1Day_all"),
        "universe_source": str(args.universe),
        "source_candidate_count": len(candidates),
        "candidate_count": len({row["ticker"] for row in questions}),
        "question_count": len(questions),
        "paired_control_count": len(questions),
        "question_set_sha256": digest,
        "monthly_balanced_counts": counts,
        "status": "FROZEN_BEFORE_INFERENCE",
    }
    atomic_text(args.manifest, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    config = yaml.safe_load(CONFIG_PATH.read_text())
    config["results"]["question_set_sha256"] = digest
    config["results"]["question_count"] = len(questions)
    config["results"]["candidate_ticker_count"] = manifest["candidate_count"]
    atomic_text(CONFIG_PATH, yaml.safe_dump(config, sort_keys=False))
    print(
        json.dumps(
            {
                "questions": len(questions),
                "tickers": manifest["candidate_count"],
                "source_candidates": len(candidates),
                "sha256": digest,
            }
        )
    )


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total < 1:
        return (math.nan, math.nan)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def one_sided_binomial_p(successes: int, total: int) -> float:
    if total < 1:
        return 1.0
    return sum(math.comb(total, k) for k in range(successes, total + 1)) / (2**total)


def holm_adjust(p_values: list[float]) -> list[float]:
    ordered = sorted(enumerate(p_values), key=lambda pair: pair[1])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, (index, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * p_value))
        adjusted[index] = running
    return adjusted


def summarize(questions: list[dict[str, Any]], scores: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["id"]: row for row in scores}
    if len(by_id) != len(questions) or set(by_id) != {row["id"] for row in questions}:
        raise ValueError("Score set is incomplete or does not match the frozen question set")
    months: list[dict[str, Any]] = []
    for month in months_inclusive(FIRST_MONTH, LAST_MONTH):
        subset = [row for row in questions if row["month"] == month]
        if not subset:
            continue
        named = sum(by_id[row["id"]]["named_prediction"] == row["answer"] for row in subset)
        masked = sum(by_id[row["id"]]["masked_prediction"] == row["answer"] for row in subset)
        low, high = wilson_interval(named, len(subset))
        control_low, control_high = wilson_interval(masked, len(subset))
        months.append(
            {
                "month": month,
                "n_per_condition": len(subset),
                "named_correct": named,
                "named_accuracy": named / len(subset),
                "named_ci95": [low, high],
                "masked_correct": masked,
                "masked_accuracy": masked / len(subset),
                "masked_ci95": [control_low, control_high],
                "named_one_sided_binomial_p": one_sided_binomial_p(named, len(subset)),
                "named_vs_masked_delta": (named - masked) / len(subset),
            }
        )
    adjusted = holm_adjust([row["named_one_sided_binomial_p"] for row in months])
    for row, p_adjusted in zip(months, adjusted, strict=True):
        row["named_p_holm_adjusted"] = p_adjusted
        row["memory_positive"] = p_adjusted < 0.05
    positive = [row["month"] for row in months if row["memory_positive"]]
    cutoff = max(positive) if positive else None
    clean_start = None
    if cutoff:
        cutoff_year, cutoff_month = map(int, cutoff.split("-"))
        clean_start = f"{cutoff_year + 1:04d}-01" if cutoff_month == 12 else f"{cutoff_year:04d}-{cutoff_month + 1:02d}"
    return {
        "model": "autotrust/JEV-9B",
        "revision": MODEL_REVISION,
        "question_count": len(questions),
        "monthly_results": months,
        "detected_cutoff_month": cutoff,
        "candidate_clean_window_start": clean_start,
        "clean_window_status": "CANDIDATE_ONLY" if clean_start else "UNESTABLISHED",
        "interpretation": (
            "A detected month flags above-chance historical recall under this probe. An undetected month does not "
            "certify absence of training overlap. The proposed clean start is a conservative test boundary, not proof."
        ),
    }


def score(args: argparse.Namespace) -> None:
    question_text = args.questions.read_text()
    digest = hashlib.sha256(question_text.encode()).hexdigest()
    manifest = json.loads(args.manifest.read_text())
    if digest != manifest["question_set_sha256"]:
        raise ValueError("Question set hash differs from the frozen manifest")
    questions = [json.loads(line) for line in question_text.splitlines() if line]
    model = TextModel(
        args.endpoint,
        "jev-decision",
        Path("data/inference-cache"),
        "jev",
        Path("data/models/JEV-9B"),
        MODEL_REVISION,
    )
    output: list[dict[str, Any]] = []
    previous: dict[str, dict[str, Any]] = {}
    if args.scores.exists():
        previous = {
            row["id"]: row for row in (json.loads(line) for line in args.scores.read_text().splitlines() if line)
        }
    for question in questions:
        if question["id"] in previous:
            output.append(previous[question["id"]])
            continue
        named = model.choice(
            "Historical public-market fact recall check.",
            question["named_question"],
            question["options"],
        )
        masked = model.choice(
            "Historical public-market fact recall check.",
            question["masked_question"],
            question["options"],
        )
        row = {
            "id": question["id"],
            "month": question["month"],
            "ticker": question["ticker"],
            "answer": question["answer"],
            "options": question["options"],
            "named_prediction": named["label"],
            "named_probabilities": named["probabilities"],
            "masked_prediction": masked["label"],
            "masked_probabilities": masked["probabilities"],
        }
        output.append(row)
        previous[row["id"]] = row
        if len(output) % args.checkpoint_every == 0:
            atomic_text(args.scores, canonical_jsonl(output))
            print(f"Scored {len(output)}/{len(questions)} matched questions", flush=True)
    atomic_text(args.scores, canonical_jsonl(output))
    report = summarize(questions, output)
    report["question_set_sha256"] = digest
    report["chance_baseline"] = 0.5
    atomic_text(REPORT_PATH, json.dumps(report, indent=2, allow_nan=False) + "\n")
    config = yaml.safe_load(CONFIG_PATH.read_text())
    config["results"].update(
        {
            "named_accuracy": sum(row["named_correct"] for row in report["monthly_results"])
            / sum(row["n_per_condition"] for row in report["monthly_results"]),
            "masked_control_accuracy": sum(row["masked_correct"] for row in report["monthly_results"])
            / sum(row["n_per_condition"] for row in report["monthly_results"]),
            "detected_cutoff_month": report["detected_cutoff_month"],
            "candidate_clean_window_start": report["candidate_clean_window_start"],
            "clean_window_status": report["clean_window_status"],
        }
    )
    atomic_text(CONFIG_PATH, yaml.safe_dump(config, sort_keys=False))
    write_markdown(report)


def write_markdown(report: dict[str, Any]) -> None:
    existing = REPORT_MD_PATH.read_text()
    protocol, marker, remainder = existing.partition("## Results")
    if not marker:
        raise ValueError("Probe document is missing its frozen Results section")
    _, limits_marker, limits = remainder.partition("## Limits")
    total_n = sum(row["n_per_condition"] for row in report["monthly_results"])
    named_accuracy = sum(row["named_correct"] for row in report["monthly_results"]) / total_n
    masked_accuracy = sum(row["masked_correct"] for row in report["monthly_results"]) / total_n
    lines = [
        protocol.rstrip(),
        "",
        "## Results",
        "",
        (
            f"Revision `{report['revision']}`; {report['question_count']} questions; set SHA-256 "
            f"`{report['question_set_sha256']}`."
        ),
        (
            f"Pooled named accuracy: **{named_accuracy:.1%}**; identity-masked control: "
            f"**{masked_accuracy:.1%}**; chance: 50%."
        ),
        "",
        (
            f"Detected cutoff month: **{report['detected_cutoff_month'] or 'not detected'}**. "
            "Candidate clean-window start: "
            f"**{report['candidate_clean_window_start'] or 'unestablished'}**."
        ),
        "",
        (
            "A positive result indicates detectable above-chance recall on this probe. No positive result does not "
            "establish that the model lacks relevant training overlap."
        ),
        "",
        (
            "| Month | n / condition | Named accuracy (95% CI) | Identity-masked accuracy (95% CI) "
            "| Holm-adjusted p | Flagged |"
        ),
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for row in report["monthly_results"]:
        low, high = row["named_ci95"]
        control_low, control_high = row["masked_ci95"]
        lines.append(
            f"| {row['month']} | {row['n_per_condition']} | {row['named_accuracy']:.3f} "
            f"[{low:.3f}, {high:.3f}] | {row['masked_accuracy']:.3f} [{control_low:.3f}, {control_high:.3f}] "
            f"| {row['named_p_holm_adjusted']:.4g} | {'yes' if row['memory_positive'] else 'no'} |"
        )
    lines.extend(["", report["interpretation"], ""])
    if limits_marker:
        lines.extend(["## Limits", limits.strip(), ""])
    atomic_text(REPORT_MD_PATH, "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    build_parser.add_argument(
        "--universe",
        type=Path,
        default=Path("data/universe/listed_stock_candidates_2026-10-04.json"),
    )
    build_parser.add_argument("--questions", type=Path, default=QUESTION_PATH)
    build_parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    build_parser.add_argument("--start", default=FIRST_MONTH)
    build_parser.add_argument("--end", default=LAST_MONTH)
    build_parser.set_defaults(func=build)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--endpoint", default="http://127.0.0.1:8001")
    score_parser.add_argument("--questions", type=Path, default=QUESTION_PATH)
    score_parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    score_parser.add_argument("--scores", type=Path, default=SCORES_PATH)
    score_parser.add_argument("--checkpoint-every", type=int, default=50)
    score_parser.set_defaults(func=score)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
