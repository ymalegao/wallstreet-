"""Evaluate JEV news scores without Kronos filters on all PIT top-25 ticker-cycle samples."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from ws.labels import label
from ws.store.atomic import atomic_text
from ws.store.bar_loader import read_bar_store
from ws.store.event_store import BarStore

MODEL_REVISION = "b63f651ce8ed64481d3f5e73ecdb05f740042f01"


def block_ci(values: list[float], block: int = 10, seed: int = 42) -> list[float | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(finite) < 8:
        return [None, None]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(3000):
        chosen: list[float] = []
        while len(chosen) < len(finite):
            start = int(rng.integers(0, len(finite)))
            chosen.extend(finite[(start + np.arange(block)) % len(finite)].tolist())
        draws.append(float(np.mean(chosen[: len(finite)])))
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def rank_corr(values: list[float], outcomes: list[float]) -> float | None:
    x, y = np.asarray(values, dtype=float), np.asarray(outcomes, dtype=float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    result = spearmanr(x, y).statistic
    return float(result) if np.isfinite(result) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--universe", type=Path, default=Path("data/kronos-pit-2024-2026/universe.json"))
    parser.add_argument("--inputs", type=Path, default=Path("data/jev-pit-replay-inputs.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/jev-pit-replay-manifest.json"))
    parser.add_argument("--signals", type=Path, default=Path("data/jev-pit-signals.json"))
    parser.add_argument("--additional-daily-store", action="append", default=[])
    parser.add_argument("--additional-intraday-store", action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("docs/jev-pit-news-signal.json"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/jev-pit-news-signal.md"))
    args = parser.parse_args()

    input_bytes = args.inputs.read_bytes()
    input_hash = hashlib.sha256(input_bytes).hexdigest()
    manifest = json.loads(args.manifest.read_text())
    scored = json.loads(args.signals.read_text())
    if input_hash != manifest["input_sha256"] or input_hash != scored["input_sha256"]:
        raise ValueError("The scored inputs differ from the frozen JEV-only manifest")
    if not scored.get("complete"):
        raise ValueError("JEV scoring is incomplete")
    signal_rows = scored["signals"]
    signal_by_key = {(row["ticker"], row["cycle"]): float(row["signal"]) for row in signal_rows}

    universe = json.loads(args.universe.read_text())
    start = date.fromisoformat(universe["decision_window"]["start_inclusive"])
    end = date.fromisoformat(universe["decision_window"]["end_exclusive"])
    samples = []
    for session, tickers in universe["selection_by_session"].items():
        session_date = date.fromisoformat(session)
        if not start <= session_date < end:
            continue
        for slot, cycle in enumerate(universe_cycle_times(session_date)):
            for ticker in tickers:
                samples.append(
                    {
                        "ticker": ticker,
                        "cycle_ts": cycle,
                        "session": session_date,
                        "slot": slot,
                    }
                )
    sample_df = pl.DataFrame(
        samples,
        schema={
            "ticker": pl.String,
            "cycle_ts": pl.Datetime("us", "UTC"),
            "session": pl.Date,
            "slot": pl.Int8,
        },
    )
    tickers = sorted({row["ticker"] for row in samples})
    symbols_for_labels = sorted(set(tickers) | {"SPY"})
    intraday_parts = [
        BarStore(args.data_dir, store).read(symbols_for_labels)
        for store in ["15Min", *args.additional_intraday_store]
    ]
    bars = pl.concat([part for part in intraday_parts if not part.is_empty()], how="diagonal_relaxed")
    bars = bars.unique(subset=["symbol", "ts"], keep="first").sort(["symbol", "ts"])
    labels = label(sample_df, bars, horizons=(5,), beta_window=None).sort("session", "slot", "ticker")

    daily_parts = [
        read_bar_store(args.data_dir, store, symbols_for_labels)
        for store in ["1Day_all", *args.additional_daily_store]
    ]
    daily = pl.concat([part for part in daily_parts if not part.is_empty()], how="diagonal_relaxed")
    daily = daily.unique(subset=["symbol", "ts"], keep="first").sort(["symbol", "ts"])
    daily_by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in daily.iter_rows(named=True):
        daily_by_symbol[row["symbol"]].append({**row, "session": row["ts"].date()})

    rows: list[dict[str, Any]] = []
    for row in labels.iter_rows(named=True):
        ticker, session = row["ticker"], row["session"]
        cycle = row["cycle_ts"].isoformat()
        history = [bar for bar in daily_by_symbol[ticker] if bar["session"] < session]
        reverse_20 = None
        momentum_12_1 = None
        if len(history) >= 21:
            reverse_20 = -(float(history[-1]["close"]) / float(history[-21]["close"]) - 1)
        if len(history) >= 252:
            momentum_12_1 = float(history[-21]["close"]) / float(history[-252]["close"]) - 1
        raw_return = row.get("ret_5")
        abnormal = row.get("abret_5")
        rows.append(
            {
                "ticker": ticker,
                "cycle": cycle,
                "session": session.isoformat(),
                "slot": int(row["slot"]),
                "jev_signal": signal_by_key.get((ticker, cycle), 0.0),
                "has_news": (ticker, cycle) in signal_by_key,
                "forward_return_5": float(raw_return) if raw_return is not None else None,
                "abnormal_return_5": float(abnormal) if abnormal is not None else None,
                "reversal_20": reverse_20,
                "momentum_12_1": momentum_12_1,
            }
        )

    valid = [row for row in rows if row["abnormal_return_5"] is not None and np.isfinite(row["abnormal_return_5"])]
    by_cycle: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        by_cycle[(row["session"], row["slot"])].append(row)
    ordered_cycles = sorted(by_cycle)
    metrics: dict[str, Any] = {}
    for feature in ["jev_signal", "reversal_20", "momentum_12_1"]:
        daily_ics: list[float] = []
        for key in ordered_cycles:
            group = [row for row in by_cycle[key] if row[feature] is not None]
            rho = rank_corr([float(row[feature]) for row in group], [float(row["abnormal_return_5"]) for row in group])
            if rho is not None:
                daily_ics.append(rho)
        all_rows = [row for row in valid if row[feature] is not None]
        pooled = rank_corr(
            [float(row[feature]) for row in all_rows],
            [float(row["abnormal_return_5"]) for row in all_rows],
        )
        metrics[feature] = {
            "pooled_spearman_descriptive": pooled,
            "mean_cycle_rank_ic": float(np.mean(daily_ics)) if daily_ics else None,
            "median_cycle_rank_ic": float(np.median(daily_ics)) if daily_ics else None,
            "cycles_with_rank_ic": len(daily_ics),
            "block5_session_95pct_ci": block_ci(daily_ics, block=10),
            "fraction_positive_cycles": float(np.mean(np.asarray(daily_ics) > 0)) if daily_ics else None,
        }

    strata: dict[str, dict[str, float | int]] = {}
    for name, predicate in [
        ("negative_signal", lambda value: value < 0),
        ("zero_signal", lambda value: value == 0),
        ("positive_signal", lambda value: value > 0),
        ("fixed_entry_rule_signal_ge_1_3", lambda value: value >= 1 / 3),
    ]:
        group = [row for row in valid if predicate(float(row["jev_signal"]))]
        strata[name] = {
            "samples": len(group),
            "mean_abnormal_return_5": float(np.mean([row["abnormal_return_5"] for row in group])) if group else 0.0,
            "median_abnormal_return_5": float(np.median([row["abnormal_return_5"] for row in group])) if group else 0.0,
            "positive_fraction": float(np.mean([row["abnormal_return_5"] > 0 for row in group])) if group else 0.0,
        }

    total_sample_count = len(rows)
    universe_source = str(universe.get("candidate_pool", {}).get("source"))
    result = {
        "status": "EXPLORATORY JEV-only feature test; no Kronos forecasts, gates, or exclusions",
        "model": scored.get("model"),
        "revision": scored.get("revision"),
        "period": {"start_inclusive": start.isoformat(), "end_exclusive": end.isoformat()},
        "universe_source": universe_source,
        "label_bar_stores": ["15Min", *args.additional_intraday_store],
        "daily_feature_bar_stores": ["1Day_all", *args.additional_daily_store],
        "universe_sessions": len(universe["selection_by_session"]),
        "samples_all_top25_cycles": total_sample_count,
        "samples_with_complete_5d_prices": len(valid),
        "missing_price_labels": total_sample_count - len(valid),
        "news_ticker_cycles_scored": len(signal_rows),
        "event_texts_scored": sum(len(row["events"]) for row in signal_rows),
        "news_coverage_fraction_of_top25_cycles": len(signal_rows) / total_sample_count,
        "input_sha256": input_hash,
        "latency_minutes": manifest["latency_minutes"],
        "latency_status": manifest["latency_status"],
        "primary_outcome": "five-session stock return less SPY return over the identical interval (beta fixed at 1)",
        "cross_sectional_metrics": metrics,
        "fixed_signal_groups": strata,
        "limitations": [
            (
                "The candidate pool uses the Massive active/delisted master, but unavailable historical bars "
                "still leave residual survivorship bias."
                if "massive" in universe_source.lower()
                else "The candidate pool is the earlier current-asset snapshot; delisted-security survivorship "
                "bias remains until the revised universe is used."
            ),
            "Near-duplicate news threshold and historical news latency remain unverified. Revised text is delayed "
            "until updated_at.",
            "Five-session outcomes overlap. Confidence intervals use a five-session block bootstrap; pooled "
            "correlations are descriptive only.",
            "This feature test is not a live execution backtest, and it does not establish an options strategy.",
            "Supplemental Massive bars use their own adjustment rules and are recorded in the report; "
            "adjustment differences around corporate actions may affect mixed-source labels and price features.",
        ],
    }
    atomic_text(args.output, json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_markdown(args.markdown, result)


def universe_cycle_times(session: date) -> tuple[datetime, datetime]:
    from ws.calendar import cycles_for_session

    return cycles_for_session(session)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# JEV news signal on the point-in-time daily top 25",
        "",
        f"**{report['status']}**",
        "",
        f"{report['period']['start_inclusive']} to {report['period']['end_exclusive']} (exclusive): "
        f"{report['samples_with_complete_5d_prices']:,}/{report['samples_all_top25_cycles']:,} complete "
        "top-25 ticker-cycles, "
        f"{report['news_ticker_cycles_scored']:,} with news scored and {report['event_texts_scored']:,} event texts.",
        "",
        "The table reports JEV by itself, without any Kronos feature, filter, or Kronos-based sample exclusion. "
        "Rank IC is calculated cross-sectionally per decision cycle against five-session SPY-relative returns; "
        "95% intervals resample five-session blocks. With beta fixed at 1, subtracting the same SPY return "
        "from every stock in a cycle does not change that cycle's ranks; SPY is the market benchmark in this "
        "stock test, not an options signal test.",
        "",
        "| Feature | Pooled Spearman (descriptive) | Mean cycle rank IC | Median cycle IC | Positive cycles "
        "| 5-session block 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "jev_signal": "JEV news signal",
        "reversal_20": "20-session reversal",
        "momentum_12_1": "12–1 momentum",
    }
    for feature, name in labels.items():
        row = report["cross_sectional_metrics"][feature]
        ci = row["block5_session_95pct_ci"]
        ci_text = "n/a" if ci[0] is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
        pooled = row["pooled_spearman_descriptive"]
        lines.append(
            f"| {name} | {pooled:+.3f} | {row['mean_cycle_rank_ic']:+.3f} | {row['median_cycle_rank_ic']:+.3f} "
            f"| {row['fraction_positive_cycles']:.1%} | {ci_text} |"
            if pooled is not None and row["mean_cycle_rank_ic"] is not None
            else f"| {name} | n/a | n/a | n/a | n/a | n/a |"
        )
    lines.extend(
        [
            "",
            "## Signal groups",
            "",
            "| Fixed group | Samples | Mean 5-session abnormal return | Median | Positive fraction |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, row in report["fixed_signal_groups"].items():
        lines.append(
            f"| {name} | {row['samples']:,} | {row['mean_abnormal_return_5']:+.2%} "
            f"| {row['median_abnormal_return_5']:+.2%} | {row['positive_fraction']:.1%} |"
        )
    lines.extend(["", "## Limits", ""])
    lines.extend(f"- {limit}" for limit in report["limitations"])
    lines.append("")
    atomic_text(path, "\n".join(lines))


if __name__ == "__main__":
    main()
