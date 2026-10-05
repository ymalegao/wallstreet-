"""Evaluate decision-time Kronos outputs against matched SPY returns and cheap price signals."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from ws.labels import regular_hours
from ws.store.atomic import atomic_text
from ws.store.event_store import BarStore


def block_ci(values: list[float], seed: int = 42, block: int = 5) -> list[float | None]:
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(x) < 8:
        return [None, None]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(3000):
        sample: list[float] = []
        while len(sample) < len(x):
            start = int(rng.integers(0, len(x)))
            sample.extend(x[(start + np.arange(block)) % len(x)].tolist())
        draws.append(float(np.mean(sample[: len(x)])))
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def rank_correlation(left: list[float] | np.ndarray, right: list[float] | np.ndarray) -> float:
    x, y = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if not len(x) or not len(y) or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def format_optional(value: Any, spec: str) -> str:
    return format(value, spec) if value is not None else "n/a"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--forecasts", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    jobs: list[dict[str, Any]] = json.loads(a.inputs.read_text())
    bootstrap_block_cycles = 5 * len({int(job["interval_minutes"]) for job in jobs})
    decision_sessions = sorted({job["session"] for job in jobs})
    forecasts = json.loads(a.forecasts.read_text())
    if not forecasts["complete"]:
        raise ValueError("Intraday forecast set is incomplete")
    forecast_by_cycle_ticker = {(f["ticker"], f["cycle"]): f for f in forecasts["forecasts"]}

    tickers = sorted({job["ticker"] for job in jobs})
    daily = BarStore(Path("data"), "1Day_all").read(tickers).sort("ts")
    daily_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in daily.iter_rows(named=True):
        daily_rows[row["symbol"]].append({**row, "session": row["ts"].date()})

    rth15 = regular_hours(BarStore(Path("data"), "15Min").read(tickers)).with_columns(
        session=pl.col("ts").dt.convert_time_zone("America/New_York").dt.date()
    )
    close_by_session = {
        (row["symbol"], row["session"]): float(row["close"])
        for row in (
            rth15.sort("ts")
            .group_by("symbol", "session", maintain_order=True)
            .agg(pl.col("close").last())
            .iter_rows(named=True)
        )
    }
    job_by_cycle_ticker = {(job["ticker"], job["cycle"]): job for job in jobs}
    output = []
    invalid = 0
    insufficient_history = 0
    missing_labels = 0
    for job in jobs:
        if job["ticker"] == "SPY":
            continue
        key = (job["ticker"], job["cycle"])
        fc = forecast_by_cycle_ticker.get(key)
        spy_job = job_by_cycle_ticker.get(("SPY", job["cycle"]))
        spy_fc = forecast_by_cycle_ticker.get(("SPY", job["cycle"]))
        if not fc or fc.get("status") != "ok" or not spy_job or not spy_fc or spy_fc.get("status") != "ok":
            invalid += 1
            continue
        ticker, session = job["ticker"], date.fromisoformat(job["session"])
        target = date.fromisoformat(job["target_session"])
        history = daily_rows[ticker]
        prior = [row for row in history if row["session"] < session]
        if len(prior) < 400:
            insufficient_history += 1
            continue
        prev = prior[-1]
        ref = float(job["history"][-1]["close"])
        actual_close = close_by_session.get((ticker, target))
        spy_close = close_by_session.get(("SPY", target))
        if actual_close is None or spy_close is None:
            missing_labels += 1
            continue
        actual = actual_close / ref - 1
        spy_ref = float(spy_job["history"][-1]["close"])
        spy_actual = spy_close / spy_ref - 1
        forecast_return = float(fc["mean_return"])
        spy_forecast_return = float(spy_fc["mean_return"])
        past_close = np.asarray([float(row["close"]) for row in prior])
        true_ranges = []
        for idx in range(-14, 0):
            row = prior[idx]
            prev_close = float(prior[idx - 1]["close"])
            true_ranges.append(
                max(
                    float(row["high"]) - float(row["low"]),
                    abs(float(row["high"]) - prev_close),
                    abs(float(row["low"]) - prev_close),
                )
            )
        ticker_session_row = next((row for row in daily_rows[ticker] if row["session"] == session), None)
        if ticker_session_row is None:
            continue
        realized_prices = [ref]
        for day_row in [row for row in daily_rows[ticker] if session <= row["session"] <= target]:
            close = close_by_session.get((ticker, day_row["session"]))
            if close is None:
                realized_prices = []
                break
            realized_prices.append(close)
        if len(realized_prices) < 2:
            continue
        realized_vol = float(np.std(np.diff(np.log(realized_prices)), ddof=1))
        output.append(
            {
                "ticker": ticker,
                "session": str(session),
                "cycle": job["cycle"],
                "interval_minutes": job["interval_minutes"],
                "target_session": str(target),
                "actual_return": actual,
                "forecast_return": forecast_return,
                "actual_excess": actual - spy_actual,
                "forecast_excess": forecast_return - spy_forecast_return,
                "spy_actual_return": spy_actual,
                "spy_forecast_return": spy_forecast_return,
                "forecast_dispersion": float(fc["return_std"]),
                "realized_vol": realized_vol,
                "atr14": float(np.mean(true_ranges) / float(prev["close"])),
                "reversal20": -(past_close[-1] / past_close[-21] - 1),
                "momentum12_1": past_close[-22] / past_close[-252] - 1,
                "below_ma400": -(past_close[-1] / np.mean(past_close[-400:]) - 1),
                "current_day_return": ref / float(prev["close"]) - 1,
                "open_gap": float(ticker_session_row["open"]) / float(prev["close"]) - 1,
                "intraday_return_to_cycle": ref / float(ticker_session_row["open"]) - 1,
            }
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        groups[row["cycle"]].append(row)
    features = [
        "forecast_return",
        "reversal20",
        "momentum12_1",
        "below_ma400",
        "current_day_return",
        "open_gap",
        "intraday_return_to_cycle",
    ]
    metrics: dict[str, dict[str, Any]] = {}
    for feature in [*features, "forecast_dispersion", "atr14"]:
        corr_target = "realized_vol" if feature in {"forecast_dispersion", "atr14"} else "actual_return"
        values = []
        for cycle, rows in sorted(groups.items()):
            if len(rows) < 6:
                continue
            corr = rank_correlation([r[feature] for r in rows], [r[corr_target] for r in rows])
            if np.isfinite(corr):
                values.append((cycle, float(corr)))
        ordered = sorted(values, key=lambda x: x[0])
        ic = [x[1] for x in ordered]
        midpoint = len(ic) // 2
        metrics[feature] = {
            "mean_cycle_rank_ic": float(np.mean(ic)) if ic else None,
            "median_cycle_rank_ic": float(np.median(ic)) if ic else None,
            "positive_cycle_fraction": float(np.mean(np.asarray(ic) > 0)) if ic else None,
            "block5_bootstrap_95pct_ci": block_ci(ic, block=bootstrap_block_cycles),
            "first_half_mean_ic": float(np.mean(ic[:midpoint])) if midpoint else None,
            "second_half_mean_ic": float(np.mean(ic[midpoint:])) if midpoint else None,
            "cycles": len(ic),
        }

    # A common SPY return is constant across a cycle, so raw-return and
    # SPY-relative cross-sectional rank ICs must match within floating error.
    excess_ics: list[float] = []
    rank_ic_differences: list[float] = []
    for _, rows in sorted(groups.items()):
        if len(rows) < 6:
            continue
        raw_corr = rank_correlation([r["forecast_return"] for r in rows], [r["actual_return"] for r in rows])
        excess_corr = rank_correlation([r["forecast_excess"] for r in rows], [r["actual_excess"] for r in rows])
        if np.isfinite(raw_corr) and np.isfinite(excess_corr):
            excess_ics.append(float(excess_corr))
            rank_ic_differences.append(float(abs(raw_corr - excess_corr)))
    metrics["forecast_excess"] = {
        "mean_cycle_rank_ic": float(np.mean(excess_ics)) if excess_ics else None,
        "median_cycle_rank_ic": float(np.median(excess_ics)) if excess_ics else None,
        "positive_cycle_fraction": float(np.mean(np.asarray(excess_ics) > 0)) if excess_ics else None,
        "block5_bootstrap_95pct_ci": block_ci(excess_ics, block=bootstrap_block_cycles),
        "first_half_mean_ic": float(np.mean(excess_ics[: len(excess_ics) // 2])) if len(excess_ics) >= 2 else None,
        "second_half_mean_ic": float(np.mean(excess_ics[len(excess_ics) // 2 :])) if len(excess_ics) >= 2 else None,
        "cycles": len(excess_ics),
    }
    rank_invariance = {
        "cycles_compared": len(rank_ic_differences),
        "max_abs_raw_vs_spy_relative_rank_ic_difference": max(rank_ic_differences, default=None),
    }

    # Partial out cheap price context on each date, then test whether the
    # forecast still ranks residual returns. This is diagnostic, not a model.
    residual_ics: list[float] = []
    controls = ["reversal20", "below_ma400", "current_day_return", "open_gap"]
    for _, rows in sorted(groups.items()):
        if len(rows) < len(controls) + 4:
            continue
        design = np.asarray([[1.0, *[float(row[name]) for name in controls]] for row in rows])
        forecast_values = np.asarray([float(row["forecast_excess"]) for row in rows])
        actual_values = np.asarray([float(row["actual_excess"]) for row in rows])
        forecast_resid = forecast_values - design @ np.linalg.lstsq(design, forecast_values, rcond=None)[0]
        actual_resid = actual_values - design @ np.linalg.lstsq(design, actual_values, rcond=None)[0]
        corr = rank_correlation(forecast_resid, actual_resid)
        if np.isfinite(corr):
            residual_ics.append(float(corr))
    residual_midpoint = len(residual_ics) // 2
    metrics["forecast_excess_after_price_controls"] = {
        "mean_cycle_rank_ic": float(np.mean(residual_ics)) if residual_ics else None,
        "median_cycle_rank_ic": float(np.median(residual_ics)) if residual_ics else None,
        "positive_cycle_fraction": float(np.mean(np.asarray(residual_ics) > 0)) if residual_ics else None,
        "block5_bootstrap_95pct_ci": block_ci(residual_ics, block=bootstrap_block_cycles),
        "first_half_mean_ic": float(np.mean(residual_ics[:residual_midpoint])) if residual_midpoint else None,
        "second_half_mean_ic": float(np.mean(residual_ics[residual_midpoint:])) if residual_midpoint else None,
        "cycles": len(residual_ics),
    }

    predicted_excess = np.asarray([float(row["forecast_excess"]) for row in output])
    realized_excess = np.asarray([float(row["actual_excess"]) for row in output])
    zero_baseline_mae = float(np.mean(np.abs(realized_excess))) if len(output) else None
    forecast_mae = float(np.mean(np.abs(predicted_excess - realized_excess))) if len(output) else None
    absolute_market_adjusted = {
        "samples": len(output),
        "mean_forecast_excess": float(np.mean(predicted_excess)) if len(output) else None,
        "mean_realized_excess": float(np.mean(realized_excess)) if len(output) else None,
        "mean_absolute_error": forecast_mae,
        "zero_forecast_baseline_mae": zero_baseline_mae,
        "mae_skill_vs_zero_baseline": (
            1.0 - forecast_mae / zero_baseline_mae
            if forecast_mae is not None and zero_baseline_mae
            else None
        ),
        "directional_accuracy": (
            float(np.mean(np.sign(predicted_excess) == np.sign(realized_excess))) if len(output) else None
        ),
    }

    limitations = [
        "Kronos inputs include completed intraday bars through each news-decision time; "
        "15m data is used at 09:45 and 30m at the afternoon cycle.",
        "Both raw and SPY-relative stock returns are reported. Subtracting one common SPY value "
        "per date cannot change cross-sectional rank IC, though paired SPY forecasts are retained "
        "for absolute market-relative thresholds and future portfolio decisions.",
        "The point-in-time candidate universe uses a current asset snapshot, so it has survivorship "
        "bias and is not a complete historical listing universe. Outcomes overlap; no thresholds "
        "were optimized. Rows with fewer than 400 prior daily sessions are omitted to keep the "
        "400-session moving-average control comparable.",
        "Alpaca all-adjusted bars are used for model inputs and labels, so outcomes are adjusted "
        "returns rather than raw spot-price returns or executable fills. They cannot stand in for "
        "historical option-contract pricing.",
        "Forecast paths and prices are vendor bars, not executable fills or quoted spreads.",
    ]
    report = {
        "status": "EXPLORATORY",
        "samples": len(output),
        "invalid_samples": invalid,
        "excluded_insufficient_400_session_history": insufficient_history,
        "excluded_missing_five_session_label": missing_labels,
        "decision_cycles": len(groups),
        "decision_session_range": {
            "first": decision_sessions[0] if decision_sessions else None,
            "last": decision_sessions[-1] if decision_sessions else None,
        },
        "bootstrap_block_cycles": bootstrap_block_cycles,
        "by_interval": {
            str(interval): sum(row["interval_minutes"] == interval for row in output) for interval in [15, 30]
        },
        "metrics": metrics,
        "absolute_market_adjusted_forecasts": absolute_market_adjusted,
        "spy_rank_invariance": rank_invariance,
        "limitations": limitations,
    }
    atomic_text(a.output, json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Kronos at the news decision time",
        "",
        "**EXPLORATORY**",
        "",
        f"{len(output)} valid ticker-cycle samples; {len(groups)} decision cycles; "
        f"{invalid} invalid/missing model comparisons; {insufficient_history} excluded for fewer than "
        f"400 prior daily sessions; {missing_labels} missing five-session labels.",
        (
            f"Prepared decision sessions: {decision_sessions[0]} to {decision_sessions[-1]}"
            if decision_sessions
            else "No prepared decision sessions."
        ),
        "Paired SPY subtraction changed rank IC by at most "
        f"{rank_invariance['max_abs_raw_vs_spy_relative_rank_ic_difference'] or 0:.3g} "
        "across evaluated cycles.",
        "",
        "| Feature | Mean cycle rank IC | Median IC | Positive cycles | 5-session block CI | First/second half |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for feature, m in metrics.items():
        ci = m["block5_bootstrap_95pct_ci"]
        ci_s = "n/a" if ci[0] is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
        half_text = (
            f"{format_optional(m['first_half_mean_ic'], '+.3f')}/"
            f"{format_optional(m['second_half_mean_ic'], '+.3f')}"
        )
        lines.append(
            f"| {feature} | {format_optional(m['mean_cycle_rank_ic'], '+.3f')} | "
            f"{format_optional(m['median_cycle_rank_ic'], '+.3f')} | "
            f"{format_optional(m['positive_cycle_fraction'], '.0%')} | {ci_s} | {half_text} |"
        )
    lines.extend(
        [
            "",
            "## Absolute SPY-relative forecasts",
            "",
            f"Mean forecast excess: {format_optional(absolute_market_adjusted['mean_forecast_excess'], '+.3%')}; "
            f"mean realized excess: {format_optional(absolute_market_adjusted['mean_realized_excess'], '+.3%')}; "
            f"MAE: {format_optional(absolute_market_adjusted['mean_absolute_error'], '.3%')}; "
            f"zero-forecast MAE: {format_optional(absolute_market_adjusted['zero_forecast_baseline_mae'], '.3%')}; "
            f"directional accuracy: {format_optional(absolute_market_adjusted['directional_accuracy'], '.1%')}.",
            "These fixed summaries use no tuned buy threshold and do not include transaction costs.",
        ]
    )
    lines.extend(["", "## Limitations", "", *[f"- {x}" for x in limitations]])
    atomic_text(a.output.with_suffix(".md"), "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
