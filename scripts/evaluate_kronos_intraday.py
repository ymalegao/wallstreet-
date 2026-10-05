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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--forecasts", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    jobs: list[dict[str, Any]] = json.loads(a.inputs.read_text())
    forecasts = json.loads(a.forecasts.read_text())
    if not forecasts["complete"]:
        raise ValueError("Intraday forecast set is incomplete")
    forecast_by_cycle_ticker = {(f["ticker"], f["cycle"]): f for f in forecasts["forecasts"]}

    tickers = sorted({job["ticker"] for job in jobs})
    daily = BarStore(Path("data"), "1Day_raw").read(tickers).sort("ts")
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
    output = []
    invalid = 0
    for job in jobs:
        if job["ticker"] == "SPY":
            continue
        key = (job["ticker"], job["cycle"])
        fc = forecast_by_cycle_ticker.get(key)
        spy_fc = forecast_by_cycle_ticker.get(("SPY", job["cycle"]))
        if not fc or not spy_fc or fc.get("status") != "ok" or spy_fc.get("status") != "ok":
            invalid += 1
            continue
        ticker, session = job["ticker"], date.fromisoformat(job["session"])
        target = date.fromisoformat(job["target_session"])
        history = daily_rows[ticker]
        prior = [row for row in history if row["session"] < session]
        if len(prior) < 252:
            continue
        prev = prior[-1]
        ref = float(job["history"][-1]["close"])
        spy_job = next(x for x in jobs if x["ticker"] == "SPY" and x["cycle"] == job["cycle"])
        spy_ref = float(spy_job["history"][-1]["close"])
        actual_close = close_by_session.get((ticker, target))
        spy_close = close_by_session.get(("SPY", target))
        if actual_close is None or spy_close is None:
            continue
        actual = actual_close / ref - 1
        spy_actual = spy_close / spy_ref - 1
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
        spy_prev_row = next((row for row in daily_rows["SPY"] if row["session"] == prev["session"]), None)
        if ticker_session_row is None or spy_prev_row is None:
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
        spy_prev_close = float(spy_prev_row["close"])
        output.append(
            {
                "ticker": ticker,
                "session": str(session),
                "cycle": job["cycle"],
                "interval_minutes": job["interval_minutes"],
                "target_session": str(target),
                "actual_excess": actual - spy_actual,
                "forecast_excess": float(fc["mean_return"]) - float(spy_fc["mean_return"]),
                "forecast_return": float(fc["mean_return"]),
                "forecast_dispersion": float(fc["return_std"]),
                "realized_vol": realized_vol,
                "atr14": float(np.mean(true_ranges) / float(prev["close"])),
                "reversal20": -(past_close[-1] / past_close[-21] - 1),
                "momentum12_1": past_close[-22] / past_close[-252] - 1,
                "below_ma400": -(past_close[-1] / np.mean(past_close[-400:]) - 1),
                "current_day_return": ref / float(prev["close"]) - 1,
                "current_day_vs_spy": (ref / float(prev["close"]) - 1) - (spy_ref / spy_prev_close - 1),
                "open_gap": float(ticker_session_row["open"]) / float(prev["close"]) - 1,
                "intraday_return_to_cycle": ref / float(ticker_session_row["open"]) - 1,
            }
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        groups[row["cycle"]].append(row)
    features = [
        "forecast_excess",
        "reversal20",
        "momentum12_1",
        "below_ma400",
        "current_day_return",
        "current_day_vs_spy",
        "open_gap",
        "intraday_return_to_cycle",
    ]
    metrics: dict[str, dict[str, Any]] = {}
    for feature in [*features, "forecast_dispersion", "atr14"]:
        corr_target = "realized_vol" if feature in {"forecast_dispersion", "atr14"} else "actual_excess"
        values = []
        for cycle, rows in sorted(groups.items()):
            if len(rows) < 6:
                continue
            corr = spearmanr([r[feature] for r in rows], [r[corr_target] for r in rows]).statistic
            if np.isfinite(corr):
                values.append((cycle, float(corr)))
        ordered = sorted(values, key=lambda x: x[0])
        ic = [x[1] for x in ordered]
        midpoint = len(ic) // 2
        metrics[feature] = {
            "mean_cycle_rank_ic": float(np.mean(ic)) if ic else None,
            "median_cycle_rank_ic": float(np.median(ic)) if ic else None,
            "positive_cycle_fraction": float(np.mean(np.asarray(ic) > 0)) if ic else None,
            "block5_bootstrap_95pct_ci": block_ci(ic),
            "first_half_mean_ic": float(np.mean(ic[:midpoint])) if midpoint else None,
            "second_half_mean_ic": float(np.mean(ic[midpoint:])) if midpoint else None,
            "cycles": len(ic),
        }

    # Partial out cheap price context on each date, then test whether the
    # forecast still ranks residual returns. This is diagnostic, not a model.
    residual_ics: list[float] = []
    controls = ["reversal20", "below_ma400", "current_day_vs_spy", "open_gap"]
    for _, rows in sorted(groups.items()):
        if len(rows) < len(controls) + 4:
            continue
        design = np.asarray([[1.0, *[float(row[name]) for name in controls]] for row in rows])
        forecast_values = np.asarray([float(row["forecast_excess"]) for row in rows])
        actual_values = np.asarray([float(row["actual_excess"]) for row in rows])
        forecast_resid = forecast_values - design @ np.linalg.lstsq(design, forecast_values, rcond=None)[0]
        actual_resid = actual_values - design @ np.linalg.lstsq(design, actual_values, rcond=None)[0]
        corr = spearmanr(forecast_resid, actual_resid).statistic
        if np.isfinite(corr):
            residual_ics.append(float(corr))
    residual_midpoint = len(residual_ics) // 2
    metrics["forecast_excess_after_price_controls"] = {
        "mean_cycle_rank_ic": float(np.mean(residual_ics)) if residual_ics else None,
        "median_cycle_rank_ic": float(np.median(residual_ics)) if residual_ics else None,
        "positive_cycle_fraction": float(np.mean(np.asarray(residual_ics) > 0)) if residual_ics else None,
        "block5_bootstrap_95pct_ci": block_ci(residual_ics),
        "first_half_mean_ic": float(np.mean(residual_ics[:residual_midpoint])) if residual_midpoint else None,
        "second_half_mean_ic": float(np.mean(residual_ics[residual_midpoint:])) if residual_midpoint else None,
        "cycles": len(residual_ics),
    }

    limitations = [
        "Kronos inputs include completed intraday bars through each news-decision time; "
        "15m data is used at 09:45 and 30m at the afternoon cycle.",
        "Returns run from the decision-bar close to the close five sessions later, "
        "with simple SPY subtraction (beta=1).",
        "Within a date, subtracting the same SPY return from every ticker cannot change rank IC. "
        "Interpret rank metrics as stock-return ranking; market adjustment matters for absolute "
        "forecasts and thresholds.",
        "The 23-name watchlist, overlapping outcomes and candidate-only news cycles "
        "limit inference; no thresholds were optimized.",
        "Forecast paths and prices are vendor bars, not executable fills or quoted spreads.",
    ]
    report = {
        "status": "EXPLORATORY",
        "samples": len(output),
        "invalid_samples": invalid,
        "decision_cycles": len(groups),
        "by_interval": {
            str(interval): sum(row["interval_minutes"] == interval for row in output) for interval in [15, 30]
        },
        "metrics": metrics,
        "limitations": limitations,
    }
    atomic_text(a.output, json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Kronos at the news decision time",
        "",
        "**EXPLORATORY**",
        "",
        f"{len(output)} valid ticker-cycle samples; {len(groups)} decision cycles; "
        f"{invalid} invalid/missing model comparisons.",
        "",
        "| Feature | Mean cycle rank IC | Median IC | Positive cycles | 5-session block CI | First/second half |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for feature, m in metrics.items():
        ci = m["block5_bootstrap_95pct_ci"]
        ci_s = "n/a" if ci[0] is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
        half_text = f"{m['first_half_mean_ic']:+.3f}/{m['second_half_mean_ic']:+.3f}"
        lines.append(
            f"| {feature} | {m['mean_cycle_rank_ic']:+.3f} | {m['median_cycle_rank_ic']:+.3f} | "
            f"{m['positive_cycle_fraction']:.0%} | {ci_s} | {half_text} |"
        )
    lines.extend(["", "## Limitations", "", *[f"- {x}" for x in limitations]])
    atomic_text(a.output.with_suffix(".md"), "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
