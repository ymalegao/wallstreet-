"""Evaluate fixed Kronos forecasts as features against same-date simple controls."""

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

from ws.store.atomic import atomic_text
from ws.store.event_store import BarStore


def block_ci(values: list[float], seed: int = 42, block: int = 5) -> list[float | None]:
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(x) < 8:
        return [None, None]
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(3000):
        draw: list[float] = []
        while len(draw) < len(x):
            start = int(rng.integers(0, len(x)))
            draw.extend(x[(start + np.arange(block)) % len(x)].tolist())
        estimates.append(float(np.mean(draw[: len(x)])))
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, default=Path("data/broad-kronos-sweep-inputs.json"))
    ap.add_argument("--sweep", action="append", required=True, help="label=forecast-result.json")
    ap.add_argument("--output", type=Path, default=Path("docs/kronos-feature-report.json"))
    a = ap.parse_args()

    jobs = json.loads(a.inputs.read_text())
    job_by_key = {(job["ticker"], job["session"]): job for job in jobs}
    symbols = sorted({job["ticker"] for job in jobs})
    daily = BarStore(Path("data"), "1Day_raw").read(symbols).with_columns(session=pl.col("ts").dt.date())
    bars = {(row["symbol"], row["session"]): row for row in daily.iter_rows(named=True)}

    settings: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    failures: dict[str, int] = {}
    for item in a.sweep:
        label, path = item.split("=", 1)
        data = json.loads(Path(path).read_text())
        if not data["complete"]:
            raise ValueError(f"Incomplete sweep {label}")
        settings[label] = {(x["ticker"], x["session"]): x for x in data["features"]}
        failures[label] = sum(x.get("status") != "ok" for x in data["features"])

    by_setting: dict[str, list[dict[str, Any]]] = {}
    all_dates = sorted({job["session"] for job in jobs if job["ticker"] != "SPY"})
    for label, forecasts in settings.items():
        spy_by_date = {session: forecasts.get(("SPY", session)) for session in all_dates}
        rows = []
        for key, job in job_by_key.items():
            ticker, session = key
            if ticker == "SPY":
                continue
            fc = forecasts.get(key)
            spy_fc = spy_by_date.get(session)
            if not fc or not spy_fc or fc.get("status") != "ok" or spy_fc.get("status") != "ok":
                continue
            history = job["history"]
            ref = float(history[-1]["close"])
            future_dates = [date.fromisoformat(x) for x in job["future_sessions"]]
            future = [bars.get((ticker, d)) for d in future_dates]
            spy_future = [bars.get(("SPY", d)) for d in future_dates]
            if any(x is None for x in future) or any(x is None for x in spy_future):
                continue
            future_rows = [x for x in future if x is not None]
            spy_future_rows = [x for x in spy_future if x is not None]
            actual_prices = [ref, *[float(x["close"]) for x in future_rows]]
            spy_ref = float(job_by_key[("SPY", session)]["history"][-1]["close"])
            spy_prices = [spy_ref, *[float(x["close"]) for x in spy_future_rows]]
            actual = actual_prices[-1] / ref - 1
            spy_actual = spy_prices[-1] / spy_ref - 1
            returns = np.diff(np.log(actual_prices))
            realized_vol = float(np.std(returns, ddof=1)) if len(returns) > 1 else float("nan")
            prior_close = np.asarray([float(x["close"]) for x in history], dtype=float)
            true_ranges = []
            for i in range(-14, 0):
                prev_close = prior_close[i - 1]
                high = float(history[i]["high"])
                low = float(history[i]["low"])
                true_ranges.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
            atr14 = float(np.mean(true_ranges) / ref)
            mean_path_return = float(np.mean(np.asarray(fc["terminal_closes"]) / ref - 1))
            rows.append(
                {
                    "ticker": ticker,
                    "session": session,
                    "actual_excess": actual - spy_actual,
                    "forecast_excess": mean_path_return - float(spy_fc["mean_return"]),
                    "forecast_return": mean_path_return,
                    "forecast_dispersion": float(fc["return_std"]),
                    "realized_vol": realized_vol,
                    "atr14": atr14,
                    "reversal20": -(prior_close[-1] / prior_close[-21] - 1),
                    "momentum12_1": prior_close[-22] / prior_close[-252] - 1,
                    "below_ma400": -(prior_close[-1] / np.mean(prior_close[-400:]) - 1),
                }
            )
        by_setting[label] = rows

    feature_keys = [
        "forecast_excess",
        "reversal20",
        "momentum12_1",
        "below_ma400",
    ]
    vol_keys = ["forecast_dispersion", "atr14"]
    detail: dict[str, Any] = {}
    for label, rows in by_setting.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row["session"]].append(row)
        date_order = sorted(grouped)
        metrics: dict[str, Any] = {}
        for feature in [*feature_keys, *vol_keys]:
            target = "realized_vol" if feature in vol_keys else "actual_excess"
            daily_ic = []
            for day in date_order:
                cross = grouped[day]
                if len(cross) < 8:
                    continue
                corr = spearmanr([x[feature] for x in cross], [x[target] for x in cross]).statistic
                if np.isfinite(corr):
                    daily_ic.append(float(corr))
            split = len(daily_ic) // 2
            metrics[feature] = {
                "mean_daily_rank_ic": float(np.mean(daily_ic)) if daily_ic else None,
                "median_daily_rank_ic": float(np.median(daily_ic)) if daily_ic else None,
                "positive_date_fraction": float(np.mean(np.asarray(daily_ic) > 0)) if daily_ic else None,
                "block5_bootstrap_95pct_ci": block_ci(daily_ic),
                "first_half_mean_ic": float(np.mean(daily_ic[:split])) if split else None,
                "second_half_mean_ic": float(np.mean(daily_ic[split:])) if split else None,
                "dates": len(daily_ic),
            }
        detail[label] = {
            "samples": len(rows),
            "dates": len({row["session"] for row in rows}),
            "invalid_forecasts": failures[label],
            "metrics": metrics,
        }

    report = {
        "status": "EXPLORATORY",
        "purpose": "Daily Kronos features forecast from prior close; not the intraday news-time setup.",
        "settings": detail,
        "limitations": [
            "Forecasts use daily bars through the prior session, omitting the current news-day price reaction.",
            "Within a date, subtracting the same SPY return from every ticker cannot change rank IC. "
            "Interpret rank metrics as stock-return ranking; market adjustment matters for absolute "
            "forecasts and thresholds.",
            "Date-level rank ICs use a 23-name proxy watchlist and overlapping 5-session outcomes.",
            "Temperature/lookback settings were informed by a preceding 12-date sweep; "
            "this comparison is not a pristine holdout.",
            "No portfolio thresholds were tuned and no live or paper trading was performed.",
        ],
    }
    atomic_text(a.output, json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Kronos daily feature sweep",
        "",
        "**EXPLORATORY**",
        "",
        "This uses only data through the prior daily close. It tests market-adjusted cross-sectional "
        "ranking and volatility features, not the intraday news-time setup.",
        "",
        "| Setting | Features | Samples | Dates | Mean daily rank IC | Median IC | Positive dates | "
        "5-day block CI | First/second half IC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, result in detail.items():
        for feature in feature_keys + vol_keys:
            m = result["metrics"][feature]
            ci = m["block5_bootstrap_95pct_ci"]
            ci_s = "n/a" if ci[0] is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
            half = f"{m['first_half_mean_ic']:+.3f}/{m['second_half_mean_ic']:+.3f}"
            lines.append(
                f"| {label} | {feature} | {result['samples']} | {m['dates']} | "
                f"{m['mean_daily_rank_ic']:+.3f} | {m['median_daily_rank_ic']:+.3f} | "
                f"{m['positive_date_fraction']:.0%} | {ci_s} | {half} |"
            )
    lines.extend(["", "## Limitations", "", *[f"- {x}" for x in report["limitations"]]])
    atomic_text(a.output.with_suffix(".md"), "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
