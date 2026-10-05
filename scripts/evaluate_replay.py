"""Exploratory zero-shot replay. Fixed rules; no fitting or parameter selection on outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import date, datetime, timedelta
from datetime import time as wall_time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from ws import calendar as cal
from ws.labels import regular_hours
from ws.replay import Rules, performance, run_replay
from ws.store.atomic import atomic_text
from ws.store.event_store import BarStore
from ws.timeutil import ET


def prepare(
    signals: list[dict[str, Any]], root: Path, kronos_cache: Path
) -> tuple[list[dict[str, Any]], pl.DataFrame, dict[str, int]]:
    bars = regular_hours(BarStore(root, "15Min").read()).sort("ts")
    lookup = {(r["symbol"], r["ts"]): r for r in bars.iter_rows(named=True)}
    daily = (
        bars.with_columns(session=pl.col("ts").dt.convert_time_zone(str(ET)).dt.date())
        .group_by("symbol", "session")
        .agg(pl.col("high").max(), pl.col("low").min(), pl.col("open").first(), pl.col("close").last())
        .sort("session")
    )
    daily_lookup = {(r["symbol"], r["session"]): r for r in daily.iter_rows(named=True)}
    momentum_symbols = sorted({signal["ticker"] for signal in signals} | {"SPY"})
    daily_history = BarStore(root, "1Day_all").read(momentum_symbols)
    output = []
    excluded: Counter[str] = Counter()
    for s in signals:
        ticker = s["ticker"]
        cycle = datetime.fromisoformat(s["cycle"])
        session = date.fromisoformat(s["session"])
        entry = cycle + timedelta(minutes=15)
        exit_ts = cal.cycles_for_session(cal.add_sessions(session, 5))[1] + timedelta(minutes=15)
        entry_bar = lookup.get((ticker, entry))
        exit_bar = lookup.get((ticker, exit_ts))
        reference = lookup.get((ticker, cycle - timedelta(minutes=15)))
        market_entry = lookup.get(("SPY", entry))
        market_exit = lookup.get(("SPY", exit_ts))
        today = daily_lookup.get((ticker, session))
        market_today = daily_lookup.get(("SPY", session))
        if not all([entry_bar, exit_bar, reference, market_entry, market_exit, today, market_today]):
            excluded["missing exact entry/exit/reference bar"] += 1
            continue
        assert entry_bar is not None and exit_bar is not None and reference is not None
        assert market_entry is not None and market_exit is not None
        assert today is not None and market_today is not None
        hist = daily.filter((pl.col("symbol") == ticker) & (pl.col("session") < session)).tail(15)
        if hist.height < 15:
            excluded["ATR warm-up"] += 1
            continue
        prev = hist["close"].to_numpy()[:-1]
        high, low = hist["high"].to_numpy()[1:], hist["low"].to_numpy()[1:]
        tr = np.maximum(high - low, np.maximum(abs(high - prev), abs(low - prev)))
        previous_close = float(hist["close"][-1])
        atr = float(np.mean(tr)) / previous_close
        momentum_hist = (
            daily_history.filter((pl.col("symbol") == ticker) & (pl.col("ts").dt.date() < session)).sort("ts").tail(252)
        )
        if momentum_hist.height < 252:
            excluded["12-1 momentum warm-up"] += 1
            continue
        forecast_path = kronos_cache / f"{ticker}-{session}.json"
        if not forecast_path.exists():
            excluded["missing Kronos forecast"] += 1
            continue
        forecast = json.loads(forecast_path.read_text())
        if forecast is None or not isinstance(forecast, dict):
            excluded["invalid Kronos forecast"] += 1
            continue
        if forecast.get("status") != "ok" and forecast.get("status") is not None:
            excluded["invalid Kronos forecast"] += 1
            continue
        if datetime.fromisoformat(forecast["history_end"]).date() >= session:
            raise ValueError("Kronos used the current or a future daily bar")
        # Convert raw-price forecasts to the adjusted series using a prior-session scale.
        terminal = np.array(forecast["terminal_closes"]) / forecast["reference_close"] * previous_close
        p_up = float(np.mean(terminal > reference["close"]))
        ret = exit_bar["open"] / entry_bar["open"] - 1
        mret = market_exit["open"] / market_entry["open"] - 1
        same_day_ret = today["close"] / entry_bar["open"] - 1
        same_day_market = market_today["close"] / market_entry["open"] - 1
        output.append(
            {
                "ticker": ticker,
                "cycle_ts": cycle,
                "entry_ts": entry,
                "exit_ts": exit_ts,
                "feature_ts": cycle,
                "signal": s["signal"],
                "p_up": p_up,
                "atr_fraction": atr,
                "forward_return": ret,
                "market_return": mret,
                "market_adjusted_return": ret - mret,
                "same_day_return": same_day_ret,
                "same_day_market_return": same_day_market,
                "same_day_market_adjusted_return": same_day_ret - same_day_market,
                # Standard 12-1 momentum: trailing 12 months, skipping the latest month.
                "momentum": float(momentum_hist["close"][-21] / momentum_hist["close"][0] - 1),
            }
        )
    return output, bars, dict(excluded)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--signals", type=Path, default=Path("data/jev-signals.json"))
    ap.add_argument("--inputs", type=Path, default=Path("data/replay-inputs.json"))
    ap.add_argument("--manifest", type=Path, default=Path("data/research-manifest.json"))
    ap.add_argument("--kronos-cache-dir", type=Path, default=Path("data/kronos-cache"))
    ap.add_argument("--output", type=Path, default=Path("docs/replay-report.json"))
    ap.add_argument("--random-trials", type=int, default=30)
    ap.add_argument("--max-per-sector", type=int, default=2)
    a = ap.parse_args()
    root = Path("data")
    signals = json.loads(a.signals.read_text())
    if not signals["complete"]:
        raise ValueError("Scoring is incomplete; do not silently evaluate a partial run")
    raw = a.inputs.read_bytes()
    if hashlib.sha256(raw).hexdigest() != signals["input_sha256"]:
        raise ValueError("Frozen inputs changed after scoring")
    candidates, bars, excluded = prepare(signals["signals"], root, a.kronos_cache_dir)
    if not candidates:
        raise ValueError("No complete replay samples")
    manifest = json.loads(a.manifest.read_text())
    rules = Rules(max_per_sector=a.max_per_sector)
    begin = date.fromisoformat(manifest["start"])
    # End comparison when all scheduled holding windows have matured, not at an arbitrary later date.
    last_session = max(c["exit_ts"].astimezone(ET).date() for c in candidates)
    bars = bars.filter((pl.col("ts").dt.date() >= begin) & (pl.col("ts").dt.date() <= last_session))
    integrated = [c for c in candidates if c["signal"] >= 1 / 3 and c["p_up"] >= 0.5]
    selected = {
        "jev_kronos": integrated,
        "text_only": [c for c in candidates if c["signal"] >= 1 / 3],
        "kronos_only": [{**c, "signal": c["p_up"]} for c in candidates if c["p_up"] >= 0.5],
        "12_1_momentum": [{**c, "signal": c["momentum"]} for c in candidates if c["momentum"] > 0],
    }
    results = {name: run_replay(rows, bars, rules) for name, rows in selected.items()}
    random_runs = []
    for seed in range(a.random_trials):
        rows = random.Random(seed).sample(candidates, len(integrated))
        rows = [{**c, "signal": 1.0} for c in rows]
        run = run_replay(rows, bars, rules)
        random_runs.append({"seed": seed, **run["metrics"]})
    spy = bars.filter(pl.col("symbol") == "SPY").sort("ts")
    spy_daily = (
        spy.with_columns(session=pl.col("ts").dt.date())
        .group_by("session", maintain_order=True)
        .agg(pl.col("close").last())
    )
    cost = rules.cost_bps_per_side / 10000
    qty = rules.capital / (spy["open"][0] * (1 + cost))
    spy_values = [float(x) * qty for x in spy_daily["close"]]
    spy_values[-1] *= 1 - cost
    spy_result = performance(spy_values, rules.capital)
    sensitivity = {
        str(bps): run_replay(integrated, bars, Rules(max_per_sector=a.max_per_sector, cost_bps_per_side=bps))["metrics"]
        for bps in [5.0, 15.0, 30.0]
    }
    morning: dict[date, list[dict[str, Any]]] = {}
    for candidate in candidates:
        if candidate["cycle_ts"].astimezone(ET).time() == wall_time(9, 45):
            morning.setdefault(candidate["cycle_ts"].astimezone(ET).date(), []).append(candidate)

    def day_run(rows_by_day: dict[date, list[dict[str, Any]]]) -> dict[str, Any]:
        equity = Rules().capital
        daily_equity = []
        for day in sorted(morning):
            chosen = rows_by_day.get(day, [])[:5]
            net_returns = [row["same_day_return"] - 2 * Rules().cost_bps_per_side / 10_000 for row in chosen]
            equity *= 1 + 0.2 * sum(net_returns)
            daily_equity.append(equity)
        return performance(daily_equity, Rules().capital)

    day_picks: dict[str, dict[date, list[dict[str, Any]]]] = {
        "jev_kronos": {},
        "text_only": {},
        "kronos_only": {},
        "12_1_momentum": {},
    }
    for day, rows in morning.items():
        filters = {
            "jev_kronos": [row for row in rows if row["signal"] >= 1 / 3 and row["p_up"] >= 0.5],
            "text_only": [row for row in rows if row["signal"] >= 1 / 3],
            "kronos_only": [row for row in rows if row["p_up"] >= 0.5],
            "12_1_momentum": [row for row in rows if row["momentum"] > 0],
        }
        priorities = {
            "jev_kronos": lambda row: row["signal"] * row["p_up"],
            "text_only": lambda row: row["signal"],
            "kronos_only": lambda row: row["p_up"],
            "12_1_momentum": lambda row: row["momentum"],
        }
        for name, eligible_rows in filters.items():
            day_picks[name][day] = sorted(eligible_rows, key=priorities[name], reverse=True)[:5]
    day_strategies = {name: day_run(rows) for name, rows in day_picks.items()}
    spy_day_returns = []
    for day in sorted(morning):
        spy_market_return = morning[day][0]["same_day_market_return"]
        spy_day_returns.append(spy_market_return - 2 * Rules().cost_bps_per_side / 10_000)
    spy_equity = Rules().capital
    spy_curve = []
    for daily_return in spy_day_returns:
        spy_equity *= 1 + daily_return
        spy_curve.append(spy_equity)
    spy_day_metrics = performance(spy_curve, Rules().capital)
    integrated_day_counts = {day: len(rows) for day, rows in day_picks["jev_kronos"].items()}
    random_day_runs = []
    for seed in range(a.random_trials):
        rng = random.Random(seed)
        picks: dict[date, list[dict[str, Any]]] = {}
        for day, rows in morning.items():
            count = integrated_day_counts[day]
            picks[day] = rng.sample(rows, count) if count else []
        random_day_runs.append({"seed": seed, **day_run(picks)})
    random_day_returns = [row["total_return"] for row in random_day_runs]
    # Descriptive IC only; overlapping 5-session outcomes are not independent observations.
    ic = spearmanr([c["signal"] for c in candidates], [c["market_adjusted_return"] for c in candidates]).statistic
    kp = spearmanr([c["p_up"] for c in candidates], [c["forward_return"] for c in candidates]).statistic
    day_ic = spearmanr(
        [c["signal"] for c in candidates], [c["same_day_market_adjusted_return"] for c in candidates]
    ).statistic
    day_kp = spearmanr([c["p_up"] for c in candidates], [c["same_day_return"] for c in candidates]).statistic
    random_returns = [r["total_return"] for r in random_runs]
    report = {
        "status": "EXPLORATORY; not G1/G2 certification",
        "manifest": manifest,
        "model": signals["model"],
        "revision": signals["revision"],
        "sample_count": len(candidates),
        "excluded": excluded,
        "input_sha256": signals["input_sha256"],
        "entry_rule": "first bar open 15 minutes after decision cycle",
        "exit_rule": "first bar open 15 minutes after afternoon cycle five sessions later, or ATR stop",
        "rules": vars(rules),
        "cost_description": "15 bps adverse cost each side; quotes/liquidity not observed",
        "excluded_samples": excluded,
        "selected_counts": {name: len(rows) for name, rows in selected.items()},
        "strategies": results,
        "spy_buy_hold": spy_result,
        "same_day_strategies": day_strategies,
        "same_day_spy_intraday": spy_day_metrics,
        "same_day_random_controls": {
            "trials": len(random_day_runs),
            "median_return": float(np.median(random_day_returns)),
            "min_return": min(random_day_returns),
            "max_return": max(random_day_returns),
            "fraction_beat_spy_intraday": sum(x > spy_day_metrics["total_return"] for x in random_day_returns)
            / len(random_day_returns),
        },
        "random_controls": random_runs,
        "cost_sensitivity": sensitivity,
        "signal_metrics": {
            "spearman_text_vs_market_adjusted_5d": float(ic) if np.isfinite(ic) else None,
            "spearman_kronos_vs_return_5d": float(kp) if np.isfinite(kp) else None,
            "spearman_text_vs_same_day_market_adjusted_return": float(day_ic) if np.isfinite(day_ic) else None,
            "spearman_kronos_vs_same_day_return": float(day_kp) if np.isfinite(day_kp) else None,
        },
        "limitations": [
            "Only 23 liquid selected stocks and a short recent window; model training overlap unknown.",
            "Samples missing exact intraday bars, ATR warm-up, or valid Kronos forecasts are excluded; "
            "see excluded_samples.",
            "Historical revised text conservatively delayed; assumed five-minute news latency remains unmeasured.",
            "Adjusted bars provide a total-return proxy, not actual broker fills, quotes or dividend cash accounting.",
            "Fixed thresholds were not optimized. This interval is exploratory and is not a final holdout.",
            "Random controls match candidate count and use identical rules; realized trade counts/exposure can differ.",
            "Five-day overlapping outcomes are dependent; descriptive IC is not a significance test.",
            "Kronos daily close forecasts and intraday exits have different exact horizons.",
            "No broker, order book, partial fill or market impact simulation; no live/paper orders.",
            "Text scores select the strongest absolute event per cycle; contradictory events can cancel in reality.",
            "Same-day return diagnostics and the capped toy basket omit sector caps, stops, quotes, "
            "and measured market impact.",
            "The same-day toy replay uses only 09:45 ET candidates, a five-position 20%-equity cap, "
            "and exits at the close. It has no sector cap, stop, quotes or measured market impact.",
        ],
    }
    atomic_text(a.output, json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Exploratory zero-shot market replay",
        "",
        f"**{report['status']}**",
        "",
        f"{len(candidates)} ticker-cycle samples; {begin} through {last_session}. Initial equity $500.",
        "",
        "| Strategy | Return | Max drawdown | Closed trades | Open positions |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        m = result["metrics"]
        lines.append(
            f"| {name} | {m['total_return']:.2%} | {m['max_drawdown']:.2%} "
            f"| {m['closed_trades']} | {m['open_positions']} |"
        )
    lines += [
        f"| SPY buy-and-hold | {spy_result['total_return']:.2%} | {spy_result['max_drawdown']:.2%} | — | — |",
        "",
        f"Random-entry controls ({len(random_runs)} seeds): median return {np.median(random_returns):.2%}, "
        f"range {min(random_returns):.2%} to {max(random_returns):.2%}.",
        "",
        "## Signal diagnostics",
        "",
        json.dumps(report["signal_metrics"], indent=2),
        "",
        "## Same-day 09:45 ET basket (toy day-trading comparison)",
        "",
        "At most five positions, 20% equity each, entered at the next 15-minute open and exited at the "
        "regular-session close; 15 bps per side.",
        "",
        "| Strategy | Return | Max drawdown |",
        "|---|---:|---:|",
        *[
            f"| {name} | {metrics['total_return']:.2%} | {metrics['max_drawdown']:.2%} |"
            for name, metrics in day_strategies.items()
        ],
        f"| SPY same-time intraday | {spy_day_metrics['total_return']:.2%} | {spy_day_metrics['max_drawdown']:.2%} |",
        f"| Random controls median | {np.median(random_day_returns):.2%} | — |",
        f"Random-control range: {min(random_day_returns):.2%} to {max(random_day_returns):.2%}; "
        f"{sum(x > spy_day_metrics['total_return'] for x in random_day_returns)}"
        f"/{len(random_day_returns)} beat the SPY intraday baseline.",
        "",
        "## Costs",
        "",
        "Per-side cost sensitivity for combined strategy:",
    ]
    lines += [f"- {bps} bps: {m['total_return']:.2%}" for bps, m in sensitivity.items()]
    lines += ["", "## Limitations", "", *[f"- {x}" for x in report["limitations"]]]
    atomic_text(a.output.with_suffix(".md"), "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
