"""Evaluate fixed broad-equity momentum and buy-and-hold controls without fitting a model."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from datetime import date
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import Any

import polars as pl

from ws import universe
from ws.store.atomic import atomic_text
from ws.store.event_store import BarStore


def total_return(returns: list[float]) -> float:
    value = 1.0
    for ret in returns:
        value *= 1 + ret
    return value - 1


def drawdown(returns: list[float]) -> float:
    value = peak = 1.0
    worst = 0.0
    for ret in returns:
        value *= 1 + ret
        peak = max(peak, value)
        worst = min(worst, value / peak - 1)
    return worst


def month_ends(sessions: list[date]) -> list[date]:
    by_month: dict[tuple[int, int], date] = {}
    for session in sessions:
        by_month[(session.year, session.month)] = session
    return list(by_month.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=20, help="number of monthly 12–1 momentum names")
    parser.add_argument("--random-trials", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("docs/broad-universe-report.json"))
    args = parser.parse_args()
    if args.top < 1 or args.random_trials < 1:
        raise ValueError("top and random-trials must be positive")

    root = Path("data")
    metadata_path = sorted((root / "universe").glob("alpaca_assets_*.jsonl"))[-1]
    assets = [json.loads(line) for line in metadata_path.read_text().splitlines()]
    stock_assets = universe.listed_stock_candidates(assets)
    stocks = {row["symbol"] for row in stock_assets}
    master_symbols = {str(row.get("symbol") or "") for row in assets}
    raw = BarStore(root, "1Day_raw").read()
    all_adjusted = BarStore(root, "1Day_all").read()
    if raw.is_empty() or all_adjusted.is_empty():
        raise RuntimeError("Need both raw and all-adjusted broad daily bars")
    total_adjusted_rows = all_adjusted.height
    adjusted_symbols = all_adjusted["symbol"].n_unique()
    cached_symbols = set(raw["symbol"].unique().to_list())

    eligible = universe.eligibility(raw).filter(pl.col("eligible")).select("symbol", "session")
    adjusted = all_adjusted.filter(pl.col("symbol").is_in(stocks | {"SPY"})).with_columns(
        session=pl.col("ts").dt.date()
    )
    momentum = (
        adjusted.sort("symbol", "session")
        .with_columns(
            momentum=pl.col("close").shift(21).over("symbol") / pl.col("close").shift(252).over("symbol") - 1
        )
        .select("symbol", "session", "open", "close", "momentum")
    )
    rows = momentum.iter_rows(named=True)
    price: dict[tuple[str, date], dict[str, Any]] = {}
    closes_by_symbol: dict[str, dict[date, float]] = defaultdict(dict)
    sessions_set: set[date] = set()
    for row in rows:
        key = (row["symbol"], row["session"])
        price[key] = row
        closes_by_symbol[row["symbol"]][row["session"]] = float(row["close"])
        if row["symbol"] == "SPY":
            sessions_set.add(row["session"])
    sessions = sorted(sessions_set)
    month_end_list = month_ends(sessions)
    pairs = list(pairwise(month_end_list))
    elig_by_date: dict[date, set[str]] = defaultdict(set)
    for eligibility_row in eligible.iter_rows(named=True):
        if eligibility_row["symbol"] in stocks:
            elig_by_date[eligibility_row["session"]].add(eligibility_row["symbol"])

    months: list[dict[str, Any]] = []
    for signal_date, exit_date in pairs:
        idx = sessions.index(signal_date)
        if idx + 1 >= len(sessions):
            continue
        entry_date = sessions[idx + 1]
        if ("SPY", entry_date) not in price or ("SPY", exit_date) not in price:
            continue
        ranked = []
        for symbol in elig_by_date.get(signal_date, set()):
            price_row = price.get((symbol, signal_date))
            if price_row and price_row["momentum"] is not None and math.isfinite(price_row["momentum"]):
                ranked.append((symbol, float(price_row["momentum"])))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        if len(ranked) < args.top:
            continue
        months.append(
            {
                "signal_date": signal_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "eligible": [symbol for symbol, _ in ranked],
                "top": [symbol for symbol, _ in ranked[: args.top]],
            }
        )
    if not months:
        raise RuntimeError("No monthly windows with sufficient point-in-time eligible history")

    def run(picks: list[list[str]], cost_bps_per_side: float = 0.0) -> dict[str, Any]:
        monthly: list[float] = []
        for window, symbols in zip(months, picks, strict=True):
            symbol_returns = []
            for symbol in symbols:
                start = price.get((symbol, window["entry_date"]))
                end = price.get((symbol, window["exit_date"]))
                if start and end and start["open"] > 0:
                    symbol_returns.append(float(end["close"]) / float(start["open"]) - 1)
            if not symbol_returns:
                raise RuntimeError(f"No portfolio prices for {window['entry_date']}..{window['exit_date']}")
            gross = sum(symbol_returns) / len(symbol_returns)
            # Fixed full round trip per month; conservative even where holdings repeat.
            monthly.append(gross - 2 * cost_bps_per_side / 10_000)
        return {
            "total_return": total_return(monthly),
            "monthly_max_drawdown": drawdown(monthly),
            "monthly_returns": monthly,
        }

    top_picks = [window["top"] for window in months]
    all_picks = [window["eligible"] for window in months]
    top_gross = run(top_picks)
    top_costed = run(top_picks, 15.0)
    all_gross = run(all_picks)
    first_entry, last_exit = months[0]["entry_date"], months[-1]["exit_date"]
    spy_start = price[("SPY", first_entry)]["open"]
    spy_end = price[("SPY", last_exit)]["close"]
    spy_return = float(spy_end) / float(spy_start) - 1
    spy_path = [price[("SPY", session)]["close"] for session in sessions if first_entry <= session <= last_exit]
    spy_daily_returns = [float(spy_path[0]) / float(spy_start) - 1]
    spy_daily_returns.extend(float(b) / float(a) - 1 for a, b in pairwise(spy_path))
    spy_drawdown = drawdown(spy_daily_returns)
    random_returns = []
    for seed in range(args.random_trials):
        rng = random.Random(seed)
        random_picks = [rng.sample(window["eligible"], args.top) for window in months]
        random_returns.append(run(random_picks)["total_return"])

    report = {
        "status": "EXPLORATORY price-only baseline; not model alpha or certification",
        "asset_master": metadata_path.name,
        "start": str(first_entry),
        "end": str(last_exit),
        "total_raw_bar_rows": raw.height,
        "total_adjusted_bar_rows": total_adjusted_rows,
        "symbols_with_raw_bars": raw["symbol"].n_unique(),
        "symbols_with_adjusted_bars": adjusted_symbols,
        "stock_proxy_symbols_with_raw_bars": len(stocks & cached_symbols),
        "stock_proxy_symbols_without_raw_bars": len(stocks - cached_symbols),
        "delisted_symbol_coverage_checks": {
            symbol: {"in_current_asset_master": symbol in master_symbols, "has_daily_bars": symbol in cached_symbols}
            for symbol in ["SIVB", "FRC", "BBBY", "GME"]
        },
        "current_stock_proxy_count": len(stocks),
        "current_session_eligible_count": len(elig_by_date.get(max(elig_by_date), set())),
        "monthly_windows": len(months),
        "top_n": args.top,
        "rules": {
            "eligibility": "Prior-session close >= $5 and prior 20-session median dollar volume >= $20m.",
            "momentum": "Adjusted close at t-21 divided by adjusted close at t-252, minus one.",
            "entry": "Next market session open after month-end signal.",
            "exit": "Following month-end close.",
            "positioning": "Equal weight; no leverage or shorts.",
            "costed_case": "Subtracts 15 bps per side each month, assuming full turnover.",
        },
        "results": {
            "top_12_1_momentum_equal_weight_gross": {
                "total_return": top_gross["total_return"],
                "monthly_max_drawdown": top_gross["monthly_max_drawdown"],
            },
            "top_12_1_momentum_equal_weight_15bps_each_side": {
                "total_return": top_costed["total_return"],
                "monthly_max_drawdown": top_costed["monthly_max_drawdown"],
            },
            "all_eligible_with_12_1_history_equal_weight_gross": {
                "total_return": all_gross["total_return"],
                "monthly_max_drawdown": all_gross["monthly_max_drawdown"],
            },
            "SPY_buy_and_hold": {"total_return": spy_return, "daily_max_drawdown": spy_drawdown},
            "random_top_n_controls": {
                "trials": args.random_trials,
                "median_return": median(random_returns),
                "min_return": min(random_returns),
                "max_return": max(random_returns),
                "fraction_beat_SPY": sum(value > spy_return for value in random_returns) / len(random_returns),
            },
        },
        "monthly_selections": [
            {
                "signal_date": str(window["signal_date"]),
                "entry_date": str(window["entry_date"]),
                "exit_date": str(window["exit_date"]),
                "eligible_count": len(window["eligible"]),
                "top_symbols": window["top"],
            }
            for window in months
        ],
        "limitations": [
            "Alpaca's current active/inactive security master is not a historical point-in-time security master.",
            "SIVB, FRC and BBBY are absent from both the current asset snapshot and this bar cache; "
            "historical delisted coverage is incomplete.",
            "The listed-stock proxy excludes obvious funds/derivatives by exchange, ticker syntax and name; "
            "some may remain.",
            "Daily OHLC bars omit quotes, intraday volatility, fills, borrow costs, taxes and market impact.",
            "No feature thresholds were tuned here, but this is still one exploratory period and one fixed rule.",
            "The random controls are descriptive, and monthly observations are not independent significance tests.",
        ],
    }
    atomic_text(args.output, json.dumps(report, indent=2, allow_nan=False) + "\n")
    summary = [
        "# Broad-universe price-only baseline",
        "",
        f"**{report['status']}**",
        "",
        f"{report['start']} through {report['end']}; {len(months)} monthly windows; "
        f"{len(stocks)} current listed-stock proxy symbols; {report['current_session_eligible_count']} "
        f"pass the point-in-time price/liquidity screen on the last date. "
        f"Daily bars exist for {report['stock_proxy_symbols_with_raw_bars']} of the proxy symbols.",
        "",
        "| Baseline | Return | Max drawdown |",
        "|---|---:|---:|",
        f"| 12–1 momentum, top {args.top}, gross | {top_gross['total_return']:.1%} "
        f"| {top_gross['monthly_max_drawdown']:.1%} |",
        f"| 12–1 momentum, top {args.top}, 15 bps/side | {top_costed['total_return']:.1%} "
        f"| {top_costed['monthly_max_drawdown']:.1%} |",
        f"| All eligible with 12–1 history, equal weight | {all_gross['total_return']:.1%} "
        f"| {all_gross['monthly_max_drawdown']:.1%} |",
        f"| SPY buy-and-hold | {spy_return:.1%} | {spy_drawdown:.1%} |",
        f"| Random top-{args.top} median ({args.random_trials} trials) "
        f"| {median(random_returns):.1%} | — |",
        "",
        "This is a price-only benchmark. It does not test JEV, Kronos, intraday execution, or a learned strategy.",
        "Drawdown is measured from monthly portfolio marks for rebalanced stock strategies and daily closes for SPY.",
        f"Random portfolios ranged from {min(random_returns):.1%} to {max(random_returns):.1%}; "
        f"{sum(value > spy_return for value in random_returns)}/{len(random_returns)} beat SPY.",
        "",
        *[f"- {item}" for item in report["limitations"]],
    ]
    atomic_text(args.output.with_suffix(".md"), "\n".join(summary) + "\n")
    print("\n".join(summary), flush=True)


if __name__ == "__main__":
    main()
