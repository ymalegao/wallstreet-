"""Cost-aware portfolio evaluation through the daily v0 pipeline."""

from __future__ import annotations

import asyncio
import csv
import json
from collections import defaultdict
from datetime import date
from math import prod, sqrt
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import lightgbm as lgb
import polars as pl

from ws.broker import CostModel
from ws.pipeline_v0.run import BacktestResult, run_backtest
from ws.store.pit_bars import PITBarStore

_ROOT = Path(__file__).parents[3]
_PIPELINE_TEMPLATE = _ROOT / "configs" / "pipeline" / "training-v1.json"


def run_portfolio_scenarios(
    start: date,
    end: date,
    *,
    model_id: str,
    output_dir: Path,
    store_path: Path,
    universe_path: Path,
    fold: str | None = None,
    initial_cash: float = 1000.0,
) -> dict[str, Any]:
    """Run model and momentum portfolios at one and two times the same cost model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for multiplier in (1, 2):
        model_name = f"lgbm@{model_id}" + (f"#fold={fold}" if fold else "")
        model_result = _run_one(
            start,
            end,
            name="lightgbm",
            alpha_model_id=model_name,
            output_dir=output_dir / f"lightgbm-{multiplier}x",
            multiplier=multiplier,
            store_path=store_path,
            universe_path=universe_path,
            initial_cash=initial_cash,
        )
        momentum_result = _run_one(
            start,
            end,
            name="momentum-12-1",
            alpha_model_id="momentum-12-1",
            output_dir=output_dir / f"momentum-{multiplier}x",
            multiplier=multiplier,
            store_path=store_path,
            universe_path=universe_path,
            initial_cash=initial_cash,
        )
        results[f"{multiplier}x"] = {
            "lightgbm": _run_metrics(model_result, initial_cash),
            "momentum_12_1": _run_metrics(momentum_result, initial_cash),
            "equal_weight_buy_hold": _baseline_metrics(model_result.baselines_path, "equal_weight_buy_hold"),
            "spy_buy_hold": _baseline_metrics(model_result.baselines_path, "spy_buy_hold"),
            "monthly": _combine_monthly(model_result, momentum_result),
            "concentration": _concentration(model_result.contributions_path),
            "artifacts": {
                "lightgbm": str(model_result.equity_path),
                "momentum_12_1": str(momentum_result.equity_path),
                "baselines": str(model_result.baselines_path),
                "contributions": str(model_result.contributions_path),
            },
        }
    return results


def _run_one(
    start: date,
    end: date,
    *,
    name: str,
    alpha_model_id: str,
    output_dir: Path,
    multiplier: int,
    store_path: Path,
    universe_path: Path,
    initial_cash: float,
) -> BacktestResult:
    config = json.loads(_PIPELINE_TEMPLATE.read_text(encoding="utf-8"))
    for item in config:
        if item["cls"] == "ws.pipeline_v0.nodes.Alpha":
            item["kwargs"]["model_id"] = alpha_model_id
        if item["cls"] == "ws.pipeline_v0.nodes.Execution":
            item["kwargs"]["rebalance_rule"] = "monthly"
    output_dir.mkdir(parents=True, exist_ok=False)
    config_path = output_dir / "pipeline.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    base = CostModel()
    costs = CostModel(
        commission_bps=base.commission_bps * multiplier,
        half_spread_bps=base.half_spread_bps * multiplier,
        impact_coefficient=base.impact_coefficient * multiplier,
    )
    return asyncio.run(
        run_backtest(
            start,
            end,
            store=PITBarStore(store_path),
            universe_path=universe_path,
            config_path=config_path,
            out_dir=output_dir / "backtest",
            initial_cash=initial_cash,
            cost_model=costs,
        )
    )


def predict_ensemble(models: list[lgb.Booster], frame: pl.DataFrame, feature_names: tuple[str, ...]) -> list[float]:
    if not models:
        raise ValueError("at least one fitted model is required")
    matrix = frame.select(feature_names).to_numpy()
    predictions = [model.predict(matrix) for model in models]
    return [sum(float(values[index]) for values in predictions) / len(predictions) for index in range(frame.height)]


def _run_metrics(result: BacktestResult, initial_cash: float) -> dict[str, float | None]:
    rows = _csv_rows(result.equity_path)
    equities = [float(row["equity"]) for row in rows]
    returns = [equities[index] / equities[index - 1] - 1.0 for index in range(1, len(equities))]
    fills = _csv_rows(result.fills_path)
    traded = sum(abs(float(row["shares"]) * float(row["price"])) for row in fills)
    costs = sum(float(row["cost"]) for row in fills)
    return _summarize_equity(equities, returns, initial_cash, traded, costs)


def _baseline_metrics(path: Path, strategy: str) -> dict[str, Any]:
    rows = [row for row in _csv_rows(path) if row["strategy"] == strategy]
    equities = [float(row["equity"]) for row in rows]
    returns = [float(row["net_return"]) for row in rows]
    initial_cash = equities[0] if equities else 0.0
    traded_notional = sum(float(row["turnover"]) for row in rows) * initial_cash
    costs = sum(float(row["transaction_costs"]) for row in rows)
    metrics: dict[str, Any] = _summarize_equity(equities, returns, initial_cash, traded_notional, costs)
    if rows:
        metrics["constituent_count"] = int(float(rows[0].get("constituent_count", "0")))
        metrics["excluded_symbols"] = rows[0].get("excluded_symbols", "")
    return metrics


def _summarize_equity(
    equities: list[float], returns: list[float], initial_cash: float, traded: float, costs: float
) -> dict[str, float | None]:
    if not equities:
        return {
            "total_return": None,
            "annualized_return": None,
            "annualized_volatility": None,
            "sharpe": None,
            "max_drawdown": None,
            "turnover": None,
            "total_costs": costs,
        }
    ending = equities[-1]
    years = len(returns) / 252.0
    annualized_return = (ending / initial_cash) ** (1.0 / years) - 1.0 if initial_cash > 0 and years > 0 else None
    volatility = stdev(returns) * sqrt(252) if len(returns) > 1 else 0.0
    sharpe = mean(returns) / stdev(returns) * sqrt(252) if len(returns) > 1 and stdev(returns) > 0 else None
    peak = initial_cash
    drawdown = 0.0
    for value in equities:
        peak = max(peak, value)
        drawdown = min(drawdown, value / peak - 1.0)
    return {
        "total_return": ending / initial_cash - 1.0 if initial_cash > 0 else None,
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "turnover": traded / mean(equities) if mean(equities) > 0 else None,
        "total_costs": costs,
    }


def _combine_monthly(model: BacktestResult, momentum: BacktestResult) -> list[dict[str, Any]]:
    series: dict[str, dict[str, float]] = {
        "lightgbm": _monthly_returns(model.equity_path),
        "momentum_12_1": _monthly_returns(momentum.equity_path),
    }
    for strategy in ("equal_weight_buy_hold", "spy_buy_hold"):
        series[strategy] = _monthly_returns(model.baselines_path, strategy=strategy)
    months = sorted(set().union(*(set(items) for items in series.values())))
    return [{"month": month, **{name: items.get(month) for name, items in series.items()}} for month in months]


def _monthly_returns(path: Path, *, strategy: str | None = None) -> dict[str, float]:
    rows = _csv_rows(path)
    if strategy is not None:
        rows = [row for row in rows if row["strategy"] == strategy]
    by_month: dict[str, list[float]] = defaultdict(list)
    previous_equity: float | None = None
    for row in rows:
        day = date.fromisoformat(row["date"])
        if "net_return" in row:
            daily_return = float(row["net_return"])
        else:
            equity = float(row["equity"])
            daily_return = equity / previous_equity - 1.0 if previous_equity else 0.0
            previous_equity = equity
        by_month[day.strftime("%Y-%m")].append(daily_return)
    return {month: prod(1.0 + value for value in values) - 1.0 for month, values in by_month.items()}


def _concentration(path: Path) -> dict[str, list[dict[str, Any]]]:
    rows = _csv_rows(path)
    by_symbol: dict[str, float] = defaultdict(float)
    by_month: dict[str, float] = defaultdict(float)
    for row in rows:
        contribution = float(row["contribution"])
        by_symbol[row["symbol"]] += contribution
        by_month[row["date"][:7]] += contribution
    top_stocks = sorted(by_symbol.items(), key=lambda item: item[1], reverse=True)[:5]
    top_months = sorted(by_month.items(), key=lambda item: item[1], reverse=True)[:5]
    return {
        "top_five_stocks": [{"symbol": symbol, "contribution": value} for symbol, value in top_stocks],
        "top_five_months": [{"month": month, "contribution": value} for month, value in top_months],
    }


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))
