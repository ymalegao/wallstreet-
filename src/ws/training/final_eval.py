"""Single-use holdout evaluation; this is the only holdout data reader."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import lightgbm as lgb
import polars as pl

from ws.features.px_v1 import PX_V1_FEATURES
from ws.store.history import BarHistory
from ws.store.pit_bars import PITBarStore
from ws.training.evaluation import predict_ensemble, run_portfolio_scenarios
from ws.training.final_fit import _ROOT, _model_config_sha256
from ws.training.metrics import rank_ic_by_date, summarize_series
from ws.training.registry import append_registry, registry_timestamp, reserve_holdout
from ws.training.table import _build_rows, _decision_time, _session_dates, _snapshot_session_start

_REGISTRY = _ROOT / "models" / "registry.jsonl"


def final_eval(model_id: str) -> dict[str, Any]:
    """Reserve, read, score, and backtest the holdout once for a final model id."""
    if not model_id or model_id in {".", ".."} or any(part in model_id for part in ("/", "\\")):
        raise ValueError("model_id must be a plain directory name")
    model_dir = _ROOT / "models" / model_id
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"final model manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_id") != model_id or manifest.get("feature_set") != "px-v1":
        raise ValueError("model manifest identity or feature set is invalid")
    if tuple(manifest.get("feature_names", ())) != PX_V1_FEATURES:
        raise ValueError("model manifest feature list is incompatible")
    if _manifest_hash(model_dir, manifest) != manifest.get("config_sha256"):
        raise ValueError("model config or model files changed after fit-final")

    store_path = Path(manifest["source_store_path"])
    universe_path = Path(manifest["universe_path"])
    if hashlib.sha256(universe_path.read_bytes()).hexdigest() != manifest.get("universe_sha256"):
        raise ValueError("universe file changed after training")
    snapshot_path = store_path / "snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if snapshot.get("snapshot_id") != manifest.get("data_snapshot_id") or snapshot.get("feed") != manifest.get(
        "market_data_feed"
    ):
        raise ValueError("data snapshot differs from the fitted model snapshot")
    universe_data = json.loads(universe_path.read_text(encoding="utf-8"))
    universe = tuple(sorted({str(symbol).strip().upper() for symbol in universe_data["symbols"]}))
    if snapshot.get("feed") not in {"iex", "sip"} or snapshot.get("adjustment") != "all":
        raise ValueError("holdout evaluation requires the fitted adjusted IEX or SIP snapshot")
    latest_values = [date.fromisoformat(value) for value in snapshot["actual_latest_by_symbol"].values()]
    latest_session = max(latest_values)
    holdout_start = date.fromisoformat(manifest["holdout_start"])
    labeled_end = date.fromisoformat(manifest["labeled_end"])
    sessions = _session_dates(_snapshot_session_start(snapshot), latest_session)
    through = _decision_time(latest_session)
    reserve_holdout(_REGISTRY, model_id, str(manifest["config_sha256"]))
    history = BarHistory.load(PITBarStore(store_path), (*universe, "SPY"), through=through)
    visible = history.visible(through, symbols=(*universe, "SPY"))
    holdout_rows = _build_rows(
        visible,
        universe,
        sessions,
        holdout_start=holdout_start,
        first_eligible_session=holdout_start,
        horizon=int(manifest["label"]["horizon"]),
        allow_holdout=True,
    ).filter((pl.col("date") >= holdout_start) & (pl.col("date") <= labeled_end))
    if holdout_rows.is_empty():
        raise ValueError("no labeled holdout rows were available")
    if holdout_rows.filter(pl.col("label_exit_date") < holdout_start).height:
        raise AssertionError("holdout rows unexpectedly include pre-holdout label outcomes")
    models = [lgb.Booster(model_file=str(model_dir / f"seed_{int(seed)}.txt")) for seed in manifest["seeds"]]
    predictions = predict_ensemble(models, holdout_rows, PX_V1_FEATURES)
    model_ic_rows = rank_ic_by_date(holdout_rows, predictions)
    momentum_ic_rows = rank_ic_by_date(holdout_rows, holdout_rows["mom_12_1"].to_list())
    portfolio = run_portfolio_scenarios(
        holdout_start,
        labeled_end,
        model_id=model_id,
        output_dir=_ROOT / "runs" / "training" / model_id / "holdout",
        store_path=store_path,
        universe_path=universe_path,
    )
    ic_by_month = {
        item["date"].strftime("%Y-%m"): float(item["rank_ic"]) for item in model_ic_rows
    }
    momentum_ic_by_month = {
        item["date"].strftime("%Y-%m"): float(item["rank_ic"]) for item in momentum_ic_rows
    }
    base_monthly = portfolio["1x"]["monthly"]
    monthly = []
    for row in base_monthly:
        month = str(row["month"])
        monthly.append(
            {
                **row,
                "lightgbm_rank_ic": ic_by_month.get(month),
                "momentum_12_1_rank_ic": momentum_ic_by_month.get(month),
            }
        )
    result = {
        "model_id": model_id,
        "data_snapshot_id": manifest["data_snapshot_id"],
        "market_data_feed": manifest["market_data_feed"],
        "trial_count": int(manifest["trial_count"]),
        "holdout_range": [holdout_start.isoformat(), labeled_end.isoformat()],
        "survivorship_biased_universe": bool(manifest.get("survivorship_biased", True)),
        "rows": holdout_rows.height,
        "prediction_quality": {
            "lightgbm": summarize_series(item["rank_ic"] for item in model_ic_rows),
            "momentum_12_1": summarize_series(item["rank_ic"] for item in momentum_ic_rows),
        },
        "monthly": monthly,
        "portfolio_by_cost_multiple": portfolio,
        "leakage_probes": {
            "training_labels_end_before_holdout": "pass",
            "holdout_outcomes_read_only_by_final_eval": "pass",
        },
    }
    run_dir = _ROOT / "runs" / "training" / model_id
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "report.json"
    existing = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    report = {**existing, "trial_count": int(manifest["trial_count"]), "final_eval": result}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    serialized_report = json.loads(report_path.read_text(encoding="utf-8"))
    (run_dir / "report.md").write_text(_render_report(serialized_report), encoding="utf-8")
    append_registry(
        _REGISTRY,
        [
            {
                "record_type": "holdout_result",
                "model_id": model_id,
                "created_at": registry_timestamp(),
                "config_sha256": manifest["config_sha256"],
                "holdout_range": result["holdout_range"],
                "prediction_quality": result["prediction_quality"],
                "portfolio_by_cost_multiple": result["portfolio_by_cost_multiple"],
            }
        ],
    )
    return result


def _manifest_hash(model_dir: Path, manifest: dict[str, Any]) -> str:
    config = {
        key: manifest[key]
        for key in (
            "feature_set",
            "market_data_feed",
            "feature_names",
            "label",
            "params",
            "seeds",
            "num_boost_round",
            "num_threads",
            "iteration_policy",
        )
    }
    actual_hashes = {
        filename: hashlib.sha256((model_dir / filename).read_bytes()).hexdigest()
        for filename in manifest["model_files_sha256"]
    }
    if actual_hashes != manifest["model_files_sha256"]:
        return "invalid-model-artifacts"
    return _model_config_sha256(
        config,
        actual_hashes,
        data_snapshot_id=str(manifest["data_snapshot_id"]),
        universe_sha256=str(manifest["universe_sha256"]),
        holdout_start=str(manifest["holdout_start"]),
        labeled_end=str(manifest["labeled_end"]),
        train_range=list(manifest["train_range"]),
    )


def _render_report(report: dict[str, Any]) -> str:
    """Render the report markdown from the JSON record using a fixed template."""
    lines = [f"# Training report: {report.get('model_id', 'unknown')}", ""]
    if "data_snapshot_id" in report:
        lines.append(f"- Data snapshot: `{report['data_snapshot_id']}`")
    if "market_data_feed" in report:
        quality_note = (
            " (single-venue development data; not for live trading decisions)"
            if report["market_data_feed"] == "iex"
            else ""
        )
        lines.append(f"- Market data feed: `{report['market_data_feed']}`{quality_note}")
    if "trial_count" in report:
        lines.append(f"- Trials: {report['trial_count']}")
    if "table_date_range" in report:
        lines.append(f"- Pre-holdout table: `{report['table_date_range'][0]} to {report['table_date_range'][1]}`")
        lines.append(f"- Holdout starts: `{report['holdout_start']}`")
    if "universe_id" in report:
        lines.append(
            f"- Universe: `{report['universe_id']}` (survivorship biased: "
            f"{report['survivorship_biased_universe']})"
        )
    if "placebo" in report:
        placebo = report["placebo"]
        lines.append(
            f"- Placebo mean test rank IC: {_format(placebo['rank_ic']['mean'])} "
            f"({ 'pass' if placebo['pass'] else 'fail' }) using {placebo['label_scope']} labels"
        )
    folds = report.get("folds", [])
    if folds:
        lines.extend(
            [
                "",
                "## Walk-forward folds",
                "",
                "| Fold | Test period | Mean rank IC | Std dev | t-stat | Positive fraction |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for fold in folds:
            metrics = fold["rank_ic"]
            lines.append(
                f"| {fold['fold']} | {fold['test_range'][0]} to {fold['test_range'][1]} | "
                f"{_format(metrics.get('mean'))} | {_format(metrics.get('std'))} | "
                f"{_format(metrics.get('t_stat'))} | {_format(metrics.get('positive_fraction'))} |"
            )
        lines.extend(
            [
                "",
                "## Walk-forward portfolio metrics",
                "",
                "| Fold | Costs | Strategy | Total return | Ann. return | Ann. volatility | Sharpe | "
                "Max drawdown | Turnover | Costs paid |",
                "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for fold in folds:
            for multiple, scenario in fold["portfolio"].items():
                for strategy in ("lightgbm", "momentum_12_1", "equal_weight_buy_hold", "spy_buy_hold"):
                    metrics = scenario[strategy]
                    lines.append(
                        f"| {fold['fold']} | {multiple} | {strategy} | {_format(metrics.get('total_return'))} | "
                        f"{_format(metrics.get('annualized_return'))} | "
                        f"{_format(metrics.get('annualized_volatility'))} | {_format(metrics.get('sharpe'))} | "
                        f"{_format(metrics.get('max_drawdown'))} | {_format(metrics.get('turnover'))} | "
                        f"{_format(metrics.get('total_costs'))} |"
                    )
        lines.extend(["", "## Monthly fold detail", ""])
        for fold in folds:
            lines.extend(
                [
                    f"### Fold {fold['fold']}",
                    "",
                    "| Month | LightGBM IC | Momentum IC | LightGBM return | Momentum return | Equal weight | SPY |",
                    "|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            ic_rows = {item["date"]: item for item in fold["monthly_rank_ic"]}
            monthly_rows = {item["month"]: item for item in fold["portfolio"]["1x"]["monthly"]}
            for month in sorted(set(ic_rows) | set(monthly_rows)):
                ic = ic_rows.get(month, {})
                pnl = monthly_rows.get(month, {})
                lines.append(
                    f"| {month} | {_format(ic.get('lightgbm'))} | {_format(ic.get('momentum_12_1'))} | "
                    f"{_format(pnl.get('lightgbm'))} | {_format(pnl.get('momentum_12_1'))} | "
                    f"{_format(pnl.get('equal_weight_buy_hold'))} | {_format(pnl.get('spy_buy_hold'))} |"
                )
            concentration = fold["portfolio"]["1x"]["concentration"]
            lines.extend(["", "Top stock contributions at 1× costs:"])
            lines.extend(
                f"- {item['symbol']}: {_format(item['contribution'])}"
                for item in concentration.get("top_five_stocks", [])
            )
            lines.append("Top monthly contributions at 1× costs:")
            lines.extend(
                f"- {item['month']}: {_format(item['contribution'])}"
                for item in concentration.get("top_five_months", [])
            )
    final = report.get("final_eval")
    if final:
        leakage = final["leakage_probes"]
        labels_status = leakage["training_labels_end_before_holdout"]
        outcomes_status = leakage["holdout_outcomes_read_only_by_final_eval"]
        prediction_rows = []
        for strategy in ("lightgbm", "momentum_12_1"):
            metrics = final["prediction_quality"][strategy]
            prediction_rows.append(
                f"| {strategy} | {_format(metrics.get('mean'))} | {_format(metrics.get('std'))} | "
                f"{_format(metrics.get('t_stat'))} | {_format(metrics.get('positive_fraction'))} |"
            )
        lines.extend(
            [
                "",
                "## Locked holdout",
                "",
                f"- Holdout period: {final['holdout_range'][0]} to {final['holdout_range'][1]}",
                f"- Survivorship-biased universe: {final['survivorship_biased_universe']}",
                "",
                "### Prediction quality",
                "",
                "| Strategy | Mean rank IC | Std dev | t-stat | Positive IC fraction |",
                "|---|---:|---:|---:|---:|",
                *prediction_rows,
                "",
                "### Leakage checks",
                "",
                f"- Training labels end before holdout: `{labels_status}`",
                f"- Holdout outcomes read only by final-eval: `{outcomes_status}`",
                "",
                "### Portfolio metrics by cost scenario",
                "",
                "| Costs | Strategy | Total return | Annualized return | Volatility | Sharpe | "
                "Max drawdown | Turnover | Costs paid |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for multiple, scenario in final["portfolio_by_cost_multiple"].items():
            for strategy, metrics in scenario.items():
                if strategy in {"monthly", "concentration", "artifacts"}:
                    continue
                lines.append(
                    f"| {multiple} | {strategy} | {_format(metrics.get('total_return'))} | "
                    f"{_format(metrics.get('annualized_return'))} | {_format(metrics.get('annualized_volatility'))} | "
                    f"{_format(metrics.get('sharpe'))} | {_format(metrics.get('max_drawdown'))} | "
                    f"{_format(metrics.get('turnover'))} | {_format(metrics.get('total_costs'))} |"
                )
        lines.extend(["", "### Monthly IC and portfolio returns", "", _markdown_table(final["monthly"]), ""])
        for multiple, scenario in final["portfolio_by_cost_multiple"].items():
            concentration = scenario.get("concentration", {})
            lines.extend(["", f"### Concentration at {multiple} costs", "", "Largest stock contributions:"])
            lines.extend(
                f"- {item['symbol']}: {_format(item['contribution'])}"
                for item in concentration.get("top_five_stocks", [])
            )
            lines.append("Largest monthly contributions:")
            lines.extend(
                f"- {item['month']}: {_format(item['contribution'])}"
                for item in concentration.get("top_five_months", [])
            )
    return "\n".join(lines)


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No monthly rows were available."
    fields = list(rows[0])
    heading = "| " + " | ".join(fields) + " |"
    separator = "|" + "|".join("---" for _ in fields) + "|"
    body = [
        "| "
        + " | ".join(
            _format(value) if isinstance(value, (float, int)) or value is None else str(value)
            for value in row.values()
        )
        + " |"
        for row in rows
    ]
    return "\n".join([heading, separator, *body])


def _format(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"
