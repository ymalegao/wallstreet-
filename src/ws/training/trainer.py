"""Fixed-grid LightGBM walk-forward training."""

from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

import lightgbm as lgb
import polars as pl

from ws.features.px_v1 import PX_V1_FEATURES
from ws.training.evaluation import run_portfolio_scenarios
from ws.training.metrics import rank_ic_by_date, summarize_series
from ws.training.placebo import PlaceboLabelScope, permute_fold_labels
from ws.training.registry import append_registry, count_trials, registry_timestamp
from ws.training.splits import WalkForwardFold, build_folds
from ws.training.table import load_pre_holdout_table

SEEDS = (0, 1, 2, 3, 4)
PLACEBO_SEED = 40_001
CONFIGS = tuple(
    {
        "num_leaves": leaves,
        "min_data_in_leaf": minimum,
        "lambda_l2": l2,
    }
    for leaves in (15, 31)
    for minimum in (50, 200)
    for l2 in (0, 10)
)
_ROOT = Path(__file__).parents[3]
_REGISTRY = _ROOT / "models" / "registry.jsonl"


def run_walk_forward(
    table_path: Path,
    output_root: Path,
    *,
    placebo_label_scope: PlaceboLabelScope,
    num_threads: int = 4,
) -> dict[str, Any]:
    """Run the fixed grid over expanding folds and register every config/seed/fold trial."""
    if num_threads < 1:
        raise ValueError("num_threads must be positive")
    metadata_path = table_path.with_suffix(table_path.suffix + ".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"training table metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("feature_set") != "px-v1" or tuple(metadata.get("feature_names", ())) != PX_V1_FEATURES:
        raise ValueError("training table feature metadata is incompatible with px-v1")
    frame = load_pre_holdout_table(table_path, holdout_start=date.fromisoformat(metadata["holdout_start"]))
    horizon = int(metadata["label_horizon"])
    folds = build_folds(frame, horizon=horizon)
    if not folds:
        raise ValueError("no walk-forward fold has at least three years of prior training data")
    model_id = f"pxv1-{str(metadata['snapshot_id'])[:12]}"
    run_dir = output_root / model_id
    model_dir = _ROOT / "models" / model_id
    if run_dir.exists() or model_dir.exists():
        raise FileExistsError(f"training output already exists for model id {model_id}")
    run_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    commit = _git_commit()
    registry_rows: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []

    for fold in folds:
        outcomes: list[dict[str, Any]] = []
        config_validation: dict[str, list[float]] = {}
        config_entries: dict[str, list[dict[str, Any]]] = {}
        config_models: dict[str, list[lgb.Booster]] = {}
        config_test_predictions: dict[str, list[list[float]]] = {}
        for config in CONFIGS:
            config_key = _config_key(config)
            config_validation[config_key] = []
            config_entries[config_key] = []
            config_models[config_key] = []
            config_test_predictions[config_key] = []
            for seed in SEEDS:
                model, best_iteration = _fit_one(fold, config, seed, num_threads=num_threads)
                valid_predictions = model.predict(_features(fold.valid), num_iteration=best_iteration)
                test_predictions = model.predict(_features(fold.test), num_iteration=best_iteration)
                valid_ic = summarize_series(
                    item["rank_ic"] for item in rank_ic_by_date(fold.valid, valid_predictions)
                )["mean"]
                test_ic = summarize_series(
                    item["rank_ic"] for item in rank_ic_by_date(fold.test, test_predictions)
                )["mean"]
                if valid_ic is not None:
                    config_validation[config_key].append(float(valid_ic))
                config_models[config_key].append(model)
                config_test_predictions[config_key].append([float(value) for value in test_predictions])
                entry = {
                    "record_type": "trial",
                    "model_id": model_id,
                    "created_at": registry_timestamp(),
                    "data_snapshot_id": metadata["snapshot_id"],
                    "market_data_feed": metadata["market_data_feed"],
                    "feature_set": "px-v1",
                    "label": {"horizon": horizon, "type": "xs_rank_open_to_open"},
                    "fold": fold.fold,
                    "train_range": [date.isoformat() for date in fold.train_range],
                    "valid_range": [date.isoformat() for date in fold.valid_range],
                    "test_range": [date.isoformat() for date in fold.test_range],
                    "params": _full_params(config, seed, num_threads),
                    "seed": seed,
                    "best_iteration": best_iteration,
                    "valid_rank_ic": valid_ic,
                    "test_rank_ic": test_ic,
                    "selected": False,
                    "git_commit": commit,
                }
                config_entries[config_key].append(entry)
                outcomes.append(entry)

        validation_means = {
            key: sum(values) / len(values) if values else float("-inf")
            for key, values in config_validation.items()
        }
        best_validation_mean = max(validation_means.values())
        best_candidates = [key for key, value in validation_means.items() if value == best_validation_mean]
        if len(best_candidates) != 1:
            raise ValueError(
                f"fold {fold.fold} has a validation rank-IC tie among configurations {best_candidates}; "
                "resolve the selection rule in docs/training-v1-questions.md before continuing"
            )
        selected_key = best_candidates[0]
        for entry in config_entries[selected_key]:
            entry["selected"] = True
        selected_dir = model_dir / "folds" / fold.fold
        selected_dir.mkdir(parents=True, exist_ok=True)
        for seed, model in zip(SEEDS, config_models[selected_key], strict=True):
            model.save_model(str(selected_dir / f"seed_{seed}.txt"))
        (selected_dir / "manifest.json").write_text(
            json.dumps(
                {"feature_set": "px-v1", "feature_names": list(PX_V1_FEATURES), "seeds": list(SEEDS)},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        selected_predictions = [
            sum(seed_values[index] for seed_values in config_test_predictions[selected_key]) / len(SEEDS)
            for index in range(fold.test.height)
        ]
        model_ic = rank_ic_by_date(fold.test, selected_predictions)
        momentum_ic = rank_ic_by_date(fold.test, fold.test["mom_12_1"].to_list())
        ic_values = [item["rank_ic"] for item in model_ic]
        model_monthly_ic = {item["date"].strftime("%Y-%m"): float(item["rank_ic"]) for item in model_ic}
        momentum_monthly_ic = {
            item["date"].strftime("%Y-%m"): float(item["rank_ic"]) for item in momentum_ic
        }
        portfolio_results = run_portfolio_scenarios(
            fold.test_range[0],
            fold.test_range[1],
            model_id=model_id,
            fold=fold.fold,
            output_dir=run_dir / "folds" / fold.fold / "portfolio",
            store_path=Path(metadata["source_store_path"]),
            universe_path=Path(metadata["universe_path"]),
        )
        for scenario in portfolio_results.values():
            scenario["monthly"] = [
                {
                    **row,
                    "lightgbm_rank_ic": model_monthly_ic.get(str(row["month"])),
                    "momentum_12_1_rank_ic": momentum_monthly_ic.get(str(row["month"])),
                }
                for row in scenario["monthly"]
            ]
        fold_reports.append(
            {
                "fold": fold.fold,
                "train_range": [value.isoformat() for value in fold.train_range],
                "valid_range": [value.isoformat() for value in fold.valid_range],
                "test_range": [value.isoformat() for value in fold.test_range],
                "selected_params": json.loads(selected_key),
                "rank_ic": summarize_series(ic_values),
                "monthly_rank_ic": [
                    {
                        "date": month,
                        "lightgbm": model_monthly_ic.get(month),
                        "momentum_12_1": momentum_monthly_ic.get(month),
                    }
                    for month in sorted(set(model_monthly_ic) | set(momentum_monthly_ic))
                ],
                "portfolio": portfolio_results,
            }
        )
        registry_rows.extend(outcomes)

    placebo_fold = permute_fold_labels(folds[0], seed=PLACEBO_SEED, scope=placebo_label_scope)
    placebo_predictions: list[list[float]] = []
    placebo_rows: list[dict[str, Any]] = []
    placebo_config = fold_reports[0]["selected_params"]
    for seed in SEEDS:
        model, best_iteration = _fit_one(placebo_fold, placebo_config, seed, num_threads=num_threads)
        test_predictions = [
            float(value) for value in model.predict(_features(placebo_fold.test), num_iteration=best_iteration)
        ]
        placebo_predictions.append(test_predictions)
        per_seed_ic = summarize_series(
            item["rank_ic"] for item in rank_ic_by_date(folds[0].test, test_predictions)
        )["mean"]
        placebo_rows.append(
            {
                "record_type": "placebo_trial",
                "model_id": model_id,
                "created_at": registry_timestamp(),
                "data_snapshot_id": metadata["snapshot_id"],
                "market_data_feed": metadata["market_data_feed"],
                "feature_set": "px-v1",
                "label": {"horizon": horizon, "type": "xs_rank_open_to_open"},
                "fold": folds[0].fold,
                "train_range": [value.isoformat() for value in folds[0].train_range],
                "valid_range": [value.isoformat() for value in folds[0].valid_range],
                "test_range": [value.isoformat() for value in folds[0].test_range],
                "params": {
                    **_full_params(placebo_config, seed, num_threads),
                    "placebo_label_scope": placebo_label_scope,
                    "placebo_permutation_seed": PLACEBO_SEED,
                },
                "seed": seed,
                "best_iteration": best_iteration,
                "valid_rank_ic": None,
                "test_rank_ic": per_seed_ic,
                "selected": False,
                "git_commit": commit,
            }
        )
    placebo_mean_predictions = [
        sum(seed_predictions[index] for seed_predictions in placebo_predictions) / len(SEEDS)
        for index in range(folds[0].test.height)
    ]
    placebo_summary = summarize_series(
        item["rank_ic"] for item in rank_ic_by_date(folds[0].test, placebo_mean_predictions)
    )
    placebo_result = {
        "fold": folds[0].fold,
        "label_scope": placebo_label_scope,
        "permutation_seed": PLACEBO_SEED,
        "rank_ic": placebo_summary,
        "pass": placebo_summary["mean"] is not None and abs(float(placebo_summary["mean"])) <= 0.02,
        "seed_count": len(SEEDS),
    }
    registry_rows.extend(placebo_rows)

    append_registry(_REGISTRY, registry_rows)
    config_counts: dict[str, int] = {}
    for report in fold_reports:
        key = _config_key(report["selected_params"])
        config_counts[key] = config_counts.get(key, 0) + 1
    config_values = {key: json.loads(key) for key in config_counts}
    selection_error: str | None = None
    try:
        model_config: dict[str, int] | None = json.loads(
            _select_most_wins_config(config_counts, config_values)
        )
    except ValueError as error:
        model_config = None
        selection_error = str(error)
    report = {
        "model_id": model_id,
        "data_snapshot_id": metadata["snapshot_id"],
        "market_data_feed": metadata["market_data_feed"],
        "feature_set": "px-v1",
        "label_horizon": horizon,
        "table_date_range": _table_date_range(frame),
        "holdout_start": metadata["holdout_start"],
        "universe_id": metadata.get("universe_id"),
        "universe_sha256": metadata.get("universe_sha256"),
        "survivorship_biased_universe": bool(metadata.get("survivorship_biased", True)),
        "trial_count": count_trials(_REGISTRY),
        "num_threads": num_threads,
        "seed_count": len(SEEDS),
        "configuration_count": len(CONFIGS),
        "selected_final_config": model_config,
        "final_config_selection_error": selection_error,
        "placebo": placebo_result,
        "folds": fold_reports,
    }
    _write_report(run_dir, report)
    if not placebo_result["pass"]:
        raise RuntimeError(
            "label-permutation placebo failed: mean test rank IC must be within ±0.02 of zero; "
            f"observed {placebo_summary['mean']}"
        )
    return report


def _fit_one(
    fold: WalkForwardFold,
    config: dict[str, int],
    seed: int,
    *,
    num_threads: int,
) -> tuple[lgb.Booster, int]:
    train = _valid_labeled(fold.train)
    valid = _valid_labeled(fold.valid)
    if train.is_empty() or valid.is_empty():
        raise ValueError("LightGBM requires labeled rows in train and validation partitions")
    train_set = lgb.Dataset(_features(train), label=train["label"].to_numpy(), feature_name=list(PX_V1_FEATURES))
    valid_set = lgb.Dataset(
        _features(valid), label=valid["label"].to_numpy(), feature_name=list(PX_V1_FEATURES), reference=train_set
    )
    model = lgb.train(
        _full_params(config, seed, num_threads),
        train_set,
        num_boost_round=2000,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model, int(model.best_iteration or 2000)


def _full_params(config: dict[str, int], seed: int, num_threads: int) -> dict[str, Any]:
    return {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": num_threads,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
        **config,
    }


def _valid_labeled(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(pl.col("label").is_not_null() & pl.col("raw_forward_return").is_not_null())


def _features(frame: pl.DataFrame):
    return frame.select(PX_V1_FEATURES).to_numpy()


def _config_key(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _select_most_wins_config(
    config_counts: dict[str, int], config_values: dict[str, dict[str, int]]
) -> str:
    """Apply the specified fold-count, tree-size, and leaf-size tie rules."""
    if not config_counts:
        raise ValueError("cannot select a final configuration without fold winners")
    most_wins = max(config_counts.values())
    candidates = [key for key, count in config_counts.items() if count == most_wins]
    smallest_tree = min(config_values[key]["num_leaves"] for key in candidates)
    candidates = [key for key in candidates if config_values[key]["num_leaves"] == smallest_tree]
    largest_leaf = max(config_values[key]["min_data_in_leaf"] for key in candidates)
    candidates = [key for key in candidates if config_values[key]["min_data_in_leaf"] == largest_leaf]
    if len(candidates) != 1:
        raise ValueError(
            "final configuration remains tied on lambda_l2 after the specified tie-break rules; "
            "resolve the selection rule in docs/training-v1-questions.md before fitting"
        )
    return candidates[0]


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_report(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # The JSON is the canonical artifact; render Markdown from its serialized form.
    report = json.loads(report_path.read_text(encoding="utf-8"))
    lines = [
        f"# Walk-forward report: {report['model_id']}",
        "",
        f"- Data snapshot: `{report['data_snapshot_id']}`",
        f"- Market data feed: `{report['market_data_feed']}`"
        + (
            " (single-venue development data; not for live trading decisions)"
            if report["market_data_feed"] == "iex"
            else ""
        ),
        f"- Feature set: `{report['feature_set']}`",
        f"- Label horizon: {report['label_horizon']} sessions",
        f"- Pre-holdout table: {report['table_date_range'][0]} to {report['table_date_range'][1]}",
        f"- Holdout starts: {report['holdout_start']}",
        f"- Universe: `{report['universe_id']}` (survivorship biased: "
        f"{report['survivorship_biased_universe']})",
        f"- Trials: {report['trial_count']}",
        f"- Fixed grid: {report['configuration_count']} configurations × {report['seed_count']} seeds per fold",
        "- Cost scenarios: 1× and 2× commission, spread, and impact",
        (
            f"- Selected final configuration: `{json.dumps(report['selected_final_config'], sort_keys=True)}`"
            if report["selected_final_config"] is not None
            else f"- Final configuration unresolved: `{report['final_config_selection_error']}`"
        ),
        f"- Placebo label scope: `{report['placebo']['label_scope']}`",
        f"- Placebo permutation seed: `{report['placebo']['permutation_seed']}`",
        f"- Placebo mean test rank IC: `{_fmt(report['placebo']['rank_ic']['mean'])}`; "
        f"t-stat: `{_fmt(report['placebo']['rank_ic']['t_stat'])}`; "
        f"status: **{ 'pass' if report['placebo']['pass'] else 'fail' }**",
        "",
        "## Annual test folds",
        "",
        "| Test year | Test range | Mean rank IC | IC standard deviation | t-stat | Positive IC fraction |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for fold in report["folds"]:
        summary = fold["rank_ic"]
        lines.append(
            f"| {fold['fold']} | {fold['test_range'][0]} to {fold['test_range'][1]} | "
            f"{_fmt(summary['mean'])} | {_fmt(summary['std'])} | {_fmt(summary['t_stat'])} | "
            f"{_fmt(summary['positive_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## Portfolio results",
            "",
            "| Fold | Costs | Strategy | Total return | Annualized return | Volatility | Sharpe | "
            "Max drawdown | Turnover | Costs paid |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for fold in report["folds"]:
        for multiple, scenario in fold["portfolio"].items():
            for strategy in ("lightgbm", "momentum_12_1", "equal_weight_buy_hold", "spy_buy_hold"):
                metrics = scenario[strategy]
                lines.append(
                    f"| {fold['fold']} | {multiple} | {strategy} | {_fmt(metrics['total_return'])} | "
                    f"{_fmt(metrics['annualized_return'])} | {_fmt(metrics['annualized_volatility'])} | "
                    f"{_fmt(metrics['sharpe'])} | {_fmt(metrics['max_drawdown'])} | "
                    f"{_fmt(metrics['turnover'])} | {_fmt(metrics['total_costs'])} |"
                )
    lines.extend(["", "## Monthly fold detail", ""])
    for fold in report["folds"]:
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
                f"| {month} | {_fmt(ic.get('lightgbm'))} | {_fmt(ic.get('momentum_12_1'))} | "
                f"{_fmt(pnl.get('lightgbm'))} | {_fmt(pnl.get('momentum_12_1'))} | "
                f"{_fmt(pnl.get('equal_weight_buy_hold'))} | {_fmt(pnl.get('spy_buy_hold'))} |"
            )
        lines.extend(["", "Top stock contributions at 1× costs:"])
        lines.extend(
            f"- {item['symbol']}: {_fmt(item['contribution'])}"
            for item in fold["portfolio"]["1x"]["concentration"]["top_five_stocks"]
        )
        lines.extend(["", "Top monthly contributions at 1× costs:"])
        lines.extend(
            f"- {item['month']}: {_fmt(item['contribution'])}"
            for item in fold["portfolio"]["1x"]["concentration"]["top_five_months"]
        )
    lines.extend(["", "This report uses only rows before the locked holdout. It contains no holdout metrics.", ""])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: int | float | None) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def _table_date_range(frame: pl.DataFrame) -> list[str]:
    dates = [value for value in frame["date"].to_list() if isinstance(value, date)]
    if len(dates) != frame.height:
        raise ValueError("training table contains missing or invalid decision dates")
    return [min(dates).isoformat(), max(dates).isoformat()]
