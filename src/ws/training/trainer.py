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
from ws.features.pxf_v1 import PXF_V1_FEATURES
from ws.training.evaluation import run_portfolio_scenarios
from ws.training.metrics import mean_daily_rank_ic, quintile_spread_by_date, rank_ic_by_date, summarize_series
from ws.training.placebo import permute_fold_labels
from ws.training.registry import append_registry, count_trials, registry_timestamp
from ws.training.splits import WalkForwardFold, build_folds
from ws.training.table import FEATURES_BY_SET, load_pre_holdout_table
from ws.fundamentals.validation import construction_check, load_monthly_factors

SEEDS = (0, 1, 2, 3, 4)
PLACEBO_SEED = 40_001
PLACEBO_RUNS = 50
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
    num_threads: int = 4,
) -> dict[str, Any]:
    """Run the fixed grid over expanding folds and register every config/seed/fold trial."""
    if num_threads < 1:
        raise ValueError("num_threads must be positive")
    metadata_path = table_path.with_suffix(table_path.suffix + ".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"training table metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_set = str(metadata.get("feature_set", ""))
    feature_names = FEATURES_BY_SET.get(feature_set)
    if feature_names is None or tuple(metadata.get("feature_names", ())) != feature_names:
        raise ValueError("training table feature metadata is incompatible with the supported feature sets")
    frame = load_pre_holdout_table(table_path, holdout_start=date.fromisoformat(metadata["holdout_start"]))
    horizon = int(metadata["label_horizon"])
    folds = build_folds(frame, horizon=horizon)
    if not folds:
        raise ValueError("no walk-forward fold has at least three years of prior training data")
    if feature_set == "px-v1":
        # f3: rerun under the fixed-iteration placebo protocol; pxv1f2-* predates it.
        model_id = f"pxv1f3-{str(metadata['snapshot_id'])[:12]}"
    else:
        fundamentals_id = str(metadata.get("fundamentals_snapshot_id", ""))
        if not fundamentals_id:
            raise ValueError("pxf-v1 table metadata is missing its fundamentals snapshot id")
        model_id = f"pxfv1-{str(metadata['snapshot_id'])[:10]}-{fundamentals_id[:10]}"
    run_dir = output_root / model_id
    model_dir = _ROOT / "models" / model_id
    if run_dir.exists() or model_dir.exists():
        raise FileExistsError(f"training output already exists for model id {model_id}")
    run_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    commit = _git_commit()
    registry_rows: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    test_frames: list[pl.DataFrame] = []
    real_test_daily_ics: list[float] = []
    placebo_daily_ics_by_run: dict[int, list[float]] = {index: [] for index in range(PLACEBO_RUNS)}

    for fold in folds:
        test_frames.append(fold.test)
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
                model, best_iteration = _fit_one(
                    fold, config, seed, num_threads=num_threads, feature_names=feature_names
                )
                valid_predictions = model.predict(_features(fold.valid, feature_names), num_iteration=best_iteration)
                test_predictions = model.predict(_features(fold.test, feature_names), num_iteration=best_iteration)
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
                    "fundamentals_snapshot_id": metadata.get("fundamentals_snapshot_id"),
                    "market_data_feed": metadata["market_data_feed"],
                    "feature_set": feature_set,
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
                {"feature_set": feature_set, "feature_names": list(feature_names), "seeds": list(SEEDS)},
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
        long_short = quintile_spread_by_date(fold.test, selected_predictions)
        real_test_daily_ics.extend(float(item["rank_ic"]) for item in model_ic)
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
            feature_set=feature_set,
            fundamentals_path=Path(metadata["fundamentals_path"]) if feature_set == "pxf-v1" else None,
        )

        placebo_run_means: list[float] = []
        selected_config = json.loads(selected_key)
        placebo_iterations = {
            int(entry["seed"]): int(entry["best_iteration"]) for entry in config_entries[selected_key]
        }
        for permutation_index in range(PLACEBO_RUNS):
            permutation_seed = PLACEBO_SEED + permutation_index
            placebo_fold = permute_fold_labels(fold, seed=permutation_seed, scope="train-only")
            seed_predictions: list[list[float]] = []
            for seed in SEEDS:
                model, best_iteration = _fit_one(
                    placebo_fold,
                    selected_config,
                    seed,
                    num_threads=num_threads,
                    feature_names=feature_names,
                    fixed_iteration=placebo_iterations[seed],
                )
                predictions = [
                    float(value)
                    for value in model.predict(
                        _features(placebo_fold.test, feature_names), num_iteration=best_iteration
                    )
                ]
                seed_predictions.append(predictions)
                seed_ic = summarize_series(
                    item["rank_ic"] for item in rank_ic_by_date(fold.test, predictions)
                )["mean"]
                registry_rows.append(
                    {
                        "record_type": "placebo_trial",
                        "model_id": model_id,
                        "created_at": registry_timestamp(),
                        "data_snapshot_id": metadata["snapshot_id"],
                        "fundamentals_snapshot_id": metadata.get("fundamentals_snapshot_id"),
                        "market_data_feed": metadata["market_data_feed"],
                        "feature_set": feature_set,
                        "label": {"horizon": horizon, "type": "xs_rank_open_to_open"},
                        "fold": fold.fold,
                        "train_range": [value.isoformat() for value in fold.train_range],
                        "valid_range": [value.isoformat() for value in fold.valid_range],
                        "test_range": [value.isoformat() for value in fold.test_range],
                        "params": {
                            **_full_params(selected_config, seed, num_threads),
                            "placebo_label_scope": "train-only",
                            "placebo_permutation_seed": permutation_seed,
                        },
                        "seed": seed,
                        "best_iteration": best_iteration,
                        "valid_rank_ic": None,
                        "test_rank_ic": seed_ic,
                        "selected": False,
                        "git_commit": commit,
                    }
                )
            ensemble_predictions = [
                sum(values[index] for values in seed_predictions) / len(SEEDS)
                for index in range(fold.test.height)
            ]
            daily_ics = [
                float(item["rank_ic"])
                for item in rank_ic_by_date(fold.test, ensemble_predictions)
            ]
            run_mean = summarize_series(daily_ics)["mean"]
            if run_mean is None:
                raise ValueError(f"placebo run {permutation_seed} has no finite test rank IC")
            placebo_run_means.append(float(run_mean))
            placebo_daily_ics_by_run[permutation_index].extend(daily_ics)

        fold_placebo_summary = summarize_series(placebo_run_means)
        fold_real_mean = summarize_series(item["rank_ic"] for item in model_ic)["mean"]
        fold_placebo_pass = (
            fold_placebo_summary["mean"] is not None
            and abs(float(fold_placebo_summary["mean"])) <= 0.02
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
                "long_short_quintile_spread": summarize_series(item["spread"] for item in long_short),
                "placebo": {
                    "permutation_count": PLACEBO_RUNS,
                    "runs": [
                        {
                            "permutation_seed": PLACEBO_SEED + index,
                            "mean_test_rank_ic": value,
                        }
                        for index, value in enumerate(placebo_run_means)
                    ],
                    "mean_test_rank_ic": fold_placebo_summary["mean"],
                    "real_model_mean_test_rank_ic": fold_real_mean,
                    "real_model_percentile": _empirical_percentile(fold_real_mean, placebo_run_means),
                    "pass": fold_placebo_pass,
                },
                "monthly_rank_ic": [
                    {
                        "date": month,
                        "lightgbm": model_monthly_ic.get(month),
                        "momentum_12_1": momentum_monthly_ic.get(month),
                    }
                    for month in sorted(set(model_monthly_ic) | set(momentum_monthly_ic))
                ],
                "portfolio": portfolio_results,
                "feature_importance_gain": _mean_gain_importance(
                    config_models[selected_key], feature_names
                ),
            }
        )
        registry_rows.extend(outcomes)

    placebo_run_means = [
        summarize_series(placebo_daily_ics_by_run[index])["mean"] for index in range(PLACEBO_RUNS)
    ]
    if any(value is None for value in placebo_run_means):
        raise ValueError("at least one aggregate placebo run has no finite test rank IC")
    placebo_run_means = [float(value) for value in placebo_run_means if value is not None]
    placebo_summary = summarize_series(placebo_run_means)
    real_model_mean = summarize_series(real_test_daily_ics)["mean"]
    all_folds_pass = all(bool(item["placebo"]["pass"]) for item in fold_reports)
    placebo_result = {
        "label_scope": "train-only",
        "permutation_count": PLACEBO_RUNS,
        "permutation_seeds": [PLACEBO_SEED + index for index in range(PLACEBO_RUNS)],
        "runs": [
            {
                "permutation_seed": PLACEBO_SEED + index,
                "mean_test_rank_ic": value,
            }
            for index, value in enumerate(placebo_run_means)
        ],
        "rank_ic": placebo_summary,
        "real_model_mean_test_rank_ic": real_model_mean,
        "real_model_percentile": _empirical_percentile(real_model_mean, placebo_run_means),
        "folds_pass": all_folds_pass,
        "pass": (
            placebo_summary["mean"] is not None
            and abs(float(placebo_summary["mean"])) <= 0.02
            and all_folds_pass
        ),
        "seed_count": len(SEEDS),
    }

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
        "feature_set": feature_set,
        "feature_names": list(feature_names),
        "fundamentals_snapshot_id": metadata.get("fundamentals_snapshot_id"),
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
    if feature_set == "pxf-v1":
        factor_path = Path(str(metadata.get("ken_french_factors_path", "")))
        momentum_path = Path(str(metadata.get("ken_french_momentum_path", "")))
        if not factor_path.is_file() or not momentum_path.is_file():
            raise FileNotFoundError(
                "pxf-v1 construction check requires the downloaded Ken French factor and momentum archives"
            )
        test_frame = pl.concat(test_frames, how="vertical_relaxed")
        report["construction_check"] = construction_check(
            test_frame,
            load_monthly_factors(factor_path, momentum_path),
        )
    _write_report(run_dir, report)
    if not placebo_result["pass"]:
        raise RuntimeError(
            "label-permutation placebo failed: aggregate and per-fold means must be within ±0.02 of zero; "
            f"observed {placebo_summary['mean']}"
        )
    return report


def _fit_one(
    fold: WalkForwardFold,
    config: dict[str, int],
    seed: int,
    *,
    num_threads: int,
    feature_names: tuple[str, ...] = PX_V1_FEATURES,
    fixed_iteration: int | None = None,
) -> tuple[lgb.Booster, int]:
    train = _valid_labeled(fold.train)
    if train.is_empty():
        raise ValueError("LightGBM requires labeled rows in the training partition")
    train_set = lgb.Dataset(
        _features(train, feature_names), label=train["label"].to_numpy(), feature_name=list(feature_names)
    )
    params = _full_params(config, seed, num_threads)
    if fixed_iteration is not None:
        if fixed_iteration < 1:
            raise ValueError("fixed_iteration must be positive")
        model = lgb.train(params, train_set, num_boost_round=fixed_iteration)
        return model, fixed_iteration

    valid = _valid_labeled(fold.valid)
    if valid.is_empty():
        raise ValueError("LightGBM requires labeled rows in the validation partition")
    valid_set = lgb.Dataset(
        _features(valid, feature_names),
        label=valid["label"].to_numpy(),
        feature_name=list(feature_names),
        reference=train_set,
    )

    validation_dates = valid.select("date", "label")

    def rank_ic_feval(predictions: Any, _dataset: lgb.Dataset) -> tuple[str, float, bool]:
        score = mean_daily_rank_ic(validation_dates, predictions)
        return "mean_daily_rank_ic", score if score is not None else 0.0, True

    model = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[valid_set],
        valid_names=["valid"],
        feval=rank_ic_feval,
        callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)],
    )
    return model, int(model.best_iteration or 2000)


def _full_params(config: dict[str, int], seed: int, num_threads: int) -> dict[str, Any]:
    return {
        "objective": "regression",
        "metric": "None",
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


def _features(frame: pl.DataFrame, feature_names: tuple[str, ...] = PX_V1_FEATURES):
    return frame.select(feature_names).to_numpy()


def _mean_gain_importance(models: list[lgb.Booster], feature_names: tuple[str, ...]) -> list[dict[str, float | str]]:
    if not models:
        return []
    importances = [model.feature_importance(importance_type="gain") for model in models]
    averaged = [sum(float(values[index]) for values in importances) / len(importances) for index in range(len(feature_names))]
    return [
        {"feature": name, "mean_gain": value}
        for name, value in sorted(zip(feature_names, averaged, strict=True), key=lambda item: (-item[1], item[0]))
    ]


def _empirical_percentile(value: float | None, distribution: list[float]) -> float | None:
    """Return the inclusive empirical percentile of a value in a finite sample."""
    if value is None or not distribution:
        return None
    return 100.0 * sum(sample <= value for sample in distribution) / len(distribution)


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
        *(
            [f"- Fundamentals snapshot: `{report['fundamentals_snapshot_id']}`"]
            if report.get("fundamentals_snapshot_id")
            else []
        ),
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
        f"- Placebo permutations: `{report['placebo']['permutation_count']}` train-only permutations "
        f"across {report['placebo']['seed_count']} model seeds per fold",
        f"- Placebo mean of run-level test rank ICs: `{_fmt(report['placebo']['rank_ic']['mean'])}`; "
        f"t-stat: `{_fmt(report['placebo']['rank_ic']['t_stat'])}`",
        f"- Real model mean test rank IC: `{_fmt(report['placebo']['real_model_mean_test_rank_ic'])}`; "
        f"percentile in placebo distribution: `{_fmt(report['placebo']['real_model_percentile'])}`; "
        f"fold checks: **{ 'pass' if report['placebo']['folds_pass'] else 'fail' }**; "
        f"status: **{ 'pass' if report['placebo']['pass'] else 'fail' }**",
        "",
        "## Annual test folds",
        "",
        "| Test year | Test range | Mean rank IC | IC standard deviation | t-stat | Positive IC fraction | "
        "Long-short mean | Long-short t-stat | "
        "Placebo mean | Real percentile | Placebo pass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for fold in report["folds"]:
        summary = fold["rank_ic"]
        lines.append(
            f"| {fold['fold']} | {fold['test_range'][0]} to {fold['test_range'][1]} | "
            f"{_fmt(summary['mean'])} | {_fmt(summary['std'])} | {_fmt(summary['t_stat'])} | "
            f"{_fmt(summary['positive_fraction'])} | "
            f"{_fmt(fold['long_short_quintile_spread']['mean'])} | "
            f"{_fmt(fold['long_short_quintile_spread']['t_stat'])} | "
            f"{_fmt(fold['placebo']['mean_test_rank_ic'])} | "
            f"{_fmt(fold['placebo']['real_model_percentile'])} | "
            f"{ 'pass' if fold['placebo']['pass'] else 'fail' } |"
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
    lines.extend(
        [
            "",
            "## Turnover with and without the holding buffer",
            "",
            "| Fold | Costs | Strategy | With buffer | Without buffer |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for fold in report["folds"]:
        for multiple, scenario in fold["portfolio"].items():
            for strategy in ("lightgbm", "momentum_12_1"):
                comparison = scenario["turnover_buffer_comparison"][strategy]
                lines.append(
                    f"| {fold['fold']} | {multiple} | {strategy} | "
                    f"{_fmt(comparison['with_buffer'])} | {_fmt(comparison['without_buffer'])} |"
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
        lines.extend(
            [
                "",
                "Mean LightGBM gain importance across fold seeds:",
                "",
                "| Feature | Mean gain |",
                "|---|---:|",
            ]
        )
        lines.extend(
            f"| {item['feature']} | {_fmt(item['mean_gain'])} |"
            for item in fold.get("feature_importance_gain", [])
        )
    if report.get("construction_check"):
        lines.extend(
            [
                "",
                "## Characteristic construction check",
                "",
                "Monthly portfolio returns use only walk-forward test dates; correlations are diagnostic, not acceptance gates.",
                "",
                "| Characteristic spread | FF factor | Months | Pearson correlation |",
                "|---|---|---:|---:|",
            ]
        )
        for name, result in report["construction_check"]["correlations"].items():
            lines.append(
                f"| {name} | {result['factor']} | {result['month_count']} | "
                f"{_fmt(result['pearson_correlation'])} |"
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
