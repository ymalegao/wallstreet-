"""Pre-holdout final model fitting and versioned model artifacts."""

from __future__ import annotations

import hashlib
import json
import statistics
from datetime import date
from pathlib import Path
from typing import Any, Literal

import lightgbm as lgb
import polars as pl

from ws.features.px_v1 import PX_V1_FEATURES
from ws.store.atomic import atomic_text
from ws.training.registry import append_registry, count_trials, read_registry, registry_timestamp
from ws.training.splits import WalkForwardFold
from ws.training.table import load_pre_holdout_table
from ws.training.trainer import (
    SEEDS,
    _features,
    _fit_one,
    _full_params,
    _git_commit,
    _select_most_wins_config,
    _valid_labeled,
)

_ROOT = Path(__file__).parents[3]
_REGISTRY = _ROOT / "models" / "registry.jsonl"
IterationPolicy = Literal["median-fold", "fresh-validation"]


def fit_final(
    table_path: Path,
    *,
    iteration_policy: IterationPolicy,
    num_threads: int = 4,
) -> dict[str, Any]:
    """Fit five seeded estimators on all pre-holdout data under an explicit stop policy."""
    if num_threads < 1:
        raise ValueError("num_threads must be positive")
    metadata_path = table_path.with_suffix(table_path.suffix + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    pre_holdout = load_pre_holdout_table(table_path, holdout_start=date.fromisoformat(metadata["holdout_start"]))
    frame = _valid_labeled(pre_holdout)
    if frame.is_empty():
        raise ValueError("the pre-holdout training table has no labeled rows")
    model_id = f"pxv1-{str(metadata['snapshot_id'])[:12]}"
    model_dir = _ROOT / "models" / model_id
    records = [item for item in read_registry(_REGISTRY) if item.get("model_id") == model_id]
    trials = [item for item in records if item.get("record_type") == "trial"]
    if not trials:
        raise ValueError(f"walk-forward trials are missing for {model_id}; run walk-forward first")
    if any(item.get("record_type") == "final_model" for item in records):
        raise FileExistsError(f"final model already exists for {model_id}")
    selected_by_fold: dict[str, list[dict[str, Any]]] = {}
    for item in trials:
        if item.get("selected"):
            selected_by_fold.setdefault(str(item["fold"]), []).append(item)
    if not selected_by_fold:
        raise ValueError("walk-forward registry contains no selected fold configurations")
    config_by_fold = {
        fold: selected[0]["params"] for fold, selected in selected_by_fold.items()
    }
    key_counts: dict[str, int] = {}
    config_values: dict[str, dict[str, int]] = {}
    for params in config_by_fold.values():
        config = {name: int(params[name]) for name in ("num_leaves", "min_data_in_leaf", "lambda_l2")}
        key = json.dumps(config, sort_keys=True, separators=(",", ":"))
        key_counts[key] = key_counts.get(key, 0) + 1
        config_values[key] = config
    selected_key = _select_most_wins_config(key_counts, config_values)
    selected_config = config_values[selected_key]
    if iteration_policy == "median-fold":
        selected_rows = [
            item
            for item in trials
            if item.get("selected")
            and {
                name: int(item["params"][name]) for name in ("num_leaves", "min_data_in_leaf", "lambda_l2")
            }
            == selected_config
        ]
        iteration_values = [int(item["best_iteration"]) for item in selected_rows]
        valid_range: list[str] = []
    elif iteration_policy == "fresh-validation":
        train, valid = _fresh_validation_split(frame, int(metadata["label_horizon"]))
        iteration_values = []
        for seed in SEEDS:
            fake_fold = WalkForwardFold(
                fold="final-validation",
                train=train,
                valid=valid,
                test=valid,
                train_range=(min(train["date"].to_list()), max(train["date"].to_list())),
                valid_range=(min(valid["date"].to_list()), max(valid["date"].to_list())),
                test_range=(min(valid["date"].to_list()), max(valid["date"].to_list())),
            )
            _, best_iteration = _fit_one(fake_fold, selected_config, seed, num_threads=num_threads)
            iteration_values.append(best_iteration)
        valid_range = [min(valid["date"].to_list()).isoformat(), max(valid["date"].to_list()).isoformat()]
    else:
        raise ValueError(f"unknown final iteration policy: {iteration_policy}")
    if not iteration_values:
        raise ValueError("no iteration counts available for the final refit")
    iterations = max(1, int(statistics.median(iteration_values) + 0.5))

    model_dir.mkdir(parents=True, exist_ok=True)
    feature_matrix = _features(frame)
    labels = frame["label"].to_numpy()
    for seed in SEEDS:
        # LightGBM binds data_random_seed when a Dataset handle is first constructed.
        # Reuse its immutable raw matrix but give each seeded fit an independent handle.
        train_set = lgb.Dataset(
            feature_matrix, label=labels, feature_name=list(PX_V1_FEATURES)
        )
        model = lgb.train(
            _full_params(selected_config, seed, num_threads),
            train_set,
            num_boost_round=iterations,
        )
        temporary_model = model_dir / f".seed_{seed}.pending.txt"
        model.save_model(str(temporary_model))
        temporary_model.replace(model_dir / f"seed_{seed}.txt")

    final_model_config: dict[str, Any] = {
        "feature_set": "px-v1",
        "market_data_feed": str(metadata["market_data_feed"]),
        "feature_names": list(PX_V1_FEATURES),
        "label": {"horizon": int(metadata["label_horizon"]), "type": "xs_rank_open_to_open"},
        "params": selected_config,
        "seeds": list(SEEDS),
        "num_boost_round": iterations,
        "num_threads": num_threads,
        "iteration_policy": iteration_policy,
    }
    model_hashes = {
        f"seed_{seed}.txt": hashlib.sha256((model_dir / f"seed_{seed}.txt").read_bytes()).hexdigest()
        for seed in SEEDS
    }
    train_range = [min(frame["date"].to_list()).isoformat(), max(frame["date"].to_list()).isoformat()]
    config_sha256 = _model_config_sha256(
        final_model_config,
        model_hashes,
        data_snapshot_id=str(metadata["snapshot_id"]),
        universe_sha256=str(metadata["universe_sha256"]),
        holdout_start=str(metadata["holdout_start"]),
        labeled_end=str(metadata["labeled_end"]),
        train_range=train_range,
    )
    manifest = {
        **final_model_config,
        "model_files_sha256": model_hashes,
        "model_id": model_id,
        "data_snapshot_id": metadata["snapshot_id"],
        "market_data_feed": metadata["market_data_feed"],
        "train_range": train_range,
        "created_at": registry_timestamp(),
        "config_sha256": config_sha256,
        "trial_count": count_trials(_REGISTRY) + len(SEEDS),
        "source_store_path": metadata["source_store_path"],
        "universe_path": metadata["universe_path"],
        "universe_sha256": metadata["universe_sha256"],
        "holdout_start": metadata["holdout_start"],
        "labeled_end": metadata["labeled_end"],
        "survivorship_biased": bool(metadata.get("survivorship_biased", True)),
    }
    atomic_text(model_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    git_commit = _git_commit()
    final_rows = []
    for seed in SEEDS:
        final_rows.append(
            {
                "record_type": "final_model",
                "model_id": model_id,
                "created_at": registry_timestamp(),
                "data_snapshot_id": metadata["snapshot_id"],
                "market_data_feed": metadata["market_data_feed"],
                "feature_set": "px-v1",
                "label": final_model_config["label"],
                "fold": "final",
                "train_range": train_range,
                "valid_range": valid_range,
                "test_range": [],
                "params": _full_params(selected_config, seed, num_threads),
                "seed": seed,
                "best_iteration": iterations,
                "valid_rank_ic": None,
                "test_rank_ic": None,
                "selected": True,
                "git_commit": git_commit,
                "config_sha256": config_sha256,
                "iteration_policy": iteration_policy,
            }
        )
    append_registry(_REGISTRY, final_rows)
    report = {
        "model_id": model_id,
        "data_snapshot_id": metadata["snapshot_id"],
        "market_data_feed": metadata["market_data_feed"],
        "feature_set": "px-v1",
        "selected_params": selected_config,
        "seeds": list(SEEDS),
        "num_boost_round": iterations,
        "iteration_policy": iteration_policy,
        "config_sha256": config_sha256,
        "trial_count": manifest["trial_count"],
    }
    (model_dir / "fit-final.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _model_config_sha256(
    model_config: dict[str, Any],
    model_file_hashes: dict[str, str],
    *,
    data_snapshot_id: str,
    universe_sha256: str,
    holdout_start: str,
    labeled_end: str,
    train_range: list[str],
) -> str:
    """Hash model settings, fitted artifacts, data identity, and evaluation boundaries."""
    payload = {
        **model_config,
        "data_snapshot_id": data_snapshot_id,
        "universe_sha256": universe_sha256,
        "holdout_start": holdout_start,
        "labeled_end": labeled_end,
        "train_range": train_range,
        "model_files_sha256": model_file_hashes,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _fresh_validation_split(frame: pl.DataFrame, horizon: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    latest = max(frame["date"].to_list())
    try:
        start = latest.replace(year=latest.year - 1)
    except ValueError:
        start = latest.replace(year=latest.year - 1, day=28)
    train = frame.filter((pl.col("date") < start) & (pl.col("label_exit_date") < start))
    valid = frame.filter((pl.col("date") >= start) & (pl.col("label_exit_date") < latest))
    if train.is_empty() or valid.is_empty():
        raise ValueError("fresh pre-holdout validation split is empty")
    return train, valid
