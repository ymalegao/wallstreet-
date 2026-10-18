"""Append-only experiment and locked-holdout registry."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ws.store.atomic import file_lock


def read_registry(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid registry JSON at {path}:{line_number}") from error
        if not isinstance(item, dict):
            raise ValueError(f"registry record at {path}:{line_number} must be an object")
        records.append(item)
    return records


def append_registry(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path.with_suffix(path.suffix + ".lock")), path.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()


def registry_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def count_trials(path: Path) -> int:
    """Count all configuration, placebo, and final-model seed fits in the registry."""
    trial_types = {"trial", "placebo_trial", "final_model"}
    return sum(record.get("record_type") in trial_types for record in read_registry(path))


def reserve_holdout(path: Path, model_id: str, config_sha256: str) -> dict[str, Any]:
    """Append the irreversible one-time evaluation reservation before holdout access."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path.with_suffix(path.suffix + ".lock")):
        records = read_registry(path)
        model_records = [item for item in records if item.get("model_id") == model_id]
        if not model_records:
            raise ValueError(f"model {model_id!r} has no registry record")
        final_models = [item for item in model_records if item.get("record_type") == "final_model"]
        if len(final_models) != 5 or {item.get("seed") for item in final_models} != {0, 1, 2, 3, 4}:
            raise ValueError(f"model {model_id!r} does not have all five registered final seed models")
        manifest_hashes = {
            item.get("config_sha256")
            for item in final_models
        }
        if manifest_hashes != {config_sha256}:
            raise ValueError("model config differs from the registered final model")
        if any(item.get("record_type") in {"holdout_started", "holdout_result"} for item in model_records):
            raise ValueError(f"holdout evaluation has already been reserved or run for {model_id}")
        record = {
            "record_type": "holdout_started",
            "model_id": model_id,
            "config_sha256": config_sha256,
            "created_at": registry_timestamp(),
        }
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
        return record
