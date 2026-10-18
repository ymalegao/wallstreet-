"""Date-wise label permutations for the required out-of-sample placebo."""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Literal

import polars as pl

from ws.training.splits import WalkForwardFold

PlaceboLabelScope = Literal["train-only", "all-splits"]


def permute_labels_by_date(frame: pl.DataFrame, *, seed: int) -> pl.DataFrame:
    """Return a copy whose labels are shuffled independently within each date."""
    if "date" not in frame.columns or "label" not in frame.columns:
        raise ValueError("placebo frame must contain date and label columns")
    labels = frame["label"].to_list()
    groups: dict[object, list[int]] = {}
    for index, session_date in enumerate(frame["date"].to_list()):
        groups.setdefault(session_date, []).append(index)
    randomizer = random.Random(seed)
    for indices in groups.values():
        values = [labels[index] for index in indices]
        randomizer.shuffle(values)
        for index, value in zip(indices, values, strict=True):
            labels[index] = value
    return frame.with_columns(pl.Series("label", labels, dtype=frame.schema["label"]))


def permute_fold_labels(fold: WalkForwardFold, *, seed: int, scope: PlaceboLabelScope) -> WalkForwardFold:
    """Create a fold with labels shuffled in training only or in each split."""
    if scope == "train-only":
        train = permute_labels_by_date(fold.train, seed=seed)
        return replace(fold, train=train)
    if scope == "all-splits":
        return replace(
            fold,
            train=permute_labels_by_date(fold.train, seed=seed),
            valid=permute_labels_by_date(fold.valid, seed=seed + 1),
            test=permute_labels_by_date(fold.test, seed=seed + 2),
        )
    raise ValueError(f"unknown placebo label scope: {scope}")
