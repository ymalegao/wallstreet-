"""Small auditable metrics without a training dependency."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np


def classification(gold: Sequence[str], predicted: Sequence[str], labels: Sequence[str]) -> dict[str, Any]:
    if len(gold) != len(predicted) or not gold:
        raise ValueError("Nonempty aligned labels required")
    matrix = [[sum(a == x and b == y for a, b in zip(gold, predicted, strict=True)) for y in labels] for x in labels]
    per_class = {}
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        precision = tp / max(1, sum(row[i] for row in matrix))
        recall = tp / max(1, sum(x == label for x in gold))
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "support": sum(x == label for x in gold),
        }
    return {
        "n": len(gold),
        "accuracy": sum(a == b for a, b in zip(gold, predicted, strict=True)) / len(gold),
        "macro_f1": float(np.mean([x["f1"] for x in per_class.values()])),
        "majority_class_accuracy": max(Counter(gold).values()) / len(gold),
        "labels": list(labels),
        "confusion_matrix": matrix,
        "per_class": per_class,
    }
