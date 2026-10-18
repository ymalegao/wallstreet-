"""Leakage-aware training and evaluation tools for the daily equity pipeline."""

from ws.training.table import LABEL_HORIZON, build_training_table

__all__ = ["LABEL_HORIZON", "build_training_table"]
