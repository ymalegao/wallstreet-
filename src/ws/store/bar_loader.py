"""Load ordinary per-symbol BarStores or date-partitioned grouped daily stores."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ws.store.event_store import BarStore
from ws.store.grouped_bar_store import GroupedBarStore


def read_bar_store(root: Path, name: str, symbols: list[str] | None = None) -> pl.DataFrame:
    if name.startswith("grouped:"):
        return GroupedBarStore(root, name.removeprefix("grouped:")).read(symbols)
    return BarStore(root, name).read(symbols)
