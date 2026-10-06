"""Date-partitioned daily bars for provider endpoints that return the whole market at once."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from pathlib import Path

import polars as pl

from ws.schema import Bar
from ws.store.atomic import atomic_path


class GroupedBarStore:
    def __init__(self, root: Path, name: str) -> None:
        self.dir = root / "grouped_bars" / name

    def write_session(self, session: date, bars: Iterable[Bar]) -> int:
        rows = [bar.model_dump() for bar in bars]
        if not rows:
            return 0
        frame = pl.DataFrame(rows).unique(subset=["symbol", "ts"]).sort("symbol")
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{session.isoformat()}.parquet"
        with atomic_path(path) as temporary:
            frame.write_parquet(temporary)
        return frame.height

    def read(self, symbols: list[str] | None = None) -> pl.DataFrame:
        files = sorted(self.dir.glob("*.parquet"))
        if not files:
            return pl.DataFrame()
        frames = [pl.read_parquet(path) for path in files]
        result = pl.concat(frames, how="diagonal_relaxed")
        if symbols is not None:
            result = result.filter(pl.col("symbol").is_in(symbols))
        return result.sort(["symbol", "ts"])
