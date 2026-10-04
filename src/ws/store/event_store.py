"""Append-only Parquet stores for events and bars, queried through DuckDB.

Rules that keep backtests honest:
  * records are never rewritten or deleted; a re-fetched event with a known ``event_id`` is dropped
    (first write wins), so later vendor revisions cannot silently replace what we saw;
  * every event row carries ``ingested_at``; the data-quality report fails if any
    ``first_seen_ts`` is later than ``ingested_at``.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from pathlib import Path

import duckdb
import polars as pl

from ws.schema import Bar, Event

EVENT_SCHEMA: dict[str, pl.DataType] = {
    "event_id": pl.String(),
    "source": pl.String(),
    "source_id": pl.String(),
    "first_seen_ts": pl.Datetime("us", "UTC"),
    "ts_origin": pl.String(),
    "published_ts": pl.Datetime("us", "UTC"),
    "updated_ts": pl.Datetime("us", "UTC"),
    "ingested_at": pl.Datetime("us", "UTC"),
    "tickers": pl.List(pl.Utf8),
    "headline": pl.String(),
    "body": pl.String(),
    "url": pl.String(),
    "kind": pl.String(),
    "meta": pl.String(),  # JSON object
}


def events_to_frame(events: Iterable[Event]) -> pl.DataFrame:
    rows = [
        {**e.model_dump(exclude={"meta"}), "ts_origin": e.ts_origin.value, "meta": json.dumps(e.meta)} for e in events
    ]
    return pl.DataFrame(rows, schema=EVENT_SCHEMA)


class EventStore:
    def __init__(self, root: Path) -> None:
        self.dir = root / "events"
        self._ids: set[str] | None = None

    @property
    def glob(self) -> str:
        return str(self.dir / "**" / "*.parquet")

    def _known_ids(self) -> set[str]:
        if self._ids is None:
            self._ids = set()
            if any(self.dir.rglob("*.parquet")):
                self._ids = set(
                    duckdb.sql(f"SELECT event_id FROM read_parquet('{self.glob}')").pl()["event_id"].to_list()
                )
        return self._ids

    def write(self, events: Iterable[Event]) -> int:
        """Persist events not seen before. Returns the number of new rows written."""
        known = self._known_ids()
        fresh: dict[str, Event] = {}
        for e in events:
            if e.event_id not in known and e.event_id not in fresh:
                fresh[e.event_id] = e
        if not fresh:
            return 0
        df = events_to_frame(fresh.values()).with_columns(pl.col("first_seen_ts").dt.strftime("%Y-%m").alias("month"))
        for (source, month), part in df.group_by(["source", "month"]):
            d = self.dir / f"source={source}" / f"month={month}"
            d.mkdir(parents=True, exist_ok=True)
            part.drop("month").write_parquet(d / f"part-{uuid.uuid4().hex}.parquet")
        known.update(fresh)
        return len(fresh)

    def read(self, where: str = "TRUE") -> pl.DataFrame:
        if not any(self.dir.rglob("*.parquet")):
            return pl.DataFrame(schema=EVENT_SCHEMA)
        df = duckdb.sql(
            f"SELECT * EXCLUDE (source, month), source FROM read_parquet('{self.glob}', hive_partitioning=true) "
            f"WHERE {where} ORDER BY first_seen_ts"
        ).pl()
        # DuckDB tags timestamps 'Etc/UTC'; restore the canonical schema so comparisons with UTC values work.
        return df.select([pl.col(c).cast(t) for c, t in EVENT_SCHEMA.items()])


class BarStore:
    def __init__(self, root: Path, timeframe: str) -> None:
        self.dir = root / "bars" / timeframe

    def write(self, bars: Iterable[Bar]) -> int:
        df = pl.DataFrame([b.model_dump() for b in bars])
        if df.is_empty():
            return 0
        n = 0
        for (symbol,), part in df.group_by(["symbol"]):
            path = self.dir / f"{symbol}.parquet"
            if path.exists():
                part = pl.concat([pl.read_parquet(path), part], how="diagonal_relaxed")
            part = part.unique(subset=["symbol", "ts"], keep="first").sort("ts")
            self.dir.mkdir(parents=True, exist_ok=True)
            part.write_parquet(path)
            n += part.height
        return n

    def read(self, symbols: list[str] | None = None) -> pl.DataFrame:
        files = [self.dir / f"{s}.parquet" for s in symbols] if symbols else sorted(self.dir.glob("*.parquet"))
        files = [f for f in files if f.exists()]
        return pl.concat([pl.read_parquet(f) for f in files]) if files else pl.DataFrame()
