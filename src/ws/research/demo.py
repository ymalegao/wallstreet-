"""Offline example: point-in-time bars → features → model → run card.

The generated prices are synthetic. This example demonstrates mechanics and
does not establish a tradable signal or forecast market performance.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import uuid
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Protocol

import numpy as np
import polars as pl
from pydantic import ConfigDict
from sklearn.ensemble import HistGradientBoostingRegressor

from ws.core import Message, Node, NodeContext, Pipeline, Tick

FEATURES = ("ret_1", "ret_5", "vol_20")
AS_OF = datetime(2023, 6, 30, 23, 59, tzinfo=UTC)
TRAIN_END = datetime(2022, 12, 31, 23, 59, tzinfo=UTC)
EVALUATION_START = date(2023, 1, 3)


class BarSource(Protocol):
    def load_as_of(self, as_of: datetime) -> tuple[pl.DataFrame, int]: ...


class CsvBarSource:
    """CSV source whose rows are visible only after their availability time."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load_as_of(self, as_of: datetime) -> tuple[pl.DataFrame, int]:
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
        raw = pl.read_csv(self.path, try_parse_dates=False).with_columns(
            pl.col("session").str.to_date(format="%Y-%m-%d"),
            pl.col("available_at").str.to_datetime(time_zone="UTC"),
        )
        visible = raw.filter(pl.col("available_at") <= as_of).sort(["symbol", "session"])
        return visible, raw.height - visible.height


class BarsMessage(Message):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    frame: pl.DataFrame
    source_sha256: str
    visible_rows: int
    excluded_future_rows: int


class FeatureMessage(Message):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    frame: pl.DataFrame
    feature_columns: tuple[str, ...]


class ModelMessage(Message):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    model: HistGradientBoostingRegressor
    feature_columns: tuple[str, ...]
    train_rows: int
    train_end: datetime


class EvaluationMessage(Message):
    run_card_path: str
    metrics: dict[str, float | int]


class LoadBars(Node):
    name = "load_bars"
    writes = BarsMessage

    def __init__(self, source: BarSource, source_path: Path) -> None:
        super().__init__()
        self.source = source
        self.source_path = source_path

    async def run(self, ctx: NodeContext) -> BarsMessage:
        frame, excluded = self.source.load_as_of(ctx.tick.at)
        digest = hashlib.sha256(self.source_path.read_bytes()).hexdigest()
        return BarsMessage(
            tick=ctx.tick,
            producer=self.name,
            frame=frame,
            source_sha256=digest,
            visible_rows=frame.height,
            excluded_future_rows=excluded,
        )


class BuildFeatures(Node):
    name = "build_features"
    reads = (BarsMessage,)
    writes = FeatureMessage

    async def run(self, ctx: NodeContext) -> FeatureMessage:
        bars = ctx.inputs[BarsMessage]
        frame = bars.frame.with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1).alias("ret_1"),
            (pl.col("close") / pl.col("close").shift(5).over("symbol") - 1).alias("ret_5"),
            (pl.col("close").shift(-1).over("symbol") / pl.col("close") - 1).alias("target_return"),
            pl.col("available_at").shift(-1).over("symbol").alias("target_available_at"),
        )
        frame = frame.with_columns(
            pl.col("ret_1").rolling_std(window_size=20, min_samples=10).over("symbol").alias("vol_20")
        )
        # Labels are outcomes, so they are usable only after the next bar was available.
        frame = frame.filter(
            (pl.col("target_available_at") <= ctx.tick.at)
            & pl.col("ret_1").is_not_null()
            & pl.col("ret_5").is_not_null()
            & pl.col("vol_20").is_not_null()
            & pl.col("target_return").is_not_null()
        )
        return FeatureMessage(tick=ctx.tick, producer=self.name, frame=frame, feature_columns=FEATURES)


class TrainBaseline(Node):
    name = "train_baseline"
    reads = (FeatureMessage,)
    writes = ModelMessage

    def __init__(self, train_end: datetime) -> None:
        super().__init__()
        self.train_end = train_end

    async def run(self, ctx: NodeContext) -> ModelMessage:
        samples = ctx.inputs[FeatureMessage].frame
        # A sample's label must itself have been observable by the training cutoff.
        train = samples.filter(pl.col("target_available_at") <= self.train_end)
        if train.is_empty():
            raise ValueError("no training samples are available by train_end")
        x_train = train.select(FEATURES).to_numpy()
        y_train = train.get_column("target_return").to_numpy()
        model = HistGradientBoostingRegressor(
            max_iter=80,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=17,
        )
        await asyncio.to_thread(model.fit, x_train, y_train)
        return ModelMessage(
            tick=ctx.tick,
            producer=self.name,
            model=model,
            feature_columns=FEATURES,
            train_rows=train.height,
            train_end=self.train_end,
        )


class EvaluateBaseline(Node):
    name = "evaluate_baseline"
    reads = (BarsMessage, FeatureMessage, ModelMessage)
    writes = EvaluationMessage

    def __init__(self, *, run_card_path: Path) -> None:
        super().__init__()
        self.run_card_path = run_card_path

    async def run(self, ctx: NodeContext) -> EvaluationMessage:
        bars = ctx.inputs[BarsMessage]
        features = ctx.inputs[FeatureMessage].frame
        trained = ctx.inputs[ModelMessage]
        test = features.filter(
            (pl.col("session") >= EVALUATION_START)
            & (pl.col("target_available_at") <= ctx.tick.at)
        )
        if test.is_empty():
            raise ValueError("no evaluation samples are available")
        x_test = test.select(trained.feature_columns).to_numpy()
        actual = test.get_column("target_return").to_numpy()
        predicted = await asyncio.to_thread(trained.model.predict, x_test)
        rmse = float(math.sqrt(np.mean(np.square(predicted - actual))))
        zero_rmse = float(math.sqrt(np.mean(np.square(actual))))
        correlation = float(np.corrcoef(predicted, actual)[0, 1]) if np.std(predicted) and np.std(actual) else 0.0
        metrics: dict[str, float | int] = {
            "train_rows": trained.train_rows,
            "evaluation_rows": test.height,
            "model_rmse": rmse,
            "zero_return_rmse": zero_rmse,
            "pearson_correlation": correlation,
        }
        card = {
            "run_id": ctx.run_id,
            "data": {
                "kind": "deterministic synthetic daily bars",
                "sha256": bars.source_sha256,
                "as_of": ctx.tick.at.isoformat(),
                "visible_rows": bars.visible_rows,
                "excluded_future_rows": bars.excluded_future_rows,
            },
            "split": {
                "train_label_available_by": trained.train_end.isoformat(),
                "evaluation_decision_from": EVALUATION_START.isoformat(),
                "evaluation_label_available_by": ctx.tick.at.isoformat(),
            },
            "model": {
                "class": "sklearn.ensemble.HistGradientBoostingRegressor",
                "parameters": trained.model.get_params(),
                "features": list(trained.feature_columns),
                "random_state": 17,
            },
            "metrics": metrics,
            "interpretation": "Synthetic mechanics demo only; not evidence of tradable alpha.",
        }
        self.run_card_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_card_path.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return EvaluationMessage(
            tick=ctx.tick,
            producer=self.name,
            run_card_path=str(self.run_card_path),
            metrics=metrics,
        )


def _business_days(start: date, end: date) -> list[date]:
    days: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def generate_synthetic_bars(path: Path, *, seed: int = 13) -> None:
    """Create a reproducible panel with deliberately post-cutoff rows."""
    rng = random.Random(seed)
    rows: list[dict[str, str | float]] = []
    days = _business_days(date(2020, 1, 1), date(2023, 7, 14))
    for symbol_index, symbol in enumerate(("AAA", "BBB", "CCC", "DDD", "EEE")):
        close = 40.0 + symbol_index * 35
        prior_returns: list[float] = []
        for session in days:
            momentum = sum(prior_returns[-5:]) / min(5, len(prior_returns)) if prior_returns else 0.0
            daily_return = 0.0001 + 0.12 * momentum + rng.gauss(0.0, 0.012)
            close *= max(0.5, 1.0 + daily_return)
            prior_returns.append(daily_return)
            available = datetime(session.year, session.month, session.day, 21, 0, tzinfo=UTC)
            rows.append(
                {
                    "symbol": symbol,
                    "session": session.isoformat(),
                    "available_at": available.isoformat(),
                    "close": close,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(path)


async def run_demo(output_dir: Path) -> dict[str, object]:
    data_path = output_dir / "synthetic_bars.csv"
    generate_synthetic_bars(data_path)
    source = CsvBarSource(data_path)
    run_card = output_dir / "run_card.json"
    nodes = (
        LoadBars(source, data_path),
        BuildFeatures(),
        TrainBaseline(TRAIN_END),
        EvaluateBaseline(run_card_path=run_card),
    )
    pipeline = Pipeline.build(*nodes)
    run_id = f"synthetic-daily-v0-{uuid.uuid4().hex[:8]}"
    outputs = await pipeline.run(
        Tick.from_datetime(AS_OF), run_id=run_id, log_path=output_dir / f"{run_id}.events.jsonl"
    )
    return outputs[EvaluationMessage].model_dump(mode="json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/research_demo"))
    args = parser.parse_args()
    result = asyncio.run(run_demo(args.output_dir))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
