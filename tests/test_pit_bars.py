from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import polars as pl
from pydantic import ConfigDict

from ws.core import Message, Node, NodeContext, Pipeline
from ws.core.clock import SimClock, Tick
from ws.core.messages import message_bytes
from ws.schema import Bar
from ws.store.pit_bars import PITBarStore, daily_bar_to_pit


def _bar(day: int, close: float) -> Bar:
    return Bar(
        symbol="AAPL",
        ts=datetime(2026, 10, day, 4, tzinfo=UTC),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=100.0,
    )


def test_daily_bar_availability_is_after_exchange_close() -> None:
    bar = daily_bar_to_pit(_bar(16, 100.0))
    assert bar.event_time == datetime(2026, 10, 16, 20, tzinfo=UTC)
    assert bar.available_at == bar.event_time + timedelta(minutes=15)


def test_read_filters_by_clock_and_future_shuffle_does_not_change_outputs(tmp_path) -> None:
    visible = daily_bar_to_pit(_bar(15, 100.0))
    future_a = daily_bar_to_pit(_bar(16, 101.0))
    future_b = daily_bar_to_pit(_bar(16, 999.0))
    tick = Tick(id="before-close-data", at=datetime(2026, 10, 16, 20, 14, tzinfo=UTC), kind="close")
    clock = SimClock([tick])

    store_a = PITBarStore(tmp_path / "a")
    store_b = PITBarStore(tmp_path / "b")
    store_a.write([visible, future_a])
    store_b.write([visible, future_b])

    result_a = store_a.read(clock, symbols=["AAPL"])
    result_b = store_b.read(clock, symbols=["AAPL"])
    assert result_a.equals(result_b)
    assert result_a.get_column("close").to_list() == [100.0]

    class BarsAtTick(Message):
        model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
        frame: pl.DataFrame

    class ReadPITBars(Node):
        name = "read_pit_bars"
        writes = BarsAtTick

        def __init__(self, store: PITBarStore) -> None:
            super().__init__()
            self.store = store

        async def run(self, ctx: NodeContext) -> BarsAtTick:
            return BarsAtTick(tick=ctx.tick, producer=self.name, frame=self.store.read(ctx.clock))

    async def node_output(store: PITBarStore, log_name: str) -> BarsAtTick:
        pipeline = Pipeline.build(ReadPITBars(store))
        output = await pipeline.run(tick, run_id=log_name, log_path=tmp_path / f"{log_name}.jsonl")
        return output[BarsAtTick]

    node_a, node_b = asyncio.run(node_output(store_a, "future-a")), asyncio.run(node_output(store_b, "future-b"))
    assert message_bytes(node_a) == message_bytes(node_b)


def test_same_bar_write_is_idempotent(tmp_path) -> None:
    store = PITBarStore(tmp_path / "bars")
    bar = daily_bar_to_pit(_bar(16, 100.0))
    assert store.write([bar]) == 1
    assert store.write([bar]) == 0

