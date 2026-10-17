from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ws.core import Message, Missing, Node, NodeContext, Pipeline, Policy, SimClock, Tick
from ws.core.messages import message_bytes


class Constant(Message):
    value: int


class Doubled(Message):
    value: int


class SinkValue(Message):
    value: int


class ConstantNode(Node):
    name = "constant"
    writes = Constant

    async def run(self, ctx: NodeContext) -> Constant:
        return Constant(tick=ctx.tick, producer=self.name, value=21)


class DoubleNode(Node):
    name = "double"
    reads = (Constant,)
    writes = Doubled

    async def run(self, ctx: NodeContext) -> Doubled:
        source = ctx.inputs[Constant]
        return Doubled(tick=ctx.tick, producer=self.name, value=source.value * 2)


class SinkNode(Node):
    name = "sink"
    reads = (Doubled,)
    writes = SinkValue

    async def run(self, ctx: NodeContext) -> SinkValue:
        source = ctx.inputs[Doubled]
        return SinkValue(tick=ctx.tick, producer=self.name, value=source.value)


def toy_pipeline() -> Pipeline:
    return Pipeline.builder().add(SinkNode()).add(DoubleNode()).add(ConstantNode()).build()


def ten_ticks() -> list[Tick]:
    start = datetime(2026, 1, 5, 21, 0, tzinfo=UTC)
    return [Tick.from_datetime(start + timedelta(days=day), "close") for day in range(10)]


def test_toy_pipeline_runs_ten_ticks_and_replays_identically(tmp_path: Path) -> None:
    ticks = ten_ticks()
    pipeline = toy_pipeline()
    assert [node.name for node in pipeline.levels[0]] == ["constant"]
    recorded_log = tmp_path / "toy.jsonl"
    replay_log = tmp_path / "replay.jsonl"

    first = asyncio.run(pipeline.run_many(ticks, run_id="toy-run", log_path=recorded_log))
    replayed = asyncio.run(
        pipeline.replay(
            ticks,
            recorded_log=recorded_log,
            recorded_run_id="toy-run",
            replay_log=replay_log,
        )
    )

    assert len(first) == len(replayed) == 10
    assert [message_bytes(run[SinkValue]) for run in first] == [message_bytes(run[SinkValue]) for run in replayed]
    assert all(run[SinkValue].value == 42 for run in replayed)
    records = [line for line in recorded_log.read_text(encoding="utf-8").splitlines() if line]
    assert any('"event":"run_started"' in line for line in records)
    assert sum('"event":"node_completed"' in line for line in records) == 30


def test_sim_clock_is_monotonic_and_exposes_current_tick() -> None:
    clock = SimClock(ten_ticks()[:2])
    assert clock.now() == ten_ticks()[0].at
    clock.advance()
    assert clock.tick() == ten_ticks()[1]
    with pytest.raises(StopIteration):
        clock.advance()


def test_pipeline_builder_rejects_unwritten_inputs_and_duplicate_writers() -> None:
    class ConstantNodeTwo(ConstantNode):
        name = "constant_two"

    class UnwrittenNode(Node):
        name = "unwritten"
        reads = (Constant,)
        writes = SinkValue

        async def run(self, ctx: NodeContext) -> SinkValue:
            return SinkValue(tick=ctx.tick, producer=self.name, value=0)

    with pytest.raises(ValueError, match="multiple nodes write Constant"):
        Pipeline.builder().add(ConstantNode()).add(ConstantNodeTwo()).build()
    with pytest.raises(ValueError, match="reads unwritten message Constant"):
        Pipeline.builder().add(UnwrittenNode()).build()


def test_policy_has_fail_closed_defaults() -> None:
    policy = Policy()
    assert policy.required is True
    assert policy.retries == 0
    assert policy.executor == "async"


def test_required_node_failure_cancels_other_nodes_at_same_level(tmp_path: Path) -> None:
    cancelled = asyncio.Event()

    class SlowOutput(Message):
        value: int

    class BrokenNode(ConstantNode):
        name = "broken"

        async def run(self, ctx: NodeContext) -> Constant:
            raise RuntimeError("required input failed")

    class SlowNode(ConstantNode):
        name = "slow"
        writes = SlowOutput

        async def run(self, ctx: NodeContext) -> SlowOutput:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return SlowOutput(tick=ctx.tick, producer=self.name, value=1)

    pipeline = Pipeline.builder().add(BrokenNode()).add(SlowNode()).build()

    async def run() -> None:
        with pytest.raises(ExceptionGroup) as error:
            await pipeline.run(ten_ticks()[0], run_id="fail-closed", log_path=tmp_path / "failed.jsonl")
        assert "required input failed" in repr(error.value.exceptions)

    asyncio.run(run())
    assert cancelled.is_set()
    assert '"detail":"RuntimeError: required input failed"' in (tmp_path / "failed.jsonl").read_text()


def test_optional_node_failure_is_explicit_and_downstream_can_degrade(tmp_path: Path) -> None:
    class OptionalValue(Message):
        value: int

    class DegradedValue(Message):
        value: int

    class OptionalFailure(Node):
        name = "optional_failure"
        writes = OptionalValue

        async def run(self, ctx: NodeContext) -> OptionalValue:
            raise RuntimeError("optional source unavailable")

    class Degrade(Node):
        name = "degrade"
        reads = (OptionalValue,)
        writes = DegradedValue

        async def run(self, ctx: NodeContext) -> DegradedValue:
            missing = ctx.inputs.get(OptionalValue)
            return DegradedValue(tick=ctx.tick, producer=self.name, value=int(isinstance(missing, Missing)))

    pipeline = Pipeline.builder().add(OptionalFailure(policy=Policy(required=False))).add(Degrade()).build()
    result = asyncio.run(pipeline.run(ten_ticks()[0], run_id="optional", log_path=tmp_path / "optional.jsonl"))
    assert isinstance(result[OptionalValue], Missing)
    assert result[DegradedValue].value == 1


def test_pipeline_builder_rejects_cycles_and_stateful_to_stateless_edges() -> None:
    class Left(Message):
        value: int

    class Right(Message):
        value: int

    class LeftNode(Node):
        name = "left"
        reads = (Right,)
        writes = Left

        async def run(self, ctx: NodeContext) -> Left:
            return Left(tick=ctx.tick, producer=self.name, value=1)

    class RightNode(Node):
        name = "right"
        reads = (Left,)
        writes = Right

        async def run(self, ctx: NodeContext) -> Right:
            return Right(tick=ctx.tick, producer=self.name, value=1)

    class StatefulNode(ConstantNode):
        name = "stateful"
        kind = "stateful"

    class StatelessReader(DoubleNode):
        name = "stateless_reader"

    with pytest.raises(ValueError, match="contains a cycle"):
        Pipeline.builder().add(LeftNode()).add(RightNode()).build()
    with pytest.raises(ValueError, match="cannot read stateful output"):
        Pipeline.builder().add(StatefulNode()).add(StatelessReader()).build()
