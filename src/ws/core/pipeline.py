"""Builder, DAG validation, async execution and deterministic log replay."""

from __future__ import annotations

import asyncio
import importlib
import json
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ws.core.clock import Clock, SimClock, Tick
from ws.core.event_log import EventLogger
from ws.core.messages import Inputs, Message, Missing, message_digest
from ws.core.node import AsyncExecutor, Executor, Node, NodeContext


class ReplayMismatch(AssertionError):
    """A replayed node emitted bytes different from the recorded run."""


class NodeSpec(BaseModel):
    """Configuration for constructing one pipeline node."""

    model_config = ConfigDict(frozen=True)

    cls: str
    kwargs: dict[str, Any] = Field(default_factory=dict)


class PipelineBuilder:
    def __init__(self) -> None:
        self._nodes: list[Node] = []

    def add(self, node: Node) -> PipelineBuilder:
        self._nodes.append(node)
        return self

    def build(
        self,
        *,
        inputs: Iterable[type[Message]] = (),
        executors: Mapping[str, Executor] | None = None,
    ) -> Pipeline:
        """Build the graph, treating declared input message types as coordinator supplied."""
        return Pipeline(self._nodes, inputs=inputs, executors=executors)


class Pipeline:
    def __init__(
        self,
        nodes: Sequence[Node],
        *,
        inputs: Iterable[type[Message]] = (),
        executors: Mapping[str, Executor] | None = None,
    ) -> None:
        self.nodes = tuple(nodes)
        self.external_inputs = frozenset(inputs)
        self.executors = {"async": AsyncExecutor(), **(executors or {})}
        self.config_specs: list[dict[str, Any]] | None = None
        self.levels = self._validate_and_order()

    @classmethod
    def builder(cls) -> PipelineBuilder:
        return PipelineBuilder()

    @classmethod
    def build(
        cls,
        *nodes: Node,
        inputs: Iterable[type[Message]] = (),
        executors: Mapping[str, Executor] | None = None,
    ) -> Pipeline:
        """Convenience for small graphs; prefer ``builder().add(...).build()``."""
        return cls(nodes, inputs=inputs, executors=executors)

    def _validate_and_order(self) -> tuple[tuple[Node, ...], ...]:
        if not self.nodes:
            raise ValueError("pipeline must contain at least one node")
        names = [node.name for node in self.nodes]
        if len(set(names)) != len(names):
            raise ValueError("node names must be unique")
        writers: dict[type[Message], Node] = {}
        for node in self.nodes:
            if node.writes in writers:
                raise ValueError(f"multiple nodes write {node.writes.__name__}")
            writers[node.writes] = node
            if node.policy.executor not in self.executors:
                raise ValueError(f"no executor registered for {node.policy.executor!r} ({node.name})")
        overlap = set(writers).intersection(self.external_inputs)
        if overlap:
            names = sorted(message_type.__name__ for message_type in overlap)
            raise ValueError(f"message types cannot be both externally supplied and node-written: {names}")
        for node in self.nodes:
            for message_type in node.reads:
                if message_type not in writers and message_type not in self.external_inputs:
                    raise ValueError(f"{node.name} reads unwritten message {message_type.__name__}")
                if message_type in self.external_inputs:
                    continue
                upstream = writers[message_type]
                if upstream is node:
                    raise ValueError(f"{node.name} reads its own output")
                if node.kind == "stateless" and upstream.kind == "stateful":
                    raise ValueError(f"stateless node {node.name} cannot read stateful output {message_type.__name__}")

        remaining = list(self.nodes)
        completed: set[Node] = set()
        levels: list[tuple[Node, ...]] = []
        while remaining:
            ready = tuple(
                node
                for node in remaining
                if all(
                    message_type in self.external_inputs or writers[message_type] in completed
                    for message_type in node.reads
                )
            )
            if not ready:
                raise ValueError("pipeline graph contains a cycle")
            levels.append(ready)
            completed.update(ready)
            remaining = [node for node in remaining if node not in completed]
        return tuple(levels)

    async def run(
        self,
        tick: Tick,
        *,
        run_id: str,
        log_path: Path,
        clock: Clock | None = None,
        tools: Any = None,
        inputs: Mapping[type[Message], Message] | None = None,
    ) -> dict[type[Message], Message | Missing]:
        logger = EventLogger(log_path)
        active_clock = clock or SimClock([tick])
        if active_clock.tick() != tick:
            raise ValueError("clock's current tick does not match requested pipeline tick")
        supplied = dict(inputs or {})
        if set(supplied) != set(self.external_inputs):
            missing = sorted(message_type.__name__ for message_type in self.external_inputs - set(supplied))
            unexpected = sorted(message_type.__name__ for message_type in set(supplied) - self.external_inputs)
            raise ValueError(
                f"external inputs do not match graph declarations: missing={missing}, unexpected={unexpected}"
            )
        for message_type, external_message in supplied.items():
            if not isinstance(external_message, message_type):
                raise TypeError(
                    f"external input for {message_type.__name__} has type {type(external_message).__name__}"
                )
            if external_message.tick != tick:
                raise ValueError(f"external input {message_type.__name__} is for a different tick")
        messages: dict[type[Message], Message | Missing] = dict(supplied)
        outputs: dict[type[Message], Message | Missing] = {}
        logger.emit("run_started", run_id=run_id, tick=tick.id, node_count=len(self.nodes), config=self.config_specs)
        logger.emit("tick_started", run_id=run_id, tick=tick.id)
        try:
            for level in self.levels:
                tasks: dict[Node, asyncio.Task[Message | Missing]] = {}
                async with asyncio.TaskGroup() as group:
                    for node in level:
                        tasks[node] = group.create_task(
                            self._run_node(node, active_clock, run_id, logger, tools, messages)
                        )
                results = [tasks[node].result() for node in level]
                for node, node_message in zip(level, results, strict=True):
                    messages[node.writes] = node_message
                    outputs[node.writes] = node_message
            logger.emit("tick_completed", run_id=run_id, tick=tick.id)
            logger.emit("run_completed", run_id=run_id, outputs=len(outputs))
            return outputs
        except Exception as exc:
            error_type, detail = _failure_info(exc)
            logger.emit("run_failed", run_id=run_id, tick=tick.id, error=error_type, detail=detail)
            raise

    async def run_many(
        self,
        ticks: Iterable[Tick],
        *,
        run_id: str,
        log_path: Path,
        tools: Any = None,
        inputs_for_tick: Callable[[Tick], Mapping[type[Message], Message]] | None = None,
    ) -> list[dict[type[Message], Message | Missing]]:
        ordered = tuple(ticks)
        clock = SimClock(ordered)
        outputs: list[dict[type[Message], Message | Missing]] = []
        for tick in clock:
            tick_inputs = inputs_for_tick(tick) if inputs_for_tick is not None else None
            outputs.append(
                await self.run(
                    tick,
                    run_id=run_id,
                    log_path=log_path,
                    clock=clock,
                    tools=tools,
                    inputs=tick_inputs,
                )
            )
        return outputs

    async def replay(
        self,
        ticks: Iterable[Tick],
        *,
        recorded_log: Path,
        recorded_run_id: str,
        replay_log: Path,
        tools: Any = None,
        inputs_for_tick: Callable[[Tick], Mapping[type[Message], Message]] | None = None,
    ) -> list[dict[type[Message], Message | Missing]]:
        """Rerun a deterministic graph and compare every output to recorded hashes."""
        expected = _recorded_outputs(recorded_log, recorded_run_id)
        ordered = tuple(ticks)
        if not ordered:
            raise ValueError("replay requires at least one tick")
        actual: dict[tuple[str, str], str] = {}
        outputs: list[dict[type[Message], Message | Missing]] = []
        replay_id = f"{recorded_run_id}-replay-{uuid.uuid4().hex[:8]}"
        clock = SimClock(ordered)
        for tick in clock:
            tick_inputs = inputs_for_tick(tick) if inputs_for_tick is not None else None
            result = await self.run(
                tick,
                run_id=replay_id,
                log_path=replay_log,
                clock=clock,
                tools=tools,
                inputs=tick_inputs,
            )
            outputs.append(result)
            for message in result.values():
                actual[(tick.id, message.producer)] = message_digest(message)
        if actual != expected:
            missing = sorted(set(expected) - set(actual))
            unexpected = sorted(set(actual) - set(expected))
            changed = sorted(key for key in set(actual) & set(expected) if actual[key] != expected[key])
            raise ReplayMismatch(
                f"replay outputs differ: missing={missing}, unexpected={unexpected}, changed={changed}"
            )
        return outputs

    async def _run_node(
        self,
        node: Node,
        clock: Clock,
        run_id: str,
        logger: EventLogger,
        tools: Any,
        messages: Mapping[type[Message], Message | Missing],
    ) -> Message | Missing:
        node_inputs = Inputs({message_type: messages[message_type] for message_type in node.reads})
        ctx = NodeContext(clock=clock, inputs=node_inputs, log=logger, tools=tools, run_id=run_id)
        for attempt in range(node.policy.retries + 1):
            started = time.perf_counter()
            logger.emit("node_started", run_id=run_id, tick=clock.tick().id, node=node.name, attempt=attempt + 1)
            try:
                executor = self.executors[node.policy.executor]
                output = await asyncio.wait_for(executor.execute(node, ctx), timeout=node.policy.timeout_s)
                if not isinstance(output, node.writes):
                    raise TypeError(f"{node.name} returned {type(output).__name__}, expected {node.writes.__name__}")
                if output.tick != clock.tick():
                    raise ValueError(f"{node.name} returned a message for a different tick")
                logger.emit(
                    "node_completed",
                    run_id=run_id,
                    tick=clock.tick().id,
                    node=node.name,
                    node_version=output.producer_version,
                    status="ok",
                    duration_ms=round((time.perf_counter() - started) * 1000, 3),
                    inputs={
                        message_type.__name__: _input_digest(messages[message_type]) for message_type in node.reads
                    },
                    output=message_digest(output),
                )
                return output
            except Exception as exc:
                logger.emit(
                    "node_failed" if attempt == node.policy.retries else "node_retried",
                    run_id=run_id,
                    tick=clock.tick().id,
                    node=node.name,
                    attempt=attempt + 1,
                    error=type(exc).__name__,
                    detail=str(exc),
                )
                if attempt < node.policy.retries:
                    await asyncio.sleep(node.policy.backoff_s * (2**attempt))
                    continue
                if not node.policy.required:
                    missing = Missing(
                        tick=clock.tick(),
                        producer=node.name,
                        expected_type=node.writes.__name__,
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                    logger.emit(
                        "node_completed",
                        run_id=run_id,
                        tick=clock.tick().id,
                        node=node.name,
                        status="missing",
                        output=message_digest(missing),
                    )
                    return missing
                raise
        raise AssertionError("retry loop exited unexpectedly")


def build_from_config(
    specs: Sequence[NodeSpec],
    *,
    resources: Mapping[str, Any],
    inputs: Iterable[type[Message]] = (),
) -> Pipeline:
    """Import configured node classes, resolve named resources, and validate their graph."""
    builder = Pipeline.builder()
    for spec in specs:
        module_name, separator, class_name = spec.cls.rpartition(".")
        if not separator:
            raise ValueError(f"node class path must be dotted: {spec.cls!r}")
        module = importlib.import_module(module_name)
        node_class = getattr(module, class_name, None)
        if not isinstance(node_class, type) or not issubclass(node_class, Node):
            raise TypeError(f"configured class {spec.cls!r} is not a Node subclass")
        kwargs: dict[str, Any] = {}
        for key, value in spec.kwargs.items():
            if isinstance(value, str) and value.startswith("@"):
                resource_name = value[1:]
                if resource_name not in resources:
                    raise ValueError(f"unknown resource @{resource_name} for {spec.cls}")
                kwargs[key] = resources[resource_name]
            else:
                kwargs[key] = value
        builder.add(node_class(**kwargs))
    pipeline = builder.build(inputs=inputs)
    pipeline.config_specs = [spec.model_dump(mode="json") for spec in specs]
    return pipeline


def _input_digest(message: Message | Missing) -> str:
    return message_digest(message)


def _failure_info(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, BaseExceptionGroup):
        leaves: list[BaseException] = []
        pending = list(exc.exceptions)
        while pending:
            error = pending.pop(0)
            if isinstance(error, BaseExceptionGroup):
                pending[0:0] = list(error.exceptions)
            else:
                leaves.append(error)
        detail = "; ".join(f"{type(error).__name__}: {error}" for error in leaves)
        return "ExceptionGroup", detail or str(exc)
    return type(exc).__name__, str(exc)


def _recorded_outputs(path: Path, run_id: str) -> dict[tuple[str, str], str]:
    outputs: dict[tuple[str, str], str] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("event") == "node_completed" and row.get("run_id") == run_id:
                outputs[(row["tick"], row["node"])] = row["output"]
    if not outputs:
        raise ValueError(f"no completed node outputs for run {run_id!r} in {path}")
    return outputs
