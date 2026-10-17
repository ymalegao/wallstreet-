"""Node contract and per-execution context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from ws.core.clock import Clock, Tick
from ws.core.event_log import EventLogger
from ws.core.messages import Inputs, Message


class Policy(BaseModel):
    model_config = ConfigDict(frozen=True)

    timeout_s: float = Field(default=300.0, gt=0)
    retries: int = Field(default=0, ge=0)
    backoff_s: float = Field(default=1.0, ge=0)
    required: bool = True
    executor: Literal["async", "process", "gpu"] = "async"


class ToolRegistry(Protocol):
    """Registry of read-only tools available to analyst nodes."""

    def schemas(self) -> list[dict[str, Any]]: ...

    def call(self, name: str, args: BaseModel) -> BaseModel: ...


@dataclass(frozen=True)
class NodeContext:
    clock: Clock
    inputs: Inputs
    log: EventLogger
    tools: ToolRegistry | None = None
    run_id: str = "run"

    @property
    def tick(self) -> Tick:
        return self.clock.tick()


class Node(ABC):
    """Subclass this contract or compose instances in a PipelineBuilder."""

    name: ClassVar[str] = "Node"
    reads: ClassVar[tuple[type[Message], ...]] = ()
    writes: ClassVar[type[Message]] = Message
    kind: ClassVar[Literal["stateless", "stateful"]] = "stateless"

    def __init__(self, *, policy: Policy | None = None) -> None:
        self.policy = policy or Policy()

    @abstractmethod
    async def run(self, ctx: NodeContext) -> Message:
        """Compute one output using only declared messages in ``ctx.inputs``."""


class Executor(Protocol):
    async def execute(self, node: Node, ctx: NodeContext) -> Message: ...


class AsyncExecutor:
    async def execute(self, node: Node, ctx: NodeContext) -> Message:
        return await node.run(ctx)


