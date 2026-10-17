"""Immutable message types exchanged between pipeline nodes."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any, TypeVar, cast

from pydantic import BaseModel, ConfigDict

from ws.core.clock import Tick


class Message(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    tick: Tick
    producer: str
    producer_version: str = "0.1.0"


class Missing(BaseModel):
    """Optional-node result passed downstream instead of silently omitting input."""

    model_config = ConfigDict(frozen=True)

    tick: Tick
    producer: str
    expected_type: str
    reason: str


M = TypeVar("M", bound=Message)


class Inputs:
    """Typed view of upstream messages, including explicit optional failures."""

    def __init__(self, messages: Mapping[type[Message], Message | Missing]) -> None:
        self._messages = messages

    def __getitem__(self, message_type: type[M]) -> M:
        message = self._messages[message_type]
        if isinstance(message, Missing):
            raise RuntimeError(f"upstream {message.producer} did not produce {message.expected_type}: {message.reason}")
        if not isinstance(message, message_type):
            raise TypeError(f"stored message is not a {message_type.__name__}")
        return cast(M, message)

    def get(self, message_type: type[M]) -> M | Missing:
        message = self._messages[message_type]
        if isinstance(message, Missing):
            return message
        if not isinstance(message, message_type):
            raise TypeError(f"stored message is not a {message_type.__name__}")
        return cast(M, message)


def jsonable(value: Any) -> Any:
    """Convert supported payloads to deterministic JSON-compatible values."""
    if hasattr(value, "to_dicts"):
        return jsonable(value.to_dicts())
    if isinstance(value, BaseModel):
        return jsonable(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, "get_params"):
        serialized = pickle.dumps(value, protocol=5)
        return {
            "class": type(value).__qualname__,
            "params": jsonable(value.get_params()),
            "fitted_state_sha256": hashlib.sha256(serialized).hexdigest(),
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def message_bytes(message: Message | Missing) -> bytes:
    """Canonical representation used for replay comparisons and output hashes."""
    payload = json.dumps(jsonable(message), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return payload.encode("utf-8")


def message_digest(message: Message | Missing) -> str:
    return hashlib.sha256(message_bytes(message)).hexdigest()
