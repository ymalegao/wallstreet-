"""Append-only JSON Lines event log."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ws.core.messages import Message, Missing, message_digest


class EventLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def emit(self, event: str, **fields: Any) -> None:
        row = {"event": event, **fields}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True, default=str, separators=(",", ":")) + "\n")

    @staticmethod
    def digest(message: Message | Missing) -> str:
        return message_digest(message)
