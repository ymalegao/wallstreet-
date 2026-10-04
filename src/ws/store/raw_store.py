"""Immutable archive of raw API payloads, so normalization can be fixed and re-run without re-downloading."""

from __future__ import annotations

import gzip
import json
import uuid
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from ws.timeutil import utcnow


class RawStore:
    def __init__(self, root: Path) -> None:
        self.root = root / "raw"

    def write(self, source: str, records: Iterable[dict[str, Any]]) -> Path | None:
        records = list(records)
        if not records:
            return None
        now = utcnow()
        d = self.root / source / now.strftime("%Y-%m-%d")
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{now.strftime('%H%M%S')}-{uuid.uuid4().hex[:8]}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps({"fetched_at": now.isoformat(), "payload": r}) + "\n")
        return path

    def iter(self, source: str) -> Iterator[dict[str, Any]]:
        for path in sorted((self.root / source).rglob("*.jsonl.gz")):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    yield json.loads(line)
