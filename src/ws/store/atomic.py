"""Local-filesystem locks and atomic replacement; shared by writers and checkpoints."""

from __future__ import annotations

import fcntl
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    """Readers see either the old complete file or the new complete file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp = Path(name)
    try:
        yield tmp
        with tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_text(path: Path, text: str) -> None:
    with atomic_path(path) as tmp:
        tmp.write_text(text)
