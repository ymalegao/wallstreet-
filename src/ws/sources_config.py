"""Access to configs/sources.yaml values that must be verified by the API probe before use."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path("configs/sources.yaml")
UNVERIFIED = "UNVERIFIED"


class UnverifiedSetting(RuntimeError):
    pass


def load(path: Path = DEFAULT_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def verified(cfg: dict[str, Any], dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        node = node[part]
    if node == UNVERIFIED:
        raise UnverifiedSetting(
            f"configs/sources.yaml `{dotted}` is UNVERIFIED: run scripts/probe_apis.py and record the measured value"
        )
    return node
