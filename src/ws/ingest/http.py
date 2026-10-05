"""HTTP retries with shared, process-safe request spacing on this host."""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx

from ws.store.atomic import file_lock

RETRY_STATUS = {429, 500, 502, 503, 504}


def retry_delay(value: str | None, attempt: int) -> float:
    if value:
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                return max(0.0, (parsedate_to_datetime(value).astimezone(UTC) - datetime.now(UTC)).total_seconds())
            except (ValueError, TypeError, OverflowError):
                pass
    return min(2.0**attempt, 60.0)


class RateLimitedClient:
    """Budgets are shared by provider across adapters/processes using the same lock directory.

    All SEC requests must use budget='sec'. Other adapters derive a budget from their hostname.
    Containers on the same IP must mount the same WS_RATE_LIMIT_DIR. This cannot coordinate
    unrelated applications or other machines; those need an external shared limiter.
    """

    def __init__(
        self,
        base_url: str = "",
        headers: dict[str, str] | None = None,
        max_per_sec: float = 3.0,
        max_retries: int = 5,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        budget: str | None = None,
    ) -> None:
        if max_per_sec <= 0 or max_retries < 0:
            raise ValueError("positive request rate and nonnegative retries required")
        self._client = httpx.Client(base_url=base_url, headers=headers or {}, timeout=timeout, transport=transport)
        self._min_interval = 1.0 / max_per_sec
        self._last = 0.0
        self._max_retries = max_retries
        self.calls = 0
        self._budget = budget or (httpx.URL(base_url).host if base_url else None)
        self._shared = transport is None and bool(self._budget)

    def _wait(self) -> None:
        if not self._shared:
            wait = self._min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            return
        root = Path(os.environ.get("WS_RATE_LIMIT_DIR", f"{tempfile.gettempdir()}/ws-rates-{os.getuid()}"))
        name = hashlib.sha256(str(self._budget).encode()).hexdigest()[:20]
        state = root / f"{name}.txt"
        with file_lock(root / f"{name}.lock"):
            try:
                last = float(state.read_text())
            except (FileNotFoundError, ValueError):
                last = 0.0
            now = time.monotonic()
            # An earlier boot may leave a larger monotonic timestamp in a persistent directory.
            wait = self._min_interval - (now - last) if last <= now else 0.0
            if wait > 0:
                time.sleep(wait)
            state.write_text(str(time.monotonic()))

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        return self.get(url, params).json()

    def get(self, url: str, params: dict[str, Any] | None = None) -> httpx.Response:
        for attempt in range(self._max_retries + 1):
            self._wait()
            self.calls += 1
            try:
                resp = self._client.get(url, params=params)
            except httpx.TransportError:
                if attempt == self._max_retries:
                    raise
                time.sleep(retry_delay(None, attempt))
                continue
            if resp.status_code not in RETRY_STATUS or attempt == self._max_retries:
                resp.raise_for_status()
                return resp
            delay = retry_delay(resp.headers.get("retry-after"), attempt)
            if delay > 300:
                raise RuntimeError("Provider requested a cooldown longer than 5 minutes; resume later")
            time.sleep(delay)
        raise AssertionError("unreachable")

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> RateLimitedClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
