"""Rate-limited HTTP client with retries. All source adapters go through this."""

from __future__ import annotations

import time
from typing import Any

import httpx

RETRY_STATUS = {429, 500, 502, 503, 504}


class RateLimitedClient:
    """Synchronous client that spaces requests to stay under ``max_per_sec`` and retries transient errors."""

    def __init__(
        self,
        base_url: str = "",
        headers: dict[str, str] | None = None,
        max_per_sec: float = 3.0,
        max_retries: int = 5,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(base_url=base_url, headers=headers or {}, timeout=timeout, transport=transport)
        self._min_interval = 1.0 / max_per_sec
        self._last = 0.0
        self._max_retries = max_retries
        self.calls = 0

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        return self.get(url, params).json()

    def get(self, url: str, params: dict[str, Any] | None = None) -> httpx.Response:
        for attempt in range(self._max_retries + 1):
            wait = self._min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.calls += 1
            resp = self._client.get(url, params=params)
            if resp.status_code not in RETRY_STATUS or attempt == self._max_retries:
                resp.raise_for_status()
                return resp
            retry_after = resp.headers.get("retry-after")
            time.sleep(float(retry_after) if retry_after else min(2.0**attempt, 60.0))
        raise AssertionError("unreachable")

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> RateLimitedClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
