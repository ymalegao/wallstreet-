"""Alpaca historical daily-bar ingestion into the point-in-time bar store."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx

from ws.bars import Bar
from ws.store.atomic import atomic_text
from ws.store.pit_bars import NEW_YORK, PITBarStore, daily_bar_to_pit

_BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"


def download_alpaca_daily_bars(
    store: PITBarStore,
    symbols: Sequence[str],
    start: date,
    end: date,
    *,
    api_key: str | None = None,
    secret_key: str | None = None,
    feed: str = "sip",
    client: httpx.Client | None = None,
) -> int:
    """Download split- and dividend-adjusted daily bars and append them to the PIT store."""
    normalized = tuple(sorted({symbol.strip().upper() for symbol in symbols if symbol.strip()}))
    if not normalized:
        raise ValueError("at least one non-empty symbol is required")
    if end < start:
        raise ValueError("end must be on or after start")
    if not feed.strip():
        raise ValueError("feed cannot be empty")
    if (store.dir / "snapshot.json").exists() or any(store.dir.glob("year=*/month=*/part-*.parquet")):
        raise FileExistsError(
            f"bar snapshot store is immutable and already contains data: {store.dir}; choose a new store path"
        )
    key, secret = _resolve_credentials(api_key, secret_key)
    if not key or not secret:
        raise ValueError(
            "set ALPACA_KEY and ALPACA_SECRET (or APCA_API_KEY_ID and "
            "APCA_API_SECRET_KEY) in the environment or .env file, or pass credentials explicitly"
        )

    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    params: dict[str, str | int] = {
        "symbols": ",".join(normalized),
        "timeframe": "1Day",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "adjustment": "all",
        "feed": feed,
        "sort": "asc",
        "limit": 10_000,
    }
    inserted = 0
    seen_tokens: set[str] = set()
    snapshot_hash = hashlib.sha256()
    first_seen: dict[str, date] = {}
    last_seen: dict[str, date] = {}
    owned_client = client is None
    active_client = client or httpx.Client(timeout=30.0)
    try:
        while True:
            response = active_client.get(_BARS_URL, params=params, headers=headers)
            if response.is_error:
                raise RuntimeError(
                    f"Alpaca historical bars request failed with HTTP {response.status_code}: {response.text}"
                )
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Alpaca bars response must be a JSON object")
            bars_by_symbol = payload.get("bars")
            if not isinstance(bars_by_symbol, dict):
                raise ValueError("Alpaca bars response is missing its 'bars' object")
            if any(not isinstance(raw_bars, list) for raw_bars in bars_by_symbol.values()):
                raise ValueError("Alpaca bars response contains a malformed symbol list")
            batch = [
                daily_bar_to_pit(_parse_bar(symbol, raw_bar))
                for symbol, raw_bars in bars_by_symbol.items()
                for raw_bar in raw_bars
            ]
            batch.sort(key=lambda bar: (bar.symbol, bar.event_time))
            for bar in batch:
                session_date = bar.event_time.astimezone(NEW_YORK).date()
                first_seen[bar.symbol] = min(first_seen.get(bar.symbol, session_date), session_date)
                last_seen[bar.symbol] = max(last_seen.get(bar.symbol, session_date), session_date)
            encoded_batch = json.dumps(
                [bar.model_dump(mode="json") for bar in batch], sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            snapshot_hash.update(encoded_batch)
            inserted += store.write(batch)
            next_token = payload.get("next_page_token")
            if not next_token:
                break
            token = str(next_token)
            if token in seen_tokens:
                raise ValueError("Alpaca repeated a pagination token")
            seen_tokens.add(token)
            params["page_token"] = token
        snapshot_id = hashlib.sha256(
            json.dumps(
                {
                    "adjustment": "all",
                    "feed": feed,
                    "request_end": end.isoformat(),
                    "request_start": start.isoformat(),
                    "rows_sha256": snapshot_hash.hexdigest(),
                    "symbols": normalized,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        manifest = {
            "snapshot_id": snapshot_id,
            "feed": feed,
            "adjustment": "all",
            "symbols": list(normalized),
            "requested_start": start.isoformat(),
            "requested_end": end.isoformat(),
            "actual_earliest_by_symbol": {symbol: value.isoformat() for symbol, value in sorted(first_seen.items())},
            "actual_latest_by_symbol": {symbol: value.isoformat() for symbol, value in sorted(last_seen.items())},
        }
        manifest_path = store.dir / "snapshot.json"
        atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    finally:
        if owned_client:
            active_client.close()
    return inserted


def _resolve_credentials(api_key: str | None, secret_key: str | None) -> tuple[str | None, str | None]:
    """Resolve explicit credentials, environment variables, then known keys in cwd/.env."""
    dotenv_values = _read_dotenv_credentials(Path.cwd() / ".env")
    key = api_key or _first_value(
        os.getenv("APCA_API_KEY_ID"),
        os.getenv("ALPACA_KEY"),
        dotenv_values.get("APCA_API_KEY_ID"),
        dotenv_values.get("ALPACA_KEY"),
    )
    secret = secret_key or _first_value(
        os.getenv("APCA_API_SECRET_KEY"),
        os.getenv("ALPACA_SECRET"),
        dotenv_values.get("APCA_API_SECRET_KEY"),
        dotenv_values.get("ALPACA_SECRET"),
    )
    return key, secret


def _read_dotenv_credentials(path: Path) -> dict[str, str]:
    """Read only the supported Alpaca keys from a simple dotenv file, without executing it."""
    supported = {"ALPACA_KEY", "ALPACA_SECRET", "APCA_API_KEY_ID", "APCA_API_SECRET_KEY"}
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        name, separator, value = line.partition("=")
        if not separator or name.strip() not in supported:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if value:
            values[name.strip()] = value
    return values


def _first_value(*values: str | None) -> str | None:
    """Return the first non-empty credential string."""
    return next((value for value in values if value), None)


def _parse_bar(symbol: str, raw: Any) -> Bar:
    """Validate one Alpaca daily-bar payload and convert its fields to the shared bar type."""
    if not isinstance(raw, dict):
        raise ValueError(f"Alpaca returned a malformed bar for {symbol}")
    timestamp = raw.get("t")
    if not isinstance(timestamp, str):
        raise ValueError(f"Alpaca returned a bar without a timestamp for {symbol}")
    parsed_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return Bar(
        symbol=symbol,
        ts=parsed_time,
        open=float(raw["o"]),
        high=float(raw["h"]),
        low=float(raw["l"]),
        close=float(raw["c"]),
        volume=float(raw["v"]),
        vwap=float(raw["vw"]) if raw.get("vw") is not None else None,
        trade_count=int(raw["n"]) if raw.get("n") is not None else None,
    )
