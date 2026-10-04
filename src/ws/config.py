"""Settings from environment variables (optionally loaded from a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    alpaca_key: str | None
    alpaca_secret: str | None
    finnhub_key: str | None
    sec_user_agent: str | None  # SEC requires "Name contact@domain" in the User-Agent

    def require(self, *names: str) -> None:
        missing = [n for n in names if not getattr(self, n)]
        if missing:
            raise RuntimeError(f"missing settings: {', '.join(missing)} (see .env.example)")


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        data_dir=Path(os.environ.get("WS_DATA_DIR", "data")),
        alpaca_key=os.environ.get("ALPACA_API_KEY"),
        alpaca_secret=os.environ.get("ALPACA_SECRET_KEY"),
        finnhub_key=os.environ.get("FINNHUB_API_KEY"),
        sec_user_agent=os.environ.get("SEC_USER_AGENT"),
    )
