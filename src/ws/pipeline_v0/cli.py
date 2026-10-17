"""Command line entry points for downloading bars and running the v0 backtest."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from ws.pipeline_v0.run import run_backtest
from ws.store.alpaca import download_alpaca_daily_bars
from ws.store.pit_bars import PITBarStore

_ROOT = Path(__file__).parents[3]
_DEFAULT_UNIVERSE = _ROOT / "configs/universe/sp100-current.json"
_DEFAULT_CONFIG = _ROOT / "configs/pipeline/v0.json"


def main(argv: Sequence[str] | None = None) -> int:
    """Parse a data-download or backtest command and execute it."""
    parser = argparse.ArgumentParser(prog="ws-pipeline-v0")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download-bars", help="download adjusted Alpaca daily bars")
    _add_date_args(download)
    download.add_argument("--store", type=Path, required=True)
    download.add_argument("--universe", type=Path, default=_DEFAULT_UNIVERSE)
    download.add_argument("--feed", default="sip")
    backtest = commands.add_parser("backtest", help="run the daily v0 graph against a PIT bar store")
    _add_date_args(backtest)
    backtest.add_argument("--store", type=Path, required=True)
    backtest.add_argument("--universe", type=Path, default=_DEFAULT_UNIVERSE)
    backtest.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    backtest.add_argument("--out", type=Path, required=True)
    backtest.add_argument("--cash", type=float, default=1000.0)
    args = parser.parse_args(argv)
    if args.command == "download-bars":
        universe = json.loads(args.universe.read_text(encoding="utf-8"))
        inserted = download_alpaca_daily_bars(
            PITBarStore(args.store),
            universe["symbols"],
            args.start,
            args.end,
            feed=args.feed,
        )
        print(f"inserted {inserted} adjusted daily bars for {args.start} through {args.end}")
        return 0

    result = asyncio.run(
        run_backtest(
            args.start,
            args.end,
            store=PITBarStore(args.store),
            universe_path=args.universe,
            config_path=args.config,
            out_dir=args.out,
            initial_cash=args.cash,
        )
    )
    print(f"run_id={result.run_id} final_equity={result.final_equity:.2f} fills={result.fill_count}")
    print(f"events={result.events_path} equity={result.equity_path}")
    return 0


def _add_date_args(parser: argparse.ArgumentParser) -> None:
    """Add inclusive start and end dates to a CLI subcommand."""
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)


if __name__ == "__main__":
    raise SystemExit(main())
