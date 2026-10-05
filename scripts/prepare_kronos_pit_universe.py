"""Freeze a historical, prior-session liquidity universe for Kronos evaluation."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from ws import calendar as cal
from ws import universe
from ws.store.atomic import atomic_text
from ws.store.event_store import BarStore


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", type=Path, default=Path("data/universe/alpaca_assets_2026-10-04.jsonl"))
    ap.add_argument("--daily-store", type=str, default="1Day_raw")
    ap.add_argument("--start", type=date.fromisoformat, default=date(2024, 7, 1))
    ap.add_argument("--end", type=date.fromisoformat, default=date(2026, 7, 1), help="exclusive decision date")
    ap.add_argument("--top-n", type=int, default=25)
    ap.add_argument("--output-dir", type=Path, default=Path("data/kronos-pit-2024-2026"))
    args = ap.parse_args()
    if args.end <= args.start:
        raise ValueError("end must be after start")

    assets = [json.loads(line) for line in args.assets.read_text().splitlines() if line.strip()]
    candidates: list[dict[str, Any]] = universe.listed_stock_candidates(assets)
    symbols = sorted({str(asset["symbol"]) for asset in candidates})
    daily = BarStore(Path("data"), args.daily_store).read(symbols)
    selected = universe.point_in_time_top_n(daily, symbols, args.start, args.end, args.top_n)
    if selected.is_empty():
        raise RuntimeError("No eligible ticker-sessions in the requested period")

    signals: list[dict[str, str]] = []
    by_session: dict[str, list[str]] = {}
    sessions = sorted(selected["session"].unique().to_list())
    for session in sessions:
        session_key = session.isoformat()
        tickers = sorted(selected.filter(pl.col("session") == session)["symbol"].to_list())
        by_session[session_key] = tickers
        morning, afternoon = cal.cycles_for_session(session)
        for ticker in tickers:
            signals.append({"ticker": ticker, "cycle": morning.isoformat()})
            signals.append({"ticker": ticker, "cycle": afternoon.isoformat()})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    signal_path = args.output_dir / "signals.json"
    universe_path = args.output_dir / "universe.json"
    atomic_text(signal_path, json.dumps(signals) + "\n")
    symbols_used = sorted({row["ticker"] for row in signals})
    metadata = {
        "status": "EXPLORATORY_POINT_IN_TIME_LIQUIDITY_SELECTION",
        "decision_window": {"start_inclusive": args.start.isoformat(), "end_exclusive": args.end.isoformat()},
        "exit_label_buffer": "five subsequent sessions; outcomes may extend into July 2026",
        "selector": {
            "top_n_per_session": args.top_n,
            "price_floor_usd": universe.MIN_PRICE,
            "median_dollar_volume_floor_usd": universe.MIN_DOLLAR_VOLUME,
            "median_window_sessions": universe.WINDOW,
            "all_features_lagged_one_session": True,
            "input_daily_store": args.daily_store,
        },
        "candidate_pool": {
            "source": str(args.assets),
            "as_of": "2026-10-04 current/inactive asset snapshot",
            "candidate_count": len(symbols),
            "limitations": [
                "Price/liquidity ranking is point-in-time, but the candidate master is a current/inactive "
                "snapshot, not historical exchange membership.",
                "Known delisted names absent from the snapshot/bar cache (including BBBY) cannot enter; "
                "survivorship-free coverage is not certified.",
                "The 2024-07 to 2026-06 feature period is after Kronos's stated June 2024 pretraining "
                "cutoff, but project-level price/universe research has touched overlapping dates.",
            ],
        },
        "sessions": len(sessions),
        "ticker_sessions": selected.height,
        "unique_selected_symbols": len(symbols_used),
        "selected_symbols": symbols_used,
        "selection_by_session": by_session,
        "signals_path": str(signal_path),
        "signals_count": len(signals),
        "use_news": False,
    }
    atomic_text(universe_path, json.dumps(metadata, indent=2) + "\n")
    print(
        f"Frozen {selected.height} ticker-sessions over {len(sessions)} sessions; "
        f"{len(symbols_used)} selected symbols; {len(signals)} decision inputs; "
        f"wrote {signal_path} and {universe_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
