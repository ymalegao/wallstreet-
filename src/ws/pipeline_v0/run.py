"""Coordinator and artifact writer for reproducible daily momentum backtests."""

from __future__ import annotations

import csv
import json
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

from ws.broker import CostModel, SimBroker
from ws.core.clock import Tick
from ws.core.messages import Message
from ws.core.pipeline import NodeSpec, build_from_config
from ws.pipeline_v0.messages import MarketSnapshot, PortfolioState
from ws.pipeline_v0.snapshot import SnapshotProvider
from ws.schedule import daily_ticks
from ws.store.history import BarHistory
from ws.store.pit_bars import PITBarStore

_NEW_YORK = ZoneInfo("America/New_York")
_PIPELINE_INPUTS = (MarketSnapshot, PortfolioState)


@dataclass(frozen=True)
class BacktestResult:
    """Paths and final account summary emitted by one historical run."""

    run_id: str
    events_path: Path
    fills_path: Path
    cancelled_path: Path
    equity_path: Path
    final_equity: float
    fill_count: int


async def run_backtest(
    start: date,
    end: date,
    *,
    store: PITBarStore,
    universe_path: Path,
    config_path: Path,
    out_dir: Path,
    initial_cash: float = 1000.0,
) -> BacktestResult:
    """Run the configured daily graph over preloaded point-in-time bars and write CSV artifacts."""
    ticks = daily_ticks(start, end)
    if not ticks:
        raise ValueError("the requested date range contains no XNYS sessions")
    if initial_cash <= 0:
        raise ValueError("initial_cash must be positive")
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "events.jsonl"
    fills_path = out_dir / "fills.csv"
    cancelled_path = out_dir / "cancelled.csv"
    equity_path = out_dir / "equity.csv"
    existing = [path.name for path in (events_path, fills_path, cancelled_path, equity_path) if path.exists()]
    if existing:
        raise FileExistsError(f"output files already exist in {out_dir}: {existing}")

    universe_data = json.loads(universe_path.read_text(encoding="utf-8"))
    symbols = tuple(sorted({str(symbol).strip().upper() for symbol in universe_data["symbols"]}))
    history = BarHistory.load(store, symbols, through=ticks[-1].at)
    provider = SnapshotProvider(history, universe_path)
    broker = SimBroker(history, CostModel(), initial_cash=initial_cash)
    specs = [NodeSpec.model_validate(item) for item in json.loads(config_path.read_text(encoding="utf-8"))]
    pipeline = build_from_config(
        specs,
        resources={"broker": broker},
        inputs=_PIPELINE_INPUTS,
    )
    run_id = f"daily-{uuid.uuid4().hex[:12]}"
    equity_rows: list[dict[str, str | float]] = []
    tick_states: dict[str, PortfolioState] = {}

    def inputs_for_tick(tick: Tick) -> Mapping[type[Message], Message]:
        snapshot = provider.snapshot(tick)
        state = broker.state(as_of=tick.at)
        portfolio_state = PortfolioState(
            tick=tick,
            producer="coordinator",
            cash=state.cash,
            positions=state.positions,
            market_values=state.market_values,
            equity=state.equity,
            peak_equity=state.peak_equity,
        )
        tick_states[tick.id] = portfolio_state
        return {MarketSnapshot: snapshot, PortfolioState: portfolio_state}

    await pipeline.run_many(
        ticks,
        run_id=run_id,
        log_path=events_path,
        inputs_for_tick=inputs_for_tick,
    )
    for tick in ticks:
        state = tick_states[tick.id]
        gross = sum(abs(value) for value in state.market_values.values())
        equity_rows.append(
            {
                "date": tick.at.astimezone(_NEW_YORK).date().isoformat(),
                "cash": state.cash,
                "equity": state.equity,
                "gross_exposure": gross / state.equity if state.equity else 0.0,
            }
        )
    _write_csv(
        fills_path,
        ("session", "symbol", "side", "shares", "price", "cost"),
        broker.fills,
    )
    _write_csv(
        cancelled_path,
        ("session", "symbol", "side", "notional", "reason"),
        broker.cancelled,
    )
    _write_csv(equity_path, ("date", "cash", "equity", "gross_exposure"), equity_rows)
    return BacktestResult(
        run_id=run_id,
        events_path=events_path,
        fills_path=fills_path,
        cancelled_path=cancelled_path,
        equity_path=equity_path,
        final_equity=float(equity_rows[-1]["equity"]),
        fill_count=len(broker.fills),
    )


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: Iterable[Mapping[str, object]]) -> None:
    """Write deterministic CSV output with a header even when no rows were produced."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
