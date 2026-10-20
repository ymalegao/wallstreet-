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

import exchange_calendars as xcals
import polars as pl

from ws.broker import CostModel, SimBroker
from ws.core.clock import Tick
from ws.core.messages import Message
from ws.core.pipeline import NodeSpec, build_from_config
from ws.pipeline_v0.messages import MarketSnapshot, PortfolioState
from ws.pipeline_v0.metrics import adv_dollars_20d, realized_vol_21d
from ws.pipeline_v0.snapshot import SnapshotProvider
from ws.features.pxf_v1 import PXF_V1_LOOKBACK_SESSIONS
from ws.schedule import daily_ticks
from ws.store.history import BarHistory
from ws.store.pit_bars import PITBarStore

_NEW_YORK = ZoneInfo("America/New_York")
_PIPELINE_INPUTS = (MarketSnapshot, PortfolioState)
_BENCHMARK_SYMBOLS = ("SPY",)
_XNYS = xcals.get_calendar("XNYS")


@dataclass(frozen=True)
class BacktestResult:
    """Paths and final account summary emitted by one historical run."""

    run_id: str
    events_path: Path
    fills_path: Path
    cancelled_path: Path
    equity_path: Path
    baselines_path: Path
    contributions_path: Path
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
    cost_model: CostModel | None = None,
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
    baselines_path = out_dir / "baselines.csv"
    contributions_path = out_dir / "contributions.csv"
    existing = [
        path.name
        for path in (events_path, fills_path, cancelled_path, equity_path, baselines_path, contributions_path)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(f"output files already exist in {out_dir}: {existing}")

    universe_data = json.loads(universe_path.read_text(encoding="utf-8"))
    symbols = tuple(sorted({str(symbol).strip().upper() for symbol in universe_data["symbols"]}))
    history = BarHistory.load(store, (*symbols, *_BENCHMARK_SYMBOLS), through=ticks[-1].at)
    specs = [NodeSpec.model_validate(item) for item in json.loads(config_path.read_text(encoding="utf-8"))]
    feature_set = next(
        (
            str(spec.kwargs.get("feature_set", "mom-v0"))
            for spec in specs
            if spec.cls == "ws.pipeline_v0.nodes.FeatureBuilder"
        ),
        "mom-v0",
    )
    lookback_sessions = PXF_V1_LOOKBACK_SESSIONS if feature_set == "pxf-v1" else 300
    price_manifest_path = store.dir / "snapshot.json"
    price_snapshot_id = (
        str(json.loads(price_manifest_path.read_text(encoding="utf-8")).get("snapshot_id", "unknown"))
        if price_manifest_path.is_file()
        else "unknown"
    )
    provider = SnapshotProvider(
        history,
        universe_path,
        benchmark_symbols=_BENCHMARK_SYMBOLS,
        lookback_sessions=lookback_sessions,
        market_snapshot_id=price_snapshot_id,
    )
    active_cost_model = cost_model or CostModel()
    broker = SimBroker(history, active_cost_model, initial_cash=initial_cash)
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
    baseline_rows = _buy_and_hold_baselines(
        history,
        ticks,
        symbols,
        initial_cash=initial_cash,
        cost_model=active_cost_model,
    )
    _write_csv(
        baselines_path,
        (
            "date",
            "strategy",
            "equity",
            "net_return",
            "turnover",
            "transaction_costs",
            "constituent_count",
            "excluded_symbols",
        ),
        baseline_rows,
    )
    contribution_rows = _stock_contributions(history, ticks, broker.fills)
    _write_csv(contributions_path, ("date", "symbol", "contribution"), contribution_rows)
    return BacktestResult(
        run_id=run_id,
        events_path=events_path,
        fills_path=fills_path,
        cancelled_path=cancelled_path,
        equity_path=equity_path,
        baselines_path=baselines_path,
        contributions_path=contributions_path,
        final_equity=float(equity_rows[-1]["equity"]),
        fill_count=len(broker.fills),
    )


def _buy_and_hold_baselines(
    history: BarHistory,
    ticks: list[Tick],
    symbols: tuple[str, ...],
    *,
    initial_cash: float,
    cost_model: CostModel,
) -> list[dict[str, str | float]]:
    """Mark equal-weight and SPY buy-and-hold from the strategy's first fill session."""
    first_tick_date = ticks[0].at.astimezone(_NEW_YORK).date()
    session = _XNYS.date_to_session(first_tick_date.isoformat(), direction="none")
    fill_session = _XNYS.next_session(session)
    fill_date = fill_session.date()
    fill_open = _XNYS.session_open(fill_session).to_pydatetime()
    known = history.visible(fill_open)
    adv_rows = adv_dollars_20d(known)
    vol_rows = realized_vol_21d(known)
    adv = {row["symbol"]: row["adv_dollars_20d"] for row in adv_rows.iter_rows(named=True)}
    volatility = {row["symbol"]: row["vol_21d"] for row in vol_rows.iter_rows(named=True)}
    history_through = history.visible(ticks[-1].at)
    fill_bars = history_through.filter(
        pl.col("event_time").dt.convert_time_zone("America/New_York").dt.date() == fill_date
    )
    prices = {row["symbol"]: float(row["open"]) for row in fill_bars.iter_rows(named=True)}
    bars_by_session: dict[tuple[str, date], tuple[float, float]] = {}
    for row in history_through.iter_rows(named=True):
        day = row["event_time"].astimezone(_NEW_YORK).date()
        bars_by_session[(row["symbol"], day)] = (float(row["open"]), float(row["close"]))

    available_equities = tuple(
        symbol
        for symbol in symbols
        if symbol in prices and adv.get(symbol) is not None and volatility.get(symbol) is not None
    )
    excluded_equities = tuple(sorted(set(symbols) - set(available_equities)))
    if not available_equities:
        raise ValueError("equal_weight_buy_hold has no constituents with bars and pre-fill cost history")
    strategies = {
        "equal_weight_buy_hold": (available_equities, excluded_equities),
        "spy_buy_hold": (("SPY",), ()),
    }
    output: list[dict[str, str | float]] = []
    for strategy, (holdings, excluded) in strategies.items():
        if any(symbol not in prices for symbol in holdings):
            missing = sorted(symbol for symbol in holdings if symbol not in prices)
            raise ValueError(f"baseline {strategy} lacks bars on its first fill session: {missing}")
        if any(adv.get(symbol) is None or volatility.get(symbol) is None for symbol in holdings):
            raise ValueError(f"baseline {strategy} lacks pre-fill cost history")
        unit_weight = 1.0 / len(holdings)

        def required(scale: float, holdings: tuple[str, ...] = holdings, unit_weight: float = unit_weight) -> float:
            total = 0.0
            for symbol in holdings:
                notional = initial_cash * unit_weight * scale
                total += notional + cost_model.dollars(
                    order_notional=notional,
                    adv_dollars=float(adv[symbol]),
                    annual_volatility=float(volatility[symbol]),
                )
            return total

        low, high = 0.0, 1.0
        for _ in range(64):
            middle = (low + high) / 2.0
            if required(middle) <= initial_cash:
                low = middle
            else:
                high = middle
        scale = low
        initial_cost = sum(
            cost_model.dollars(
                order_notional=initial_cash * unit_weight * scale,
                adv_dollars=float(adv[symbol]),
                annual_volatility=float(volatility[symbol]),
            )
            for symbol in holdings
        )
        leftover = initial_cash - initial_cash * scale - initial_cost
        shares = {symbol: initial_cash * unit_weight * scale / prices[symbol] for symbol in holdings}
        invested = initial_cash * scale
        prior_equity = initial_cash
        for tick in ticks:
            day = tick.at.astimezone(_NEW_YORK).date()
            equity = initial_cash if day < fill_date else leftover + sum(
                shares[symbol] * bars_by_session[(symbol, day)][1] for symbol in holdings
            )
            if day == fill_date:
                turnover = invested / initial_cash
                transaction_costs = initial_cost
            else:
                turnover = 0.0
                transaction_costs = 0.0
            output.append(
                {
                    "date": day.isoformat(),
                    "strategy": strategy,
                    "equity": equity,
                    "net_return": equity / prior_equity - 1.0 if prior_equity else 0.0,
                    "turnover": turnover,
                    "transaction_costs": transaction_costs,
                    "constituent_count": float(len(holdings)),
                    "excluded_symbols": ",".join(excluded),
                }
            )
            prior_equity = equity
    return sorted(output, key=lambda row: (str(row["date"]), str(row["strategy"])))


def _stock_contributions(
    history: BarHistory,
    ticks: list[Tick],
    fills: list[dict[str, str | float]],
) -> list[dict[str, str | float]]:
    """Attribute daily mark-to-market P&L and execution costs to individual stocks."""
    all_bars = history.visible(ticks[-1].at)
    bars_by_date: dict[date, dict[str, tuple[float, float]]] = {}
    for row in all_bars.iter_rows(named=True):
        session_date = row["event_time"].astimezone(_NEW_YORK).date()
        bars_by_date.setdefault(session_date, {})[row["symbol"]] = (float(row["open"]), float(row["close"]))
    fills_by_date: dict[str, list[dict[str, str | float]]] = {}
    for fill in fills:
        fills_by_date.setdefault(str(fill["session"]), []).append(fill)
    holdings: dict[str, float] = {}
    previous_close: dict[str, float] = {}
    result: list[dict[str, str | float]] = []
    for tick in ticks:
        session_date = tick.at.astimezone(_NEW_YORK).date()
        day_bars = bars_by_date.get(session_date, {})
        updated = dict(holdings)
        costs: dict[str, float] = {}
        for fill in fills_by_date.get(session_date.isoformat(), []):
            symbol = str(fill["symbol"])
            shares = float(fill["shares"])
            updated[symbol] = updated.get(symbol, 0.0) + (shares if fill["side"] == "buy" else -shares)
            costs[symbol] = costs.get(symbol, 0.0) + float(fill["cost"])
        for symbol in sorted(set(holdings) | set(updated) | set(costs)):
            if symbol not in day_bars:
                raise ValueError(f"cannot attribute contribution for {symbol} on {session_date}: missing bar")
            opening, closing = day_bars[symbol]
            overnight = holdings.get(symbol, 0.0) * (opening - previous_close.get(symbol, opening))
            intraday = updated.get(symbol, 0.0) * (closing - opening)
            contribution = overnight + intraday - costs.get(symbol, 0.0)
            if contribution != 0:
                result.append(
                    {"date": session_date.isoformat(), "symbol": symbol, "contribution": contribution}
                )
        holdings = {symbol: shares for symbol, shares in updated.items() if shares > 1e-12}
        previous_close.update({symbol: prices[1] for symbol, prices in day_bars.items()})
    return result


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: Iterable[Mapping[str, object]]) -> None:
    """Write deterministic CSV output with a header even when no rows were produced."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
