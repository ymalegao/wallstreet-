from __future__ import annotations

import asyncio
import csv
import json
import math
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from statistics import mean, stdev
from zoneinfo import ZoneInfo

import httpx
import polars as pl
import pytest
from fixtures.pipeline_v0 import build_fixture
from pydantic import ConfigDict

from ws.bars import Bar
from ws.broker import BrokerState, CostModel, SimBroker
from ws.core import Inputs, Message, Node, NodeContext, Pipeline, SimClock, Tick
from ws.core.event_log import EventLogger
from ws.core.messages import message_digest
from ws.core.pipeline import NodeSpec
from ws.pipeline_v0.messages import (
    Features,
    Insights,
    MarketSnapshot,
    Order,
    Orders,
    PortfolioState,
    RiskAdjustedTargets,
)
from ws.pipeline_v0.metrics import adv_dollars_20d, momentum_12_1, realized_vol_21d
from ws.pipeline_v0.nodes import Execution
from ws.pipeline_v0.run import BacktestResult, _buy_and_hold_baselines, run_backtest
from ws.pipeline_v0.snapshot import SnapshotProvider
from ws.schedule import daily_ticks
from ws.store.alpaca import download_alpaca_daily_bars
from ws.store.history import BarHistory
from ws.store.pit_bars import PITBarStore, daily_bar_to_pit

_ROOT = Path(__file__).parents[1]
_CONFIG = _ROOT / "configs/pipeline/v0.json"


@pytest.fixture(scope="module")
def pipeline_fixture(tmp_path_factory: pytest.TempPathFactory) -> tuple[PITBarStore, Path, tuple[str, ...]]:
    """Create one reusable deterministic three-year point-in-time dataset."""
    return build_fixture(tmp_path_factory.mktemp("pipeline-v0-data"))


class RecordingBroker:
    """Record submitted orders for isolated execution-node tests."""

    def __init__(self) -> None:
        self.submitted: list[tuple[tuple[Order, ...], datetime]] = []

    def state(self, as_of: datetime) -> BrokerState:
        """Return an empty funded account for construction-only tests."""
        return BrokerState(cash=1000.0, positions={}, market_values={}, equity=1000.0, peak_equity=1000.0)

    def submit(self, orders: Sequence[Order], placed_at: datetime) -> None:
        """Retain each submitted order batch and its decision time."""
        self.submitted.append((tuple(orders), placed_at))


def _specs() -> list[dict[str, object]]:
    return json.loads(_CONFIG.read_text(encoding="utf-8"))


def _build_pipeline(broker: SimBroker) -> Pipeline:
    from ws.core.pipeline import NodeSpec, build_from_config

    return build_from_config(
        [NodeSpec.model_validate(item) for item in _specs()],
        resources={"broker": broker},
        inputs=(MarketSnapshot, PortfolioState),
    )


def _decision_tick(day: date) -> Tick:
    """Return the scheduled close-plus-thirty tick for one XNYS session."""
    return daily_ticks(day, day)[0]


def _snapshot(tick: Tick, bars: pl.DataFrame, symbols: tuple[str, ...]) -> MarketSnapshot:
    """Build a market snapshot for direct node tests."""
    return MarketSnapshot(
        tick=tick,
        producer="test-snapshot",
        universe=symbols,
        universe_id="test-universe",
        bars=bars,
        last_session=tick.at.astimezone(UTC).date(),
    )


def test_alpaca_downloader_paginates_and_requests_adjusted_bars(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.params.get("page_token") == "page-2":
            payload = {
                "bars": {
                    "MSFT": [
                        {
                            "t": "2023-01-04T05:00:00Z",
                            "o": 20,
                            "h": 22,
                            "l": 19,
                            "c": 21,
                            "v": 2000,
                            "vw": 20.5,
                            "n": 30,
                        }
                    ]
                },
                "next_page_token": None,
            }
        else:
            payload = {
                "bars": {
                    "AAPL": [
                        {
                            "t": "2023-01-03T05:00:00Z",
                            "o": 10,
                            "h": 12,
                            "l": 9,
                            "c": 11,
                            "v": 1000,
                            "vw": 10.5,
                            "n": 20,
                        }
                    ]
                },
                "next_page_token": "page-2",
            }
        return httpx.Response(200, json=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    store = PITBarStore(tmp_path / "alpaca-bars")
    inserted = download_alpaca_daily_bars(
        store,
        ("aapl", "msft"),
        date(2023, 1, 3),
        date(2023, 1, 4),
        api_key="test-key",
        secret_key="test-secret",
        client=client,
    )
    assert inserted == 2
    assert len(requests) == 2
    assert requests[0].url.params["adjustment"] == "all"
    assert requests[0].url.params["timeframe"] == "1Day"
    assert requests[1].url.params["page_token"] == "page-2"
    assert requests[0].headers["APCA-API-KEY-ID"] == "test-key"
    snapshot = json.loads((store.dir / "snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["feed"] == "sip"
    with pytest.raises(FileExistsError, match="immutable"):
        download_alpaca_daily_bars(
            store,
            ("AAPL", "MSFT"),
            date(2023, 1, 3),
            date(2023, 1, 4),
            api_key="test-key",
            secret_key="test-secret",
            client=client,
        )
    assert len(requests) == 2


def test_alpaca_downloader_reads_repo_style_dotenv_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "ALPACA_KEY", "ALPACA_SECRET"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text(
        "ALPACA_KEY=dotenv-key\nALPACA_SECRET='dotenv-secret'\n",
        encoding="utf-8",
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"bars": {}, "next_page_token": None})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    inserted = download_alpaca_daily_bars(
        PITBarStore(tmp_path / "dotenv-bars"),
        ("AAPL",),
        date(2023, 1, 3),
        date(2023, 1, 3),
        client=client,
    )

    assert inserted == 0
    assert requests[0].headers["APCA-API-KEY-ID"] == "dotenv-key"
    assert requests[0].headers["APCA-API-SECRET-KEY"] == "dotenv-secret"


def test_t3_default_config_builds_with_external_root_messages() -> None:
    from ws.core.pipeline import NodeSpec, build_from_config

    pipeline = build_from_config(
        [NodeSpec.model_validate(item) for item in _specs()],
        resources={"broker": RecordingBroker()},
        inputs=(MarketSnapshot, PortfolioState),
    )
    assert [[node.name for node in level] for level in pipeline.levels] == [
        ["features"],
        ["alpha"],
        ["portfolio_construction"],
        ["risk_management"],
        ["execution"],
    ]


def test_t4_dataframe_digest_is_invariant_to_row_and_column_order() -> None:
    class FrameMessage(Message):
        model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
        frame: pl.DataFrame

    tick = _decision_tick(date(2023, 1, 3))
    first = FrameMessage(
        tick=tick,
        producer="test",
        frame=pl.DataFrame({"b": [2, 1], "a": [4, 3]}),
    )
    reordered = FrameMessage(
        tick=tick,
        producer="test",
        frame=pl.DataFrame({"a": [3, 4], "b": [1, 2]}),
    )
    changed = FrameMessage(
        tick=tick,
        producer="test",
        frame=pl.DataFrame({"a": [3, 4], "b": [1, 9]}),
    )
    assert message_digest(first) == message_digest(reordered)
    assert message_digest(first) != message_digest(changed)


def test_t5_feature_formulas_match_hand_calculation() -> None:
    start = datetime(2021, 1, 4, 21, tzinfo=UTC)
    closes = [100.0 + index for index in range(260)]
    volumes = [1000.0 + index for index in range(260)]
    bars = pl.DataFrame(
        {
            "symbol": ["TEST"] * 260,
            "event_time": [start + timedelta(days=index) for index in range(260)],
            "close": closes,
            "volume": volumes,
        }
    )
    expected_momentum = closes[-22] / closes[-253] - 1.0
    daily_returns = [math.log(closes[index] / closes[index - 1]) for index in range(239, 260)]
    expected_volatility = stdev(daily_returns) * math.sqrt(252)
    expected_adv = mean(closes[index] * volumes[index] for index in range(240, 260))

    assert momentum_12_1(bars).item(0, "mom_12_1") == pytest.approx(expected_momentum)
    assert realized_vol_21d(bars).item(0, "vol_21d") == pytest.approx(expected_volatility)
    assert adv_dollars_20d(bars).item(0, "adv_dollars_20d") == pytest.approx(expected_adv)


def test_t6_future_rows_do_not_change_tick_outputs(
    pipeline_fixture: tuple[PITBarStore, Path, tuple[str, ...]], tmp_path: Path
) -> None:
    store, universe_path, symbols = pipeline_fixture
    later_store = PITBarStore(tmp_path / "later-store")
    shutil.copytree(store.dir, later_store.dir)
    future_day = date(2024, 1, 2)
    future_ts = datetime.combine(future_day, time(12), UTC)
    later_store.write(
        [
            daily_bar_to_pit(
                Bar(
                    symbol=symbols[0],
                    ts=future_ts,
                    open=9999.0,
                    high=10001.0,
                    low=9998.0,
                    close=10000.0,
                    volume=9_000_000.0,
                )
            )
        ]
    )
    tick = _decision_tick(date(2023, 12, 29))
    base_history = BarHistory.load(store, symbols, through=tick.at)
    later_history = BarHistory.load(
        later_store,
        symbols,
        through=datetime(2024, 1, 3, 22, tzinfo=UTC),
    )

    def run_once(history: BarHistory, name: str) -> dict[type[Message], Message]:
        broker = SimBroker(history, CostModel())
        pipeline = _build_pipeline(broker)
        snapshot = SnapshotProvider(history, universe_path).snapshot(tick)
        state = broker.state(tick.at)
        portfolio = PortfolioState(
            tick=tick,
            producer="coordinator",
            cash=state.cash,
            positions=state.positions,
            market_values=state.market_values,
            equity=state.equity,
            peak_equity=state.peak_equity,
        )
        return asyncio.run(
            pipeline.run(
                tick,
                run_id="future-probe",
                log_path=tmp_path / f"{name}.jsonl",
                inputs={MarketSnapshot: snapshot, PortfolioState: portfolio},
            )
        )

    baseline = run_once(base_history, "before")
    with_future = run_once(later_history, "after")
    assert baseline.keys() == with_future.keys()
    assert {key: message_digest(value) for key, value in baseline.items()} == {
        key: message_digest(value) for key, value in with_future.items()
    }


def test_t7_execution_orders_sell_first_and_skips_small_deltas(tmp_path: Path) -> None:
    tick = _decision_tick(date(2023, 12, 29))
    bars = pl.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD"],
            "event_time": [datetime(2023, 12, 29, 21, tzinfo=UTC)] * 4,
            "available_at": [datetime(2023, 12, 29, 21, 15, tzinfo=UTC)] * 4,
            "open": [10.0, 20.0, 50.0, 100.0],
            "high": [10.0, 20.0, 50.0, 100.0],
            "low": [10.0, 20.0, 50.0, 100.0],
            "close": [10.0, 20.0, 50.0, 100.0],
            "volume": [1000.0] * 4,
            "vwap": [None] * 4,
            "adj_factor": [None] * 4,
        }
    )
    snapshot = _snapshot(tick, bars, ("AAA", "BBB", "CCC", "DDD"))
    portfolio = PortfolioState(
        tick=tick,
        producer="coordinator",
        cash=400.0,
        positions={"AAA": 10.0, "BBB": 20.0, "DDD": 1.0},
        market_values={"AAA": 100.0, "BBB": 400.0, "DDD": 100.0},
        equity=1000.0,
        peak_equity=1000.0,
    )
    targets = RiskAdjustedTargets(
        tick=tick,
        producer="risk",
        weights={"BBB": 0.2, "CCC": 0.3, "DDD": 0.0995},
        gross_exposure=0.5995,
        constraints=(),
    )
    broker = RecordingBroker()
    node = Execution(broker, min_order_dollars=1.0)
    clock = SimClock([tick])
    ctx = NodeContext(
        clock=clock,
        inputs=Inputs({RiskAdjustedTargets: targets, PortfolioState: portfolio, MarketSnapshot: snapshot}),
        log=EventLogger(tmp_path / "node.jsonl"),
        run_id="execution-test",
    )
    result = asyncio.run(node.run(ctx))
    assert isinstance(result, Orders)
    assert [(order.symbol, order.side, order.notional) for order in result.orders] == [
        ("AAA", "sell", 100.0),
        ("BBB", "sell", 200.0),
        ("CCC", "buy", 300.0),
    ]
    assert result.skipped_small == 1
    assert broker.submitted[0][0] == result.orders
    assert all(order.reference_price > 0 for order in result.orders)


def test_t8_broker_fills_next_open_and_cancels_missing_symbol(
    pipeline_fixture: tuple[PITBarStore, Path, tuple[str, ...]],
) -> None:
    store, _, symbols = pipeline_fixture
    ticks = daily_ticks(date(2022, 3, 1), date(2022, 3, 3))
    history = BarHistory.load(store, symbols, through=ticks[-1].at)
    broker = SimBroker(history, CostModel(), initial_cash=1000.0)
    broker.submit(
        [
            Order(symbol=symbols[0], side="buy", notional=100.0, reference_price=50.0),
            Order(symbol="MISSING", side="buy", notional=50.0, reference_price=50.0),
        ],
        placed_at=ticks[0].at,
    )
    assert broker.state(ticks[0].at).cash == 1000.0
    assert broker.fills == []

    next_state = broker.state(ticks[1].at)
    assert broker.fills
    first_fill = broker.fills[0]
    assert first_fill["session"] == ticks[1].at.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    assert first_fill["price"] == pytest.approx(
        history.visible(ticks[1].at, symbols=[symbols[0]]).sort("event_time").tail(1).item(0, "open")
    )
    assert next_state.cash >= 0.0
    assert any(item["symbol"] == "MISSING" for item in broker.cancelled)


def test_t9_runner_writes_daily_artifacts_and_respects_name_cap(
    pipeline_fixture: tuple[PITBarStore, Path, tuple[str, ...]], tmp_path: Path
) -> None:
    store, universe_path, symbols = pipeline_fixture
    result = asyncio.run(
        run_backtest(
            date(2022, 3, 1),
            date(2023, 12, 29),
            store=store,
            universe_path=universe_path,
            config_path=_CONFIG,
            out_dir=tmp_path / "run",
        )
    )
    assert isinstance(result, BacktestResult)
    expected_sessions = len(daily_ticks(date(2022, 3, 1), date(2023, 12, 29)))
    with result.equity_path.open(encoding="utf-8", newline="") as stream:
        equity_rows = list(csv.DictReader(stream))
    assert len(equity_rows) == expected_sessions
    assert result.fill_count > 0
    assert result.events_path.exists() and result.cancelled_path.exists() and result.fills_path.exists()
    assert result.baselines_path.exists() and result.contributions_path.exists()
    with result.baselines_path.open(encoding="utf-8", newline="") as stream:
        baseline_rows = list(csv.DictReader(stream))
    assert {row["strategy"] for row in baseline_rows} == {"equal_weight_buy_hold", "spy_buy_hold"}
    baseline_first_fill = min(row["date"] for row in baseline_rows if float(row["turnover"]) > 0)

    history = BarHistory.load(store, symbols, through=daily_ticks(date(2023, 12, 29), date(2023, 12, 29))[0].at)
    with result.fills_path.open(encoding="utf-8", newline="") as stream:
        fills = list(csv.DictReader(stream))
    grouped: dict[str, list[dict[str, str]]] = {}
    for fill in fills:
        grouped.setdefault(fill["session"], []).append(fill)
    assert min(grouped) == baseline_first_fill
    assert len(grouped) < expected_sessions // 5
    positions: dict[str, float] = {}
    cash = 1000.0
    for session, session_fills in sorted(grouped.items()):
        session_date = date.fromisoformat(session)
        tick = _decision_tick(session_date)
        bars = history.visible(tick.at).filter(
            pl.col("event_time").dt.convert_time_zone("America/New_York").dt.date() == session_date
        )
        open_prices = {row["symbol"]: float(row["open"]) for row in bars.iter_rows(named=True)}
        for fill in session_fills:
            symbol = fill["symbol"]
            shares = float(fill["shares"])
            price = float(fill["price"])
            cost = float(fill["cost"])
            if fill["side"] == "sell":
                cash += shares * price - cost
                positions[symbol] = positions.get(symbol, 0.0) - shares
            else:
                cash -= shares * price + cost
                positions[symbol] = positions.get(symbol, 0.0) + shares
        equity_at_open = cash + sum(shares * open_prices[symbol] for symbol, shares in positions.items() if shares > 0)
        for symbol, shares in positions.items():
            assert shares * open_prices[symbol] / equity_at_open <= 0.06


def test_equal_weight_baseline_excludes_names_without_first_fill_data(
    pipeline_fixture: tuple[PITBarStore, Path, tuple[str, ...]],
) -> None:
    store, _, symbols = pipeline_fixture
    ticks = daily_ticks(date(2023, 3, 1), date(2023, 3, 15))
    history = BarHistory.load(store, (*symbols, "SPY"), through=ticks[-1].at)
    rows = _buy_and_hold_baselines(
        history,
        ticks,
        (*symbols, "MISSING"),
        initial_cash=1000.0,
        cost_model=CostModel(),
    )
    first_equal = next(row for row in rows if row["strategy"] == "equal_weight_buy_hold")
    first_spy = next(row for row in rows if row["strategy"] == "spy_buy_hold")
    assert first_equal["constituent_count"] == len(symbols)
    assert first_equal["excluded_symbols"] == "MISSING"
    assert first_spy["constituent_count"] == 1
    assert first_spy["excluded_symbols"] == ""


def test_t10_replay_matches_all_node_output_digests(
    pipeline_fixture: tuple[PITBarStore, Path, tuple[str, ...]], tmp_path: Path
) -> None:
    store, universe_path, symbols = pipeline_fixture
    ticks = daily_ticks(date(2023, 12, 27), date(2023, 12, 29))
    first_history = BarHistory.load(store, symbols, through=ticks[-1].at)
    first_broker = SimBroker(first_history, CostModel())
    first_pipeline = _build_pipeline(first_broker)
    first_provider = SnapshotProvider(first_history, universe_path)

    def first_inputs(tick: Tick) -> Mapping[type[Message], Message]:
        state = first_broker.state(tick.at)
        portfolio = PortfolioState(
            tick=tick,
            producer="coordinator",
            cash=state.cash,
            positions=state.positions,
            market_values=state.market_values,
            equity=state.equity,
            peak_equity=state.peak_equity,
        )
        return {MarketSnapshot: first_provider.snapshot(tick), PortfolioState: portfolio}

    recorded_path = tmp_path / "recorded.jsonl"
    asyncio.run(
        first_pipeline.run_many(
            ticks,
            run_id="replay-source",
            log_path=recorded_path,
            inputs_for_tick=first_inputs,
        )
    )

    second_history = BarHistory.load(store, symbols, through=ticks[-1].at)
    second_broker = SimBroker(second_history, CostModel())
    second_pipeline = _build_pipeline(second_broker)
    second_provider = SnapshotProvider(second_history, universe_path)

    def replay_inputs(tick: Tick) -> Mapping[type[Message], Message]:
        state = second_broker.state(tick.at)
        portfolio = PortfolioState(
            tick=tick,
            producer="coordinator",
            cash=state.cash,
            positions=state.positions,
            market_values=state.market_values,
            equity=state.equity,
            peak_equity=state.peak_equity,
        )
        return {MarketSnapshot: second_provider.snapshot(tick), PortfolioState: portfolio}

    replayed = asyncio.run(
        second_pipeline.replay(
            ticks,
            recorded_log=recorded_path,
            recorded_run_id="replay-source",
            replay_log=tmp_path / "replayed.jsonl",
            inputs_for_tick=replay_inputs,
        )
    )
    assert len(replayed) == len(ticks)
    run_started = json.loads(recorded_path.read_text(encoding="utf-8").splitlines()[0])
    assert run_started["config"] == second_pipeline.config_specs


def test_t11_equal_score_run_tracks_independent_buy_and_hold(
    pipeline_fixture: tuple[PITBarStore, Path, tuple[str, ...]], tmp_path: Path
) -> None:
    store, universe_path, symbols = pipeline_fixture
    start, end = date(2023, 1, 3), date(2023, 2, 28)
    ticks = daily_ticks(start, end)
    history = BarHistory.load(store, symbols, through=ticks[-1].at)
    provider = SnapshotProvider(history, universe_path)
    broker = SimBroker(history, CostModel())

    class EqualAlpha(Node):
        name = "alpha"
        reads = (Features,)
        writes = Insights

        async def run(self, ctx: NodeContext) -> Insights:
            """Assign every available symbol the same positive score."""
            frame = ctx.inputs[Features].frame
            scores = {row["symbol"]: 1.0 for row in frame.iter_rows(named=True)}
            return Insights(tick=ctx.tick, producer=self.name, scores=scores, model_id="equal-test")

    from ws.pipeline_v0.nodes import FeatureBuilder, PortfolioConstruction, RiskManagement

    pipeline = Pipeline.build(
        FeatureBuilder(),
        EqualAlpha(),
        PortfolioConstruction(holdings=25),
        RiskManagement(
            max_name_weight=0.05,
            max_sector_weight=1.0,
            target_annual_volatility=0.15,
            drawdown_halt=0.15,
            max_adv_fraction=0.01,
        ),
        Execution(broker),
        inputs=(MarketSnapshot, PortfolioState),
    )

    def inputs_for_tick(tick: Tick) -> Mapping[type[Message], Message]:
        state = broker.state(tick.at)
        portfolio = PortfolioState(
            tick=tick,
            producer="coordinator",
            cash=state.cash,
            positions=state.positions,
            market_values=state.market_values,
            equity=state.equity,
            peak_equity=state.peak_equity,
        )
        return {MarketSnapshot: provider.snapshot(tick), PortfolioState: portfolio}

    asyncio.run(
        pipeline.run_many(
            ticks,
            run_id="equal-score",
            log_path=tmp_path / "equal.jsonl",
            inputs_for_tick=inputs_for_tick,
        )
    )
    assert broker.fills
    first_session = min(fill["session"] for fill in broker.fills)
    selected = symbols[:25]
    first_fill_tick = next(
        tick for tick in ticks if tick.at.astimezone(ZoneInfo("America/New_York")).date().isoformat() == first_session
    )
    first_bars = history.visible(first_fill_tick.at, symbols=selected)
    open_prices = {
        row["symbol"]: float(row["open"])
        for row in first_bars.iter_rows(named=True)
        if row["event_time"].astimezone(ZoneInfo("America/New_York")).date().isoformat() == first_session
    }
    assert set(open_prices) == set(selected)
    final_bars = history.visible(ticks[-1].at, symbols=selected)
    final_closes = {
        row["symbol"]: float(row["close"])
        for row in final_bars.iter_rows(named=True)
        if row["event_time"].astimezone(ZoneInfo("America/New_York")).date() == end
    }
    decision_bars = provider.snapshot(ticks[0]).bars
    adv = {row["symbol"]: row["adv_dollars_20d"] for row in adv_dollars_20d(decision_bars).iter_rows(named=True)}
    vol = {row["symbol"]: row["vol_21d"] for row in realized_vol_21d(decision_bars).iter_rows(named=True)}
    cost_model = CostModel()

    def buy_hold_required(scale: float) -> float:
        required = 0.0
        for symbol in selected:
            notional = 40.0 * scale
            required += notional + cost_model.dollars(
                order_notional=notional,
                adv_dollars=float(adv[symbol]),
                annual_volatility=float(vol[symbol]),
            )
        return required

    low, high = 0.0, 1.0
    for _ in range(64):
        middle = (low + high) / 2.0
        if buy_hold_required(middle) <= 1000.0:
            low = middle
        else:
            high = middle
    buy_hold_cash = 1000.0 - buy_hold_required(low)
    buy_hold_equity = buy_hold_cash + sum(
        (40.0 * low / open_prices[symbol]) * final_closes[symbol] for symbol in selected
    )
    simulated_final = broker.state(ticks[-1].at).equity
    assert abs(simulated_final / buy_hold_equity - 1.0) <= 0.02


def test_t12_snapshot_bars_are_available_by_each_tick(
    pipeline_fixture: tuple[PITBarStore, Path, tuple[str, ...]],
) -> None:
    store, universe_path, symbols = pipeline_fixture
    ticks = daily_ticks(date(2022, 3, 1), date(2022, 3, 15))
    history = BarHistory.load(store, symbols, through=ticks[-1].at)
    provider = SnapshotProvider(history, universe_path)
    for tick in ticks:
        snapshot = provider.snapshot(tick)
        assert all(available_at <= tick.at for available_at in snapshot.bars.get_column("available_at").to_list())


def test_t13_unknown_resource_fails_and_run_started_records_config(
    pipeline_fixture: tuple[PITBarStore, Path, tuple[str, ...]], tmp_path: Path
) -> None:
    from ws.core.pipeline import NodeSpec, build_from_config

    with pytest.raises(ValueError, match="unknown resource @absent"):
        build_from_config(
            [NodeSpec(cls="ws.pipeline_v0.nodes.Execution", kwargs={"broker": "@absent"})],
            resources={},
            inputs=(MarketSnapshot, PortfolioState),
        )

    store, universe_path, symbols = pipeline_fixture
    tick = _decision_tick(date(2023, 12, 29))
    history = BarHistory.load(store, symbols, through=tick.at)
    broker = SimBroker(history, CostModel())
    pipeline = _build_pipeline(broker)
    state = broker.state(tick.at)
    inputs = {
        MarketSnapshot: SnapshotProvider(history, universe_path).snapshot(tick),
        PortfolioState: PortfolioState(
            tick=tick,
            producer="coordinator",
            cash=state.cash,
            positions=state.positions,
            market_values=state.market_values,
            equity=state.equity,
            peak_equity=state.peak_equity,
        ),
    }
    log_path = tmp_path / "config-event.jsonl"
    asyncio.run(pipeline.run(tick, run_id="config-event", log_path=log_path, inputs=inputs))
    started = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert started["event"] == "run_started"
    assert started["config"] == [item.model_dump(mode="json") for item in _specs_as_models()]


def _specs_as_models() -> list[NodeSpec]:
    """Return the v0 config parsed into the exact frozen NodeSpec representation."""
    return [NodeSpec.model_validate(item) for item in _specs()]
