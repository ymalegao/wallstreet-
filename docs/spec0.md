# Pipeline v0 — Implementation Spec

Status: approved design, ready to implement.
Owner: Yash. Implementer: coding agent.

## 0. Rules for the implementer

Read this section first.

1. **Implement only what this spec says.** Do not add nodes, features, message fields, config options, dependencies or abstractions that are not listed here.
2. **If something is ambiguous or missing, stop and ask.** Do not pick a default. Write the question into `docs/pipeline-v0-questions.md` and continue only with the parts that are not blocked.
3. **Do not change the semantics of `src/ws/core/`** (clock, node, pipeline validation, event log, replay) except where Section 4 says so.
4. **Reuse existing code** where this spec names it. Do not rewrite it.
5. **No network access in tests.** Backtest tests use the synthetic fixture in Section 10; ingestion tests use a mocked HTTP transport.
6. **No new runtime dependencies.** In particular, do not add Qlib, LEAN, NautilusTrader or vectorbt (see Section 3).
7. Every new public function and class has a type-annotated signature and a one-line docstring.

## 1. Goal

A daily framework with two coordinator-provided typed inputs and a five-node decision graph that runs end to end over historical data:

universe and market data → features → momentum insights → portfolio targets → risk → orders → simulated fills

No LLM, no news, no trained model, no live broker. The predictor is 12-1 momentum, which needs no training. A LightGBM predictor replaces it later through configuration, without changing the graph.

## 2. Non-goals

- LightGBM or any trained model (next milestone)
- News, filings, LLM nodes
- Optional nodes: every v0 node is `required=True`
- Alpaca paper or live trading, any `AlpacaBroker`, live data feeds
- Parallel precompute of the stateless stage across ticks (later; v0 runs ticks sequentially with `run_many`)
- Any on-disk cache of intermediate frames
- Short selling, options, leverage

## 3. Prior art this design follows

The stage structure is taken from existing frameworks, not invented:

| Stage (QuantConnect LEAN Algorithm Framework) | LEAN output | v0 node(s) | v0 message |
| --- | --- | --- | --- |
| Universe Selection | assets to consider | coordinator `SnapshotProvider` | `MarketSnapshot` (includes universe) |
| Alpha | insights per asset | `features`, `alpha` | `Features`, `Insights` |
| Portfolio Construction | portfolio targets | `portfolio_construction` | `PortfolioTargets` |
| Risk Management | adjusted targets | `risk_management` | `RiskAdjustedTargets` |
| Execution | orders | `execution` | `Orders` |

- **From LEAN:** the five stages, their order, and their names. Each stage is swappable.
- **From Qlib:** components built from config as `{class, module, kwargs}` (Section 8), and training as a separate offline workflow on fixed date segments, never inside the daily graph.
- **From NautilusTrader and LEAN:** data is loaded once into memory and replayed forward in time; a component only sees data that has "arrived" by the current time (Section 6.1).

**Qlib is a reference, not a dependency, in v0.** Reasons:
- Qlib's daily data has no `available_at` column; it is keyed by date only. The point-in-time guarantee of this project comes from `available_at`.
- Its official dataset is disabled; US data must be converted into Qlib's own `.bin` format, a second copy of the data in a second format.
- Its latest PyPI release (0.9.7, August 2025) lists Linux x86-64 wheels; the target machine (GB10 Spark) is ARM, so it would need a source build.
- Its strategy and backtest components overlap with this pipeline's portfolio, risk and execution stages.

Qlib is used later in two narrow ways (next milestone, not this spec): its Alpha158 feature definitions are ported into the `features` node, and a Qlib `LGBModel` run on the same data serves as a cross-check of the LightGBM baseline, the same way vectorbt cross-checks the backtester.

## 4. Changes to `src/ws/core/`

1. **Source nodes and external inputs.** A node with `reads = ()` is a valid root and runs in the first level. A graph may also declare typed external messages with `build(inputs={...})`; the coordinator supplies them at each tick. Pin root-node behavior with T1.
2. **Cycle rejection.** Already implemented; add a test (T2) if one does not exist.
3. **Fast digest for DataFrames.** Messages may hold `polars.DataFrame` fields in memory. Today `jsonable` converts a frame with `to_dicts()`, which is slow on tens of thousands of rows. Change only the DataFrame branch of `jsonable` in `src/ws/core/messages.py`:
   - Canonicalize: select columns in name order, then sort rows by all columns in that order.
   - Serialize the canonical frame to Arrow IPC, uncompressed.
   - Return `{"frame_sha256": <sha256 hex of those bytes>, "rows": <height>, "columns": [<names in order>]}`.
   - Everything else in `jsonable`, `message_bytes` and `message_digest` stays as is.
   - Replay comparison keeps working because the digest still changes whenever frame contents change.
4. **Config-driven construction.** Add to `src/ws/core/pipeline.py`:

```python
class NodeSpec(BaseModel):  # frozen
    cls: str                       # dotted path, e.g. "ws.pipeline_v0.nodes.Alpha"
    kwargs: dict[str, Any] = {}

def build_from_config(specs: Sequence[NodeSpec], *, resources: Mapping[str, Any],
                      inputs: Iterable[type[Message]] = ()) -> Pipeline: ...
```

   - Imports each class, constructs it with `kwargs`, and calls `Pipeline.builder().add(...)` in order, then `build(inputs=...)`.
   - A kwarg value that is a string of the form `"@name"` is replaced by `resources[name]` (for objects that cannot be written in config, such as the broker or the data store). An unknown `@name` raises.
   - `Pipeline.run` accepts declared external messages for its tick. `run_many` and `replay` accept a per-tick input callback.
   - `Pipeline.run` logs the specs (not the resources) in the `run_started` event, under `config`. Store specs on the configured `Pipeline`; use `None` otherwise.

No other core changes.

## 5. Time and ticks

- **One decision tick per trading session**, from the XNYS calendar (`exchange_calendars`, already a dependency), so holidays and half days are handled.
- `tick.at = session_close + 30 minutes`, `tick.kind = "close"`, `tick.id = tick.at.isoformat()`.
- Bars become visible at `available_at = event_time (close) + 15 minutes` (existing `daily_bar_to_pit`), so the tick sees that day's close.
- **Orders placed at tick t fill at the next session's open.** This matches `simulate_daily`.
- Add `daily_ticks(start: date, end: date) -> list[Tick]` in a new `src/ws/schedule.py`.

## 6. Data access: preload once, slice per tick

The coordinator's `SnapshotProvider` and `SimBroker` need bars. Neither re-reads Parquet on every tick.

### 6.1 `BarHistory`

New class in `src/ws/store/history.py`:

```python
class BarHistory:
    """All bars for a run, held in memory and revealed in time order."""

    @classmethod
    def load(cls, store: PITBarStore, symbols: Sequence[str], through: datetime) -> BarHistory: ...
    def visible(self, now: datetime, *, symbols: Sequence[str] | None = None,
                lookback_sessions: int | None = None) -> pl.DataFrame: ...
```

- `load` reads the store once using a `LiveClock` frozen at `through` (the last tick of the run), and keeps the frame sorted by `available_at`, then `symbol`.
- `visible(now)` returns only rows with `available_at <= now`. If `lookback_sessions` is given, it further keeps the last `lookback_sessions` distinct `event_time` sessions among the visible rows.
- `visible` is the only read method. There is no method that returns rows regardless of `now`.
- `now` must be timezone-aware; raise otherwise.

Point-in-time safety now lives in `visible`: the full history sits in memory, but no caller can get a row from after `now`. T6 and T12 test this.

### 6.2 Universe

- The coordinator-side `SnapshotProvider` reads `symbols` and `as_of` from `configs/universe/sp100-current.json`.
- `universe_id = f"sp100-proxy@{as_of}"`.
- Known limitation, documented, not fixed in v0: this is today's membership applied to all dates (survivorship bias).

## 7. Messages

All live in `src/ws/pipeline_v0/messages.py`. All subclass `Message`. Weights are fractions of total equity. Symbols are uppercase.

| Message | Fields | Kind |
| --- | --- | --- |
| `MarketSnapshot` | `universe: tuple[str, ...]`, `universe_id: str`, `bars: pl.DataFrame` (PIT bar schema, last `lookback_sessions` sessions visible at the tick), `last_session: date` | stateless |
| `PortfolioState` | `cash: float`, `positions: dict[str, float]` (shares), `market_values: dict[str, float]` (at last visible close), `equity: float`, `peak_equity: float` | stateful |
| `Features` | `frame: pl.DataFrame` (one row per symbol: `symbol`, `mom_12_1`, `vol_21d`, `adv_dollars_20d`), `feature_set: str` | stateless |
| `Insights` | `scores: dict[str, float]` (no NaN; symbols without enough history are absent), `model_id: str` | stateless |
| `PortfolioTargets` | `weights: dict[str, float]`, `ranked_symbols: tuple[str, ...]` | stateful |
| `RiskAdjustedTargets` | `weights: dict[str, float]`, `gross_exposure: float`, `constraints: tuple[str, ...]` | stateful |
| `Orders` | `orders: tuple[Order, ...]`, `submitted: int`, `skipped_small: int` | stateful |

```python
class Order(BaseModel):
    model_config = ConfigDict(frozen=True)
    symbol: str
    side: Literal["buy", "sell"]
    notional: float        # dollars, > 0
    reference_price: float # last visible close used for sizing
```

## 8. Nodes and coordinator inputs

Computational nodes live in `src/ws/pipeline_v0/nodes.py`. They use `required=True`, the default `Policy`, and typed message inputs. The coordinator supplies `MarketSnapshot` and `PortfolioState` as external inputs; neither is a node output.

| # | Node class (`name`) | LEAN stage | Kind | Reads | Writes |
| --- | --- | --- | --- | --- | --- |
| 1 | `FeatureBuilder` (`features`) | Alpha | stateless | `MarketSnapshot` | `Features` |
| 2 | `Alpha` (`alpha`) | Alpha | stateless | `Features` | `Insights` |
| 3 | `PortfolioConstruction` (`portfolio_construction`) | Portfolio Construction | stateful | `Insights` | `PortfolioTargets` |
| 4 | `RiskManagement` (`risk_management`) | Risk Management | stateful | `PortfolioTargets`, `PortfolioState`, `MarketSnapshot` | `RiskAdjustedTargets` |
| 5 | `Execution` (`execution`) | Execution | stateful | `RiskAdjustedTargets`, `PortfolioState`, `MarketSnapshot` | `Orders` |

Expected node levels: `{features} → {alpha} → {portfolio_construction} → {risk_management} → {execution}`. `MarketSnapshot` and `PortfolioState` are already available at the first level as coordinator inputs. Test these levels (T3).

Shared helpers in `src/ws/pipeline_v0/metrics.py`, used by every node that needs them, so a formula exists once:
- `momentum_12_1(bars) -> pl.DataFrame`
- `realized_vol_21d(bars) -> pl.DataFrame`
- `adv_dollars_20d(bars) -> pl.DataFrame`
- `portfolio_vol_21d(bars, weights) -> float`

### 8.1 Coordinator input providers

- `SnapshotProvider(history, universe_path, lookback_sessions=300)` lives outside the DAG. It reads the dated universe once and returns `MarketSnapshot` for each tick using `history.visible(...)`.
- Raise if no bar is visible for the tick's own session.
- The coordinator calls `broker.state(as_of=tick.at)` and wraps it in `PortfolioState` before running the graph.
- `BarHistory` is read-only shared infrastructure. The broker is mutable shared infrastructure, injected only into `Execution`. Other nodes exchange typed messages rather than read global mutable state.

### 8.2 `FeatureBuilder`

- No constructor arguments in v0.
- Per symbol, using `close` ordered by `event_time`, with `s` = number of sessions back from the latest visible session:
  - `mom_12_1 = close[s=21] / close[s=252] - 1`
  - `vol_21d = std(daily log returns over the last 21 sessions) * sqrt(252)`
  - `adv_dollars_20d = mean(close * volume over the last 20 sessions)`
- A symbol with fewer than 253 sessions of history gets null `mom_12_1`. Same rule for the others with their own windows.
- `feature_set = "mom-v0"`.

### 8.3 `Alpha`

- Constructor: `model_id: str = "momentum-12-1"`. No other values are accepted in v0; raise on anything else.
- `scores = {symbol: mom_12_1}` for every symbol with a non-null `mom_12_1`.

### 8.4 `PortfolioConstruction`

- Wraps existing `ws.portfolio.constructor.PortfolioConstructor(holdings=25)`. Do not modify that class.
- The existing class keeps only positive scores. Keep that behavior: when momentum is negative across the board, there are fewer than 25 holdings and more cash.

### 8.5 `RiskManagement`

- Wraps existing `ws.portfolio.risk.RiskManager(RiskConfig(...))`. Do not modify that class.
- Inputs to `RiskManager.apply`:
  - `requested` = `PortfolioTargets.weights`
  - `sectors` = see Section 11, D1
  - `current_weights` = `market_values / equity` from `PortfolioState`
  - `drawdown = 1 - equity / peak_equity`
  - `equity` = `PortfolioState.equity`
  - `adv_dollars` = `adv_dollars_20d` on the snapshot bars. Risk does not read `Features`.
  - `annual_volatility` = `portfolio_vol_21d(snapshot bars, requested weights)`: annualized std of the last 21 daily returns of the requested portfolio.

### 8.6 `Execution`

- Constructor: `broker: Broker`, `min_order_dollars: float = 1.0`.
- Reads `RiskAdjustedTargets`, `PortfolioState`, and `MarketSnapshot`; it uses the latest visible close as the order's reference price.
- For each symbol in the approved weights or current positions:
  - `target_value = weight * equity` (0 if absent from approved weights)
  - `delta = target_value - current market_value`
  - `|delta| < min_order_dollars` → skip and count in `skipped_small`
  - otherwise create an `Order` with `side` from the sign and `notional = |delta|`
- Order the list: all sells first, then buys; within each side, by symbol.
- Call `broker.submit(orders, placed_at=ctx.clock.now())`.

## 9. Broker

`src/ws/broker/base.py` and `src/ws/broker/sim.py`.

```python
class BrokerState(BaseModel):  # frozen
    cash: float
    positions: dict[str, float]       # shares, fractional allowed
    market_values: dict[str, float]   # shares * last visible close
    equity: float
    peak_equity: float

class Broker(Protocol):
    def state(self, as_of: datetime) -> BrokerState: ...
    def submit(self, orders: Sequence[Order], placed_at: datetime) -> None: ...
```

`SimBroker(history: BarHistory, cost_model: CostModel, initial_cash: float = 1000.0)`:

- Uses the same `BarHistory` instance as `SnapshotProvider` and reads only through `history.visible(...)`.
- `submit` stores orders as pending. It does not fill them.
- `state(as_of)` first settles: every pending order placed before a session open that is `<= as_of` fills at that session's **open** price.
  - Sells fill before buys.
  - Cost: existing `CostModel.dollars(...)`, with `adv_dollars` and `annual_volatility` from the shared helpers on bars visible at fill time.
  - Sell: shares sold = notional / open (capped at held shares); cash += proceeds − cost.
  - Buy: if cash after costs would go negative, scale **all buys in that batch** down proportionally so cash ends at ≥ 0.
  - A symbol with no bar for that session: the order is cancelled and recorded; never filled at another price.
- Then marks positions to the latest visible close.
- `peak_equity` = max equity seen across all `state` calls.
- Keeps a fill log (session, symbol, side, shares, price, cost) as `sim_broker.fills`, and a cancelled-order log as `sim_broker.cancelled`.

## 10. Fixture, runner and config

**Fixture** (helper under `tests/fixtures/`, not a checked-in data file):

- 30 symbols, deterministic seed, XNYS sessions from 2021-01-04 to 2023-12-29.
- Geometric random walk closes with per-symbol drift so momentum ranks differ; open = previous close × small noise; volume positive.
- Written into a temporary `PITBarStore` via `daily_bar_to_pit`.
- A matching temporary universe JSON with those 30 symbols.

**Historical data ingestion** `src/ws/store/alpaca.py` downloads Alpaca `1Day` bars using `adjustment=all`, paginates the response, and writes through `daily_bar_to_pit`. Credentials come from `ALPACA_KEY` and `ALPACA_SECRET` in `.env` or the environment, with Alpaca's standard `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` environment names also supported. Network access is permitted for this explicit download command; tests use a mocked HTTP transport and never access the network.

**Default config** `configs/pipeline/v0.json` (JSON, so no new dependency). Contents, shown here in YAML form for readability:

```yaml
- cls: ws.pipeline_v0.nodes.FeatureBuilder
- cls: ws.pipeline_v0.nodes.Alpha
  kwargs: {model_id: momentum-12-1}
- cls: ws.pipeline_v0.nodes.PortfolioConstruction
  kwargs: {holdings: 25}
- cls: ws.pipeline_v0.nodes.RiskManagement
  kwargs: {max_name_weight: 0.05, max_sector_weight: 1.0, target_annual_volatility: 0.15, drawdown_halt: 0.15, max_adv_fraction: 0.01}
- cls: ws.pipeline_v0.nodes.Execution
  kwargs: {broker: "@broker", min_order_dollars: 1.0}
```

**Runner** `src/ws/pipeline_v0/run.py`:

```python
async def run_backtest(start: date, end: date, *, store: PITBarStore, universe_path: Path,
                       config_path: Path, out_dir: Path, initial_cash: float = 1000.0) -> BacktestResult: ...
```

- Builds ticks with `daily_ticks(start, end)`.
- Supports either the synthetic fixture or a populated historical PIT store. For 12-1 momentum, download at least 253 sessions before the first decision date.
- Loads one `BarHistory` through the last tick; creates one `SimBroker` on it.
- Builds a `SnapshotProvider`, `SimBroker`, and pipeline. The graph declares `MarketSnapshot` and `PortfolioState` as external inputs.
- The coordinator supplies those messages through `pipeline.run_many` at every tick.
- Writes to `out_dir`: `events.jsonl`, `fills.csv`, `cancelled.csv`, `equity.csv` (one row per tick: date, cash, equity, gross exposure).
- `BacktestResult`: `run_id`, the paths above, final equity, number of fills.

## 11. Tests (all must pass)

- **T1** A node with `reads=()` is placed in level 0.
- **T2** A two-node cycle is rejected by `build()`.
- **T3** The v0 graph builds from the default config, and its node levels are exactly {features}, {alpha}, {portfolio_construction}, {risk_management}, {execution}; `MarketSnapshot` and `PortfolioState` are coordinator inputs.
- **T4** Frame digest: the same data in a different row order or column order gives the same digest; changing one value changes it.
- **T5** Feature formulas: on a hand-built 260-session series, `mom_12_1`, `vol_21d`, `adv_dollars_20d` equal values computed by hand.
- **T6** Leakage: run one tick, then add bars with `available_at` after the tick to the store, reload `BarHistory`, and rerun the tick; every message digest is identical.
- **T7** Execution: given approved weights and holdings, produces the expected orders, sells before buys, small orders skipped.
- **T8** SimBroker: an order placed at tick t fills at session t+1's open, not before; cash never negative; a missing bar cancels the order.
- **T9** End to end: backtest over the fixture from 2022-03-01 to 2023-12-29 completes; `equity.csv` has one row per session; no position exceeds 6% of equity at fill time.
- **T10** Replay: rerun the same backtest with a fresh `SimBroker` and `BarHistory`; all node output digests match the first run (use `Pipeline.replay`).
- **T11** Sanity baseline: with every symbol given the same positive score (equal-weight top 25), the run completes and its final equity is within 2% of an independently computed equal-weight buy-and-hold of the same 25 symbols, after costs.
- **T12** `BarHistory.visible(now)` never returns a row with `available_at > now`, checked at every tick of the fixture run.
- **T13** `build_from_config` with an unknown `@resource` raises; the `run_started` event contains the config specs.
- **T14** The Alpaca downloader paginates a mocked response, requests `adjustment=all`, and writes normalized PIT bars without network access.

## 12. Decisions made in this spec (Yash: confirm or change)

- **D1 Sector cap.** No sector data exists in the repo. Until a sector map exists, `max_sector_weight = 1.0` so the cap is off, and the run logs `"sector_cap_disabled"` once. Do not use an empty `sectors` mapping with the default 0.30 cap: every symbol would fall into `UNKNOWN` and total exposure would be capped at 30%.
- **D2 Tick time** = session close + 30 minutes.
- **D3 Holdings** = 25, equal weight, positive scores only (existing constructor behavior).
- **D4 Volatility target** = 15% annualized, on the requested portfolio's 21-day realized volatility.
- **D5 Minimum order** = $1.
- **D6 Buys scaled down together** when cash is short, instead of dropping orders.
- **D7 Qlib** is not a dependency in v0 (Section 3).

## 13. Next milestone (not part of this spec)

1. Point-in-time vendor archives and historical universe membership; the Alpaca daily endpoint does not provide a historical first-seen archive.
2. Port Qlib's Alpha158 feature definitions into `FeatureBuilder` as `feature_set = "alpha158-v1"`.
3. Training job: (date, symbol) table, label, walk-forward date splits, LightGBM; registry entry; `Alpha(model_id="lgbm@v1")` through config. Cross-check with a Qlib `LGBModel` run on the same data.
4. Parallel precompute of the stateless stage for backtests.
