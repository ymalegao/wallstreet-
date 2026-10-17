# Trading Framework v0 — System Design

Oct 16, 2026 · @yash

## Overview

v0 is an engineering build: a typed, parallel DAG pipeline that turns market data and text into trades, with exact replay and identical code in backtest and paper trading. It applies existing tools (Qlib-style features, LightGBM, vLLM, Alpaca) rather than doing new research. Research comes after, on top of this foundation.

**Goals**

- One pipeline definition that runs in backtest and in Alpaca paper trading with no code changes above the Broker interface.
- Parallel execution wherever work is independent, for latency live and throughput in backtests.
- Every decision replayable from a human-readable, append-only event log.
- No node can read data from the future, enforced by the clock rather than by convention.
- LLM agents read text and call tools; deterministic code makes every trading decision.

**Non-goals for v0**

- Live capital.
- Options and other derivatives (see open questions).
- Research claims about profitability. v0 records what the research phase will need (trial counts, LLM cutoff dates, seeds) but does not compute Deflated Sharpe or PBO.
- Training or fine-tuning LLMs.

Items marked **(Q#)** are undecided and listed in Open questions at the end. Where a section shows a proposed default, it is a starting point to confirm, not a decision.

## Architecture

The pipeline is a DAG split into a stateless signal stage and a stateful trading tail. Signals depend only on data up to the tick, so a backtest computes them for every day in parallel and caches them; the portfolio depends on yesterday's positions, so it runs in order, but it is cheap arithmetic.

&#91;embedded content: v0 pipeline · stateless signal stage, stateful trading tail\]

Live, the graph runs once per tick: nodes run level by level, and every node in a level runs concurrently (asyncio for LLM and API calls, a process pool or the GPU for model work).

## Core interfaces

A node declares the typed messages it reads and the one type it writes; the builder derives the graph edges from those declarations. Any node may read any upstream output, but only by declaring it. There is no shared mutable state, which is what makes parallel execution and exact replay safe.

Language: Python with Pydantic models for messages is the proposed default **(Q1)**.

**Messages.** Immutable, typed, and stamped with the tick that produced them.

```python
class Message(BaseModel, frozen=True):
    tick: Tick                    # which tick produced this
    producer: str                 # node name
    producer_version: str         # code/model version

class NewsScores(Message):
    scores: dict[str, float]      # ticker -> score in [-1, 1]
    missing: set[str]             # tickers the scorer could not score
    model_id: str                 # includes the LLM's training cutoff
```

**Node.**

```python
class Node(ABC):
    name: str
    reads: tuple[type[Message], ...]   # declared inputs; edges come from these
    writes: type[Message]              # exactly one output type
    policy: Policy
    kind: Literal["stateless", "stateful"]

    @abstractmethod
    async def run(self, ctx: NodeContext) -> Message: ...

class NodeContext:
    clock: Clock                       # the only source of time
    inputs: Inputs                     # inputs.get(NewsScores) -> NewsScores | Missing
    tools: ToolRegistry
    log: EventLogger
```

**Policy.** Per node, set at build time.

```python
class Policy(BaseModel, frozen=True):
    timeout_s: float
    retries: int = 0
    backoff_s: float = 1.0
    required: bool = True     # required → failure halts trading for the tick
    executor: Literal["async", "process", "gpu"] = "async"
```

**Pipeline builder.** `build()` validates and freezes the graph; `run()` executes one tick; `backtest()` executes many.

```python
pipeline = (
    Pipeline.builder()
    .add(DataLoader())
    .add(FeatureBuilder())
    .add(LightGBMPredictor(model="lgbm@v3"))
    .add(NewsScorer(llm="local-llm@v1"))
    .add(SignalCombiner())
    .add(PortfolioConstructor())
    .add(RiskManager(limits=...))
    .add(ExecutionNode(broker=...))
    .build()      # cycle check, type check, one writer per message type
)

pipeline.run(tick)                      # live: one tick
pipeline.backtest(start, end, broker=SimBroker(...))
```

`build()` rejects: cycles, a read type that no node writes, two nodes writing the same type, and a stateless node reading a stateful node's output (that would break parallel precomputation).

## Clock and time model

The clock is the only source of time, and every data read goes through it, so no node can see a record whose `available_at` is after the current tick.

**Two timestamps on every record.**

- `event_time`: when the thing happened (the bar closed, the filing was accepted, the article was published).
- `available_at`: when this system could first have known it (vendor first-seen time, EDGAR acceptance time, bar finalization time plus any delivery lag).

All reads filter on `available_at <= clock.now()`. `event_time` is only used for ordering and labels.

```python
@dataclass(frozen=True)
class Tick:
    id: str               # e.g. "2026-10-16T16:00:00-04:00/close"
    at: datetime          # timezone-aware, exchange time
    kind: Literal["open", "close", "intraday", "event"]

class Clock(Protocol):
    def now(self) -> datetime: ...
    def tick(self) -> Tick: ...

class SimClock(Clock): ...    # advances through a list of historical ticks
class LiveClock(Clock): ...   # wall clock, aligned to the exchange calendar
```

**Enforcement.**

- The data store's query API takes the clock, never a raw timestamp. A read with `as_of > clock.now()` raises.
- Tools receive the clock from the context, so an LLM asking for a future date gets an error, not data.
- A leakage test shuffles every row with `available_at > t` and asserts all node outputs at `t` are unchanged.

**Tick schedule.** The exchange calendar (holidays, half days) drives ticks. Which tick the pipeline trades on (open, close, or intraday every N minutes) is open **(Q3, Q4)**.

## Data layer

Historical data lives in partitioned Parquet read lazily with Polars or DuckDB; live intraday data flows through per-symbol ring buffers. Daily history is small (about 2M rows for 500 stocks over 15 years), so it fits in memory and needs no special structure.

**Sources (proposed, see Q5, Q6).**

| Data | Source | `available_at` rule |
| --- | --- | --- |
| Daily and minute bars | Alpaca historical API | bar end + vendor delay |
| Corporate actions | Alpaca | announcement time |
| Filings (8-K, 10-Q, 10-K) | SEC EDGAR | EDGAR acceptance timestamp |
| News | Undecided (Q6) | vendor first-seen time |

**Layout.**

```
data/
  raw/<source>/<yyyy>/<mm>/...          # as ingested, never edited
  curated/bars_1d/year=2026/part-*.parquet
  curated/filings/year=2026/...
  curated/news/year=2026/...
  snapshots/<snapshot_id>/manifest.json   # content hashes of every file used
```

**Schemas (core columns).**

- Bars: `symbol, event_time, available_at, open, high, low, close, volume, vwap, adj_factor`
- Filings: `cik, symbol, form, event_time, available_at, url, text_ref`
- News: `id, symbols[], event_time, available_at, headline, body_ref, source`

**Snapshots.** A snapshot is an immutable set of files named by the hash of its manifest. Every run records the snapshot id it used, so a backtest from three months ago can be rerun on exactly the same bytes.

**Ring buffers (live only).** A fixed-size window per symbol, for example the last 390 one-minute bars, held as a NumPy array. It feeds rolling features without re-reading Parquet on every tick. Not needed for backtests.

**Batch/stream feature parity.** Each feature is written once as a function over a window and used by both the batch path (whole history, backtest) and the streaming path (ring buffer, live). A test asserts both paths produce identical values on the same data. Feature set: Qlib's Alpha158 list, ported.

## LLM agents

LLM agents turn text into typed, validated features; they never place or size trades. Every output passes three validation gates, and anything that fails becomes "missing" rather than reaching the deterministic layer.

**Two kinds of LLM node.**

| Kind | Input | Tools | Output | Runs on |
| --- | --- | --- | --- | --- |
| Scorer | One article or filing + ticker | None | Score, confidence, evidence span | Every new item |
| Analyst | A flagged case | Read-only tools (see Tool interface) | Structured findings + explanation | Flagged items only |

Scorers carry most of the value: reading text is what the LLM does better than LightGBM. An analyst that calls the predictor and restates its number adds latency and nothing else.

**Validation gates.**

1. Format: vLLM guided decoding constrains output to the message's JSON schema.
2. Range and logic: score in \[-1, 1\], ticker in the universe, confidence in \[0, 1\].
3. Evidence: the quoted evidence span must appear verbatim in the source text.

A failure at any gate is logged with the raw output and the reason, and the ticker is marked missing for that tick. LightGBM handles missing values natively.

**Memory (proposed, Q8).** Agents are stateless per tick. Working memory is an explicit input: a `MemoryFetch` node reads the agent's own recent outputs for that ticker (for example the last 10 scores and notes) from the event log and passes them in. Memory is then logged, replayable, and bounded by the clock. Compression, if needed later, is a deterministic summarizer node with its own version.

**Determinism.** Temperature 0, pinned model weights and prompt versions, seed recorded. The model id stored with every output includes the model's training cutoff date for the research phase.

**Serving on the GB10 Spark.**

- vLLM (or SGLang) with automatic prefix caching. Every scorer prompt shares a long prefix (system prompt, schema, the day's market context), so the per-ticker suffix is the only new prefill.
- Batch all of a tick's items into one request wave; the scorer node fans out with asyncio and awaits the batch.
- Decode speed on the Spark is bound by memory bandwidth, so mid-sized models score large batches much faster than the largest ones that fit in 128 GB. Model choice is open **(Q7)**.
- Dynamo is not needed on one box; it becomes relevant if serving moves to several workers. Prefix-sharing across agents is a natural place to plug in AgentCache later.
- The Spark is ARM; use NVIDIA's Spark containers for vLLM.

## Tool interface

Tools are pure, typed, read-only functions bound to the tick's clock, so the same tool behaves identically in backtest and live and cannot see the future. Agents get a fixed tool-call budget per tick, and every call is logged.

```python
class Tool(ABC, Generic[In, Out]):
    name: str
    description: str           # shown to the LLM
    input_model: type[In]      # Pydantic → JSON schema for the LLM
    output_model: type[Out]
    version: str

    @abstractmethod
    def __call__(self, args: In, clock: Clock) -> Out: ...

class ToolRegistry:
    def bind(self, clock: Clock, budget: int) -> BoundTools: ...
    def schemas(self) -> list[dict]: ...   # for the LLM's tool list
```

**v0 tool set.**

| Tool | Input | Output |
| --- | --- | --- |
| `get_features` | ticker, names\[\] | feature values as of now |
| `predict` | ticker | model score, rank, model version |
| `get_price_history` | ticker, lookback\_days | bars up to now |
| `get_filing` | ticker, form, n | filing excerpts up to now |
| `get_news` | ticker, lookback\_hours | headlines up to now |
| `check_risk` | proposed position | constraint results (read-only, no side effects) |

No tool can place orders. Order placement exists only in the execution node, behind the risk manager.

## Portfolio, risk and broker

The stateful tail of the graph is plain deterministic code: signals become target weights, the risk manager can veto or shrink them, and one Broker interface executes against either a simulated or a paper account.

**Portfolio constructor.** Ranks the combined signal and converts it to target weights with a turnover penalty. Whether v0 is long-only or long-short is open **(Q2)**. Long-short adds gross/net exposure limits and borrow costs to the risk manager.

**Single-name view.** Ask about one ticker and get a verdict (buy, hold, avoid; short once long-short is on) with its evidence: model score, its rank within the universe, LLM reads of recent news and filings, and which risk limits it would hit. It runs the same graph for the whole universe, because a score only means something relative to other stocks; the view then reports that one ticker.

**Risk manager.** Hard constraints with veto power; never an LLM. Limit values are open **(Q9)**.

- Max weight per name
- Max gross and net exposure
- Max sector exposure
- Volatility target
- Drawdown circuit breaker: flatten or freeze new trades past a threshold
- Liquidity cap: position size as a fraction of average daily volume

Output is the approved weights plus a list of every constraint that bound, so the log shows why a trade was cut.

**Broker interface.**

```python
class Broker(Protocol):
    def positions(self) -> dict[str, Position]: ...
    def cash(self) -> Decimal: ...
    def submit(self, orders: list[Order]) -> list[OrderAck]: ...
    def fills(self, since: datetime) -> list[Fill]: ...

class SimBroker(Broker): ...      # applies the cost model, fills at modeled prices
class AlpacaPaperBroker(Broker): ...
```

**Cost model (SimBroker).** Commission + half the bid-ask spread + a square-root market-impact term + borrow fees on shorts. The same cost model is used by the research harness later.

**Parity.** Everything above the Broker interface is identical in backtest and paper trading. A parity test runs the same week through SimBroker and replays Alpaca's recorded fills, then compares decisions.

**Out of scope for v0:** options **(Q11)** and live capital. Paper account size is open **(Q10)**; it matters for position sizing and, before any real-money intraday trading, for pattern-day-trader rules.

## Event log, reports and registry

The event log is append-only JSON Lines, one event per line, readable with `less` and queryable with DuckDB. Large payloads go to Parquet and are referenced by content hash, so the log stays small.

**Event schema.**

```json
{"ts": "2026-10-16T20:00:01.214Z", "tick": "2026-10-16T16:00:00-04:00/close",
 "run_id": "bt-2026-10-16-a1", "node": "NewsScorer", "node_version": "0.3.1",
 "event": "node_completed", "status": "ok", "duration_ms": 812,
 "inputs": {"NewsBatch": "sha256:9f2c..."}, "output": "sha256:41ab...",
 "model_id": "local-llm@v1", "notes": {"scored": 412, "rejected": 3}}
```

Event types: `run_started`, `tick_started`, `node_started`, `node_completed`, `node_failed`, `node_retried`, `llm_output_rejected`, `tool_called`, `risk_constraint_bound`, `order_submitted`, `fill_received`, `tick_completed`.

**Daily report.** Rendered from the log by a template, never written by an LLM, so every line in it is true by construction. It lists: positions and changes, which constraints bound, failed or skipped nodes, rejected LLM outputs with reasons, and P&L.

**Model registry.** Every trained model and prompt gets an id, a version, the training data snapshot id, its hyperparameters, and, for LLMs, the training cutoff date. Every prediction in the log carries its model id.

**Experiment tracking.** Every backtest config ever run is recorded with its seed, snapshot id, and results. The research phase needs this trial count for the Deflated Sharpe. The tool is open **(Q12)**; log storage location and retention are open **(Q13)**.

## Failure handling, scheduling and build plan

The rule is fail closed on required nodes and degrade on optional ones: if data, risk or the broker fails, the system places no new orders and keeps current positions; if an LLM node fails, the tick proceeds with those features missing.

**Failure behavior.**

| Failure | Behavior |
| --- | --- |
| Required node times out or errors after retries | No new orders this tick; keep positions; alert |
| Optional node times out or errors | Its output is `Missing`; downstream proceeds |
| LLM output fails a validation gate | That ticker marked missing; raw output logged |
| Tool budget exhausted | Agent returns what it has, flagged incomplete |
| Broker rejects an order | Logged; no automatic resubmit in v0 |
| Clock read from the future | Raises; treated as a bug, run halts |

**Scheduling.** A scheduler emits ticks from the exchange calendar. Ingestion jobs (news, filings) run continuously and only write to the data store; the tick graph reads whatever is available as of the tick. Where the scheduler and services run (on the Spark, always on, or elsewhere) is open **(Q15)**.

**Repo layout (proposed).**

```
core/        clock, messages, node, policy, pipeline builder, executor
data/        ingestion jobs, store, snapshots, schemas
features/    Alpha158 port, ring buffers, parity tests
models/      LightGBM training, registry
agents/      scorers, analysts, validation gates, prompts
tools/       tool registry and v0 tools
portfolio/   constructor, risk manager
broker/      SimBroker, AlpacaPaperBroker, cost model
log/         event logger, report templates
tests/       leakage, parity, replay, determinism
```

**Build order.**

1. Core: clock, messages, node, builder, async executor, event log. Exit: a toy three-node graph runs, logs, and replays byte-identically.
2. Data: Alpaca bars ingestion, Parquet store, snapshots, point-in-time queries. Exit: the future-shuffle leakage test passes.
3. Backtester and SimBroker with cost model. Exit: equal-weight buy-and-hold roughly tracks the index; a 12-1 momentum strategy behaves as published. The backtest engine choice is open **(Q14)**.
4. Features and LightGBM, walk-forward training, stateless signal precompute in parallel. Exit: out-of-sample IC reported; batch/stream parity test passes.
5. LLM scorer on the Spark with validation gates and prefix caching. Exit: scores logged as features; rejection rate visible in the daily report.
6. Risk manager and AlpacaPaperBroker. Exit: the same week produces identical decisions through SimBroker and paper replay.
7. Paper trading daily with the template report.

## Decisions

Eleven of the fifteen questions are decided; four are proposed defaults waiting for confirmation (Q7, Q9, Q12, Q14).

| # | Topic | Decision | Status |
| --- | --- | --- | --- |
| Q1 | Language | Python + Pydantic. Hot paths may move to C++ later behind the same interfaces, with Python calling in. | Decided |
| Q2 | Direction | Long-only in v0; long-short next. | Decided |
| Q3 | Frequency | Daily first, intraday after. | Decided |
| Q4 | Rebalance tick | Both open and close. | Decided |
| Q5 | Universe | A fixed list of selected stocks, plus a single-name view (see Portfolio). | Decided; list to pick |
| Q6 | Data | Free sources until the pipeline works; pay after. | Decided; free news source to pick |
| Q7 | Scorer LLM | An existing local Qwen or Gemma model; benchmark two sizes on the scoring task. | Proposed |
| Q8 | Agent memory | Stateless agents with retrieved memory as logged input; layered memory (FinMem-style) later as an experiment. | Decided |
| Q9 | Risk limits | See proposed limits below. | Proposed |
| Q10 | Paper capital | $1,000. | Decided |
| Q11 | Options | Planned for later; not in v0. | Decided |
| Q12 | Experiment tracking | MLflow, self-hosted locally. | Proposed |
| Q13 | Log storage | Local. | Decided |
| Q14 | Backtest engine | Own vectorized engine, with vectorbt as a cross-check in tests; revisit NautilusTrader for intraday. | Proposed |
| Q15 | Deployment | Local on the Spark, always on. | Decided |

**Proposed risk limits (Q9), for $1,000 long-only.**

- Max 10% per name, so about 10 positions
- No leverage: gross exposure at most 100%, cash never negative
- Max 30% in one sector
- Drawdown breaker: no new buys after a 15% fall from peak equity
- Liquidity cap: 1% of 20-day average volume (never binds at $1,000; matters as capital grows)
- Volatility target: off in v0

**Constraints from $1,000.** Alpaca's fractional shares make 10 positions workable. Shorting needs a margin account, which generally requires at least $2,000 of equity, so long-short at this size works in paper trading only.

**Still open.**

Q2 Long-only or long-short for v0?

- [ ] Pick the fixed stock list (Q5)
- [ ] Pick a free news source (Q6)
- [ ] Confirm or change the proposed defaults: Q7, Q9, Q12, Q14
# Trading Framework v0 — System Design

Oct 16, 2026 · @yash

## Overview

v0 is an engineering build: a typed, parallel DAG pipeline that turns market data and text into trades, with exact replay and identical code in backtest and paper trading. It applies existing tools (Qlib-style features, LightGBM, vLLM, Alpaca) rather than doing new research. Research comes after, on top of this foundation.

**Goals**

- One pipeline definition that runs in backtest and in Alpaca paper trading with no code changes above the Broker interface.
- Parallel execution wherever work is independent, for latency live and throughput in backtests.
- Every decision replayable from a human-readable, append-only event log.
- No node can read data from the future, enforced by the clock rather than by convention.
- LLM agents read text and call tools; deterministic code makes every trading decision.

**Non-goals for v0**

- Live capital.
- Options and other derivatives (see open questions).
- Research claims about profitability. v0 records what the research phase will need (trial counts, LLM cutoff dates, seeds) but does not compute Deflated Sharpe or PBO.
- Training or fine-tuning LLMs.

Items marked **(Q#)** are undecided and listed in Open questions at the end. Where a section shows a proposed default, it is a starting point to confirm, not a decision.

## Architecture

The pipeline is a DAG split into a stateless signal stage and a stateful trading tail. Signals depend only on data up to the tick, so a backtest computes them for every day in parallel and caches them; the portfolio depends on yesterday's positions, so it runs in order, but it is cheap arithmetic.

&#91;embedded content: v0 pipeline · stateless signal stage, stateful trading tail\]

Live, the graph runs once per tick: nodes run level by level, and every node in a level runs concurrently (asyncio for LLM and API calls, a process pool or the GPU for model work).

## Core interfaces

A node declares the typed messages it reads and the one type it writes; the builder derives the graph edges from those declarations. Any node may read any upstream output, but only by declaring it. There is no shared mutable state, which is what makes parallel execution and exact replay safe.

Language: Python with Pydantic models for messages is the proposed default **(Q1)**.

**Messages.** Immutable, typed, and stamped with the tick that produced them.

```python
class Message(BaseModel, frozen=True):
    tick: Tick                    # which tick produced this
    producer: str                 # node name
    producer_version: str         # code/model version

class NewsScores(Message):
    scores: dict[str, float]      # ticker -> score in [-1, 1]
    missing: set[str]             # tickers the scorer could not score
    model_id: str                 # includes the LLM's training cutoff
```

**Node.**

```python
class Node(ABC):
    name: str
    reads: tuple[type[Message], ...]   # declared inputs; edges come from these
    writes: type[Message]              # exactly one output type
    policy: Policy
    kind: Literal["stateless", "stateful"]

    @abstractmethod
    async def run(self, ctx: NodeContext) -> Message: ...

class NodeContext:
    clock: Clock                       # the only source of time
    inputs: Inputs                     # inputs.get(NewsScores) -> NewsScores | Missing
    tools: ToolRegistry
    log: EventLogger
```

**Policy.** Per node, set at build time.

```python
class Policy(BaseModel, frozen=True):
    timeout_s: float
    retries: int = 0
    backoff_s: float = 1.0
    required: bool = True     # required → failure halts trading for the tick
    executor: Literal["async", "process", "gpu"] = "async"
```

**Pipeline builder.** `build()` validates and freezes the graph; `run()` executes one tick; `backtest()` executes many.

```python
pipeline = (
    Pipeline.builder()
    .add(DataLoader())
    .add(FeatureBuilder())
    .add(LightGBMPredictor(model="lgbm@v3"))
    .add(NewsScorer(llm="local-llm@v1"))
    .add(SignalCombiner())
    .add(PortfolioConstructor())
    .add(RiskManager(limits=...))
    .add(ExecutionNode(broker=...))
    .build()      # cycle check, type check, one writer per message type
)

pipeline.run(tick)                      # live: one tick
pipeline.backtest(start, end, broker=SimBroker(...))
```

`build()` rejects: cycles, a read type that no node writes, two nodes writing the same type, and a stateless node reading a stateful node's output (that would break parallel precomputation).

## Clock and time model

The clock is the only source of time, and every data read goes through it, so no node can see a record whose `available_at` is after the current tick.

**Two timestamps on every record.**

- `event_time`: when the thing happened (the bar closed, the filing was accepted, the article was published).
- `available_at`: when this system could first have known it (vendor first-seen time, EDGAR acceptance time, bar finalization time plus any delivery lag).

All reads filter on `available_at <= clock.now()`. `event_time` is only used for ordering and labels.

```python
@dataclass(frozen=True)
class Tick:
    id: str               # e.g. "2026-10-16T16:00:00-04:00/close"
    at: datetime          # timezone-aware, exchange time
    kind: Literal["open", "close", "intraday", "event"]

class Clock(Protocol):
    def now(self) -> datetime: ...
    def tick(self) -> Tick: ...

class SimClock(Clock): ...    # advances through a list of historical ticks
class LiveClock(Clock): ...   # wall clock, aligned to the exchange calendar
```

**Enforcement.**

- The data store's query API takes the clock, never a raw timestamp. A read with `as_of > clock.now()` raises.
- Tools receive the clock from the context, so an LLM asking for a future date gets an error, not data.
- A leakage test shuffles every row with `available_at > t` and asserts all node outputs at `t` are unchanged.

**Tick schedule.** The exchange calendar (holidays, half days) drives ticks. Which tick the pipeline trades on (open, close, or intraday every N minutes) is open **(Q3, Q4)**.

## Data layer

Historical data lives in partitioned Parquet read lazily with Polars or DuckDB; live intraday data flows through per-symbol ring buffers. Daily history is small (about 2M rows for 500 stocks over 15 years), so it fits in memory and needs no special structure.

**Sources (proposed, see Q5, Q6).**

| Data | Source | `available_at` rule |
| --- | --- | --- |
| Daily and minute bars | Alpaca historical API | bar end + vendor delay |
| Corporate actions | Alpaca | announcement time |
| Filings (8-K, 10-Q, 10-K) | SEC EDGAR | EDGAR acceptance timestamp |
| News | Undecided (Q6) | vendor first-seen time |

**Layout.**

```
data/
  raw/<source>/<yyyy>/<mm>/...          # as ingested, never edited
  curated/bars_1d/year=2026/part-*.parquet
  curated/filings/year=2026/...
  curated/news/year=2026/...
  snapshots/<snapshot_id>/manifest.json   # content hashes of every file used
```

**Schemas (core columns).**

- Bars: `symbol, event_time, available_at, open, high, low, close, volume, vwap, adj_factor`
- Filings: `cik, symbol, form, event_time, available_at, url, text_ref`
- News: `id, symbols[], event_time, available_at, headline, body_ref, source`

**Snapshots.** A snapshot is an immutable set of files named by the hash of its manifest. Every run records the snapshot id it used, so a backtest from three months ago can be rerun on exactly the same bytes.

**Ring buffers (live only).** A fixed-size window per symbol, for example the last 390 one-minute bars, held as a NumPy array. It feeds rolling features without re-reading Parquet on every tick. Not needed for backtests.

**Batch/stream feature parity.** Each feature is written once as a function over a window and used by both the batch path (whole history, backtest) and the streaming path (ring buffer, live). A test asserts both paths produce identical values on the same data. Feature set: Qlib's Alpha158 list, ported.

## LLM agents

LLM agents turn text into typed, validated features; they never place or size trades. Every output passes three validation gates, and anything that fails becomes "missing" rather than reaching the deterministic layer.

**Two kinds of LLM node.**

| Kind | Input | Tools | Output | Runs on |
| --- | --- | --- | --- | --- |
| Scorer | One article or filing + ticker | None | Score, confidence, evidence span | Every new item |
| Analyst | A flagged case | Read-only tools (see Tool interface) | Structured findings + explanation | Flagged items only |

Scorers carry most of the value: reading text is what the LLM does better than LightGBM. An analyst that calls the predictor and restates its number adds latency and nothing else.

**Validation gates.**

1. Format: vLLM guided decoding constrains output to the message's JSON schema.
2. Range and logic: score in \[-1, 1\], ticker in the universe, confidence in \[0, 1\].
3. Evidence: the quoted evidence span must appear verbatim in the source text.

A failure at any gate is logged with the raw output and the reason, and the ticker is marked missing for that tick. LightGBM handles missing values natively.

**Memory (proposed, Q8).** Agents are stateless per tick. Working memory is an explicit input: a `MemoryFetch` node reads the agent's own recent outputs for that ticker (for example the last 10 scores and notes) from the event log and passes them in. Memory is then logged, replayable, and bounded by the clock. Compression, if needed later, is a deterministic summarizer node with its own version.

**Determinism.** Temperature 0, pinned model weights and prompt versions, seed recorded. The model id stored with every output includes the model's training cutoff date for the research phase.

**Serving on the GB10 Spark.**

- vLLM (or SGLang) with automatic prefix caching. Every scorer prompt shares a long prefix (system prompt, schema, the day's market context), so the per-ticker suffix is the only new prefill.
- Batch all of a tick's items into one request wave; the scorer node fans out with asyncio and awaits the batch.
- Decode speed on the Spark is bound by memory bandwidth, so mid-sized models score large batches much faster than the largest ones that fit in 128 GB. Model choice is open **(Q7)**.
- Dynamo is not needed on one box; it becomes relevant if serving moves to several workers. Prefix-sharing across agents is a natural place to plug in AgentCache later.
- The Spark is ARM; use NVIDIA's Spark containers for vLLM.

## Tool interface

Tools are pure, typed, read-only functions bound to the tick's clock, so the same tool behaves identically in backtest and live and cannot see the future. Agents get a fixed tool-call budget per tick, and every call is logged.

```python
class Tool(ABC, Generic[In, Out]):
    name: str
    description: str           # shown to the LLM
    input_model: type[In]      # Pydantic → JSON schema for the LLM
    output_model: type[Out]
    version: str

    @abstractmethod
    def __call__(self, args: In, clock: Clock) -> Out: ...

class ToolRegistry:
    def bind(self, clock: Clock, budget: int) -> BoundTools: ...
    def schemas(self) -> list[dict]: ...   # for the LLM's tool list
```

**v0 tool set.**

| Tool | Input | Output |
| --- | --- | --- |
| `get_features` | ticker, names\[\] | feature values as of now |
| `predict` | ticker | model score, rank, model version |
| `get_price_history` | ticker, lookback\_days | bars up to now |
| `get_filing` | ticker, form, n | filing excerpts up to now |
| `get_news` | ticker, lookback\_hours | headlines up to now |
| `check_risk` | proposed position | constraint results (read-only, no side effects) |

No tool can place orders. Order placement exists only in the execution node, behind the risk manager.

## Portfolio, risk and broker

The stateful tail of the graph is plain deterministic code: signals become target weights, the risk manager can veto or shrink them, and one Broker interface executes against either a simulated or a paper account.

**Portfolio constructor.** Ranks the combined signal and converts it to target weights with a turnover penalty. Whether v0 is long-only or long-short is open **(Q2)**. Long-short adds gross/net exposure limits and borrow costs to the risk manager.

**Single-name view.** Ask about one ticker and get a verdict (buy, hold, avoid; short once long-short is on) with its evidence: model score, its rank within the universe, LLM reads of recent news and filings, and which risk limits it would hit. It runs the same graph for the whole universe, because a score only means something relative to other stocks; the view then reports that one ticker.

**Risk manager.** Hard constraints with veto power; never an LLM. Limit values are open **(Q9)**.

- Max weight per name
- Max gross and net exposure
- Max sector exposure
- Volatility target
- Drawdown circuit breaker: flatten or freeze new trades past a threshold
- Liquidity cap: position size as a fraction of average daily volume

Output is the approved weights plus a list of every constraint that bound, so the log shows why a trade was cut.

**Broker interface.**

```python
class Broker(Protocol):
    def positions(self) -> dict[str, Position]: ...
    def cash(self) -> Decimal: ...
    def submit(self, orders: list[Order]) -> list[OrderAck]: ...
    def fills(self, since: datetime) -> list[Fill]: ...

class SimBroker(Broker): ...      # applies the cost model, fills at modeled prices
class AlpacaPaperBroker(Broker): ...
```

**Cost model (SimBroker).** Commission + half the bid-ask spread + a square-root market-impact term + borrow fees on shorts. The same cost model is used by the research harness later.

**Parity.** Everything above the Broker interface is identical in backtest and paper trading. A parity test runs the same week through SimBroker and replays Alpaca's recorded fills, then compares decisions.

**Out of scope for v0:** options **(Q11)** and live capital. Paper account size is open **(Q10)**; it matters for position sizing and, before any real-money intraday trading, for pattern-day-trader rules.

## Event log, reports and registry

The event log is append-only JSON Lines, one event per line, readable with `less` and queryable with DuckDB. Large payloads go to Parquet and are referenced by content hash, so the log stays small.

**Event schema.**

```json
{"ts": "2026-10-16T20:00:01.214Z", "tick": "2026-10-16T16:00:00-04:00/close",
 "run_id": "bt-2026-10-16-a1", "node": "NewsScorer", "node_version": "0.3.1",
 "event": "node_completed", "status": "ok", "duration_ms": 812,
 "inputs": {"NewsBatch": "sha256:9f2c..."}, "output": "sha256:41ab...",
 "model_id": "local-llm@v1", "notes": {"scored": 412, "rejected": 3}}
```

Event types: `run_started`, `tick_started`, `node_started`, `node_completed`, `node_failed`, `node_retried`, `llm_output_rejected`, `tool_called`, `risk_constraint_bound`, `order_submitted`, `fill_received`, `tick_completed`.

**Daily report.** Rendered from the log by a template, never written by an LLM, so every line in it is true by construction. It lists: positions and changes, which constraints bound, failed or skipped nodes, rejected LLM outputs with reasons, and P&L.

**Model registry.** Every trained model and prompt gets an id, a version, the training data snapshot id, its hyperparameters, and, for LLMs, the training cutoff date. Every prediction in the log carries its model id.

**Experiment tracking.** Every backtest config ever run is recorded with its seed, snapshot id, and results. The research phase needs this trial count for the Deflated Sharpe. The tool is open **(Q12)**; log storage location and retention are open **(Q13)**.

## Failure handling, scheduling and build plan

The rule is fail closed on required nodes and degrade on optional ones: if data, risk or the broker fails, the system places no new orders and keeps current positions; if an LLM node fails, the tick proceeds with those features missing.

**Failure behavior.**

| Failure | Behavior |
| --- | --- |
| Required node times out or errors after retries | No new orders this tick; keep positions; alert |
| Optional node times out or errors | Its output is `Missing`; downstream proceeds |
| LLM output fails a validation gate | That ticker marked missing; raw output logged |
| Tool budget exhausted | Agent returns what it has, flagged incomplete |
| Broker rejects an order | Logged; no automatic resubmit in v0 |
| Clock read from the future | Raises; treated as a bug, run halts |

**Scheduling.** A scheduler emits ticks from the exchange calendar. Ingestion jobs (news, filings) run continuously and only write to the data store; the tick graph reads whatever is available as of the tick. Where the scheduler and services run (on the Spark, always on, or elsewhere) is open **(Q15)**.

**Repo layout (proposed).**

```
core/        clock, messages, node, policy, pipeline builder, executor
data/        ingestion jobs, store, snapshots, schemas
features/    Alpha158 port, ring buffers, parity tests
models/      LightGBM training, registry
agents/      scorers, analysts, validation gates, prompts
tools/       tool registry and v0 tools
portfolio/   constructor, risk manager
broker/      SimBroker, AlpacaPaperBroker, cost model
log/         event logger, report templates
tests/       leakage, parity, replay, determinism
```

**Build order.**

1. Core: clock, messages, node, builder, async executor, event log. Exit: a toy three-node graph runs, logs, and replays byte-identically.
2. Data: Alpaca bars ingestion, Parquet store, snapshots, point-in-time queries. Exit: the future-shuffle leakage test passes.
3. Backtester and SimBroker with cost model. Exit: equal-weight buy-and-hold roughly tracks the index; a 12-1 momentum strategy behaves as published. The backtest engine choice is open **(Q14)**.
4. Features and LightGBM, walk-forward training, stateless signal precompute in parallel. Exit: out-of-sample IC reported; batch/stream parity test passes.
5. LLM scorer on the Spark with validation gates and prefix caching. Exit: scores logged as features; rejection rate visible in the daily report.
6. Risk manager and AlpacaPaperBroker. Exit: the same week produces identical decisions through SimBroker and paper replay.
7. Paper trading daily with the template report.

## Decisions

Eleven of the fifteen questions are decided; four are proposed defaults waiting for confirmation (Q7, Q9, Q12, Q14).

| # | Topic | Decision | Status |
| --- | --- | --- | --- |
| Q1 | Language | Python + Pydantic. Hot paths may move to C++ later behind the same interfaces, with Python calling in. | Decided |
| Q2 | Direction | Long-only in v0; long-short next. | Decided |
| Q3 | Frequency | Daily first, intraday after. | Decided |
| Q4 | Rebalance tick | Both open and close. | Decided |
| Q5 | Universe | A fixed list of selected stocks, plus a single-name view (see Portfolio). | Decided; list to pick |
| Q6 | Data | Free sources until the pipeline works; pay after. | Decided; free news source to pick |
| Q7 | Scorer LLM | An existing local Qwen or Gemma model; benchmark two sizes on the scoring task. | Proposed |
| Q8 | Agent memory | Stateless agents with retrieved memory as logged input; layered memory (FinMem-style) later as an experiment. | Decided |
| Q9 | Risk limits | See proposed limits below. | Proposed |
| Q10 | Paper capital | $1,000. | Decided |
| Q11 | Options | Planned for later; not in v0. | Decided |
| Q12 | Experiment tracking | MLflow, self-hosted locally. | Proposed |
| Q13 | Log storage | Local. | Decided |
| Q14 | Backtest engine | Own vectorized engine, with vectorbt as a cross-check in tests; revisit NautilusTrader for intraday. | Proposed |
| Q15 | Deployment | Local on the Spark, always on. | Decided |

**Proposed risk limits (Q9), for $1,000 long-only.**

- Max 10% per name, so about 10 positions
- No leverage: gross exposure at most 100%, cash never negative
- Max 30% in one sector
- Drawdown breaker: no new buys after a 15% fall from peak equity
- Liquidity cap: 1% of 20-day average volume (never binds at $1,000; matters as capital grows)
- Volatility target: off in v0

**Constraints from $1,000.** Alpaca's fractional shares make 10 positions workable. Shorting needs a margin account, which generally requires at least $2,000 of equity, so long-short at this size works in paper trading only.

**Still open.**

Q2 Long-only or long-short for v0?

- [ ] Pick the fixed stock list (Q5)
- [ ] Pick a free news source (Q6)
- [ ] Confirm or change the proposed defaults: Q7, Q9, Q12, Q14
