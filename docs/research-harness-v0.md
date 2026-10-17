# Synthetic training demo notes

This note describes the offline synthetic-data demo only. The current runtime and data-layer status, including limitations and next steps, lives in [`harness-v0-status.md`](harness-v0-status.md). The demo is not connected to a broker and makes no claim about returns.

## Run the offline example

From the repository root:

```bash
uv sync --group research
uv run --group research python -m ws.research.demo
```

The example creates deterministic synthetic daily bars, including rows dated after its fixed decision cutoff. It filters those future rows, builds lagged return and volatility features, trains a CPU `HistGradientBoostingRegressor` using only labels available by 2022-12-31, and evaluates the later period through 2023-06-30. No data credentials, network data, GPU, or broker account are used. The data is synthetic, so the metrics are only a mechanics demonstration.

Generated files go under ignored `artifacts/research_demo/`:

- `synthetic_bars.csv`: input panel, regenerated deterministically on each run.
- `<run-id>.events.jsonl`: append-only node start/completion records and output hashes.
- `run_card.json`: data hash and cutoff, train/evaluation boundaries, model parameters, sample counts, and simple forecast metrics.

## Node model

Implement a node by subclassing `ws.core.Node`, declaring `reads` and `writes` as message classes, and implementing `async run(ctx)`. Read declared inputs from `ctx.inputs`; its accessor preserves the specific message subclass. Or compose node instances with `Pipeline.builder().add(...).build()`.

```python
class Score(Message):
    value: float

class ScaledScore(Message):
    value: float

class ScaleScore(Node):
    name = "scale_score"
    reads = (Score,)
    writes = ScaledScore

    async def run(self, ctx):
        score = ctx.inputs[Score]  # statically known as Score
        return ScaledScore(tick=ctx.tick, producer=self.name, value=2 * score.value)
```

The builder derives edges from types and rejects duplicate writers, unknown input types, stateful-to-stateless dependencies, and cycles. Independent nodes at one graph level execute concurrently, and a required failure cancels sibling work in that level. Every emitted message carries tick and producer metadata; the runner checks both before accepting it.

The demo graph is:

```text
LoadBars → BuildFeatures → TrainBaseline ─┐
                    └─────────────────────┴→ EvaluateBaseline
```

The core includes the pieces needed to demonstrate node composition, DAG validation, asynchronous execution, policies, a simulation clock, and a JSONL event log with output hashes for replay. CPU model fitting/prediction is sent to a worker thread so it does not block other async nodes. The separate `PITBarStore` supplies clock-filtered daily-bar Parquet reads; portfolio state, risk controls, costs, and broker interfaces remain later work.

## Point-in-time behavior in the example

- Each raw bar has `session` and `available_at` fields.
- `CsvBarSource` only returns rows where `available_at <= Tick.at` and records how many later rows it withheld.
- `BuildFeatures` uses trailing returns/volatility for the input features. The next-bar return is a label, and the label is retained only once its next bar's `available_at` is no later than the run tick.
- `TrainBaseline` has a separate historical cutoff and only fits labels available by that cutoff. The evaluation period starts after it.

This is a clear toy example, not a profitability result or a replacement for the source-probe requirements in `docs/02-design-spec.md`.

## Deliberate v0 choices and next decisions

- **Node interface:** Python classes, Pydantic messages, and type-declared dependencies, following `PLASEFOLLOWTHISDESIGN.md`.
- **Baseline:** scikit-learn histogram gradient boosting on CPU for portability. The model API can later be wrapped by LightGBM/XGBoost and a device-aware trainer after the real panel and hardware are selected.
- **Data:** this demo generates synthetic data only. The new PIT bar slice uses a dated current-universe proxy; it is survivorship-biased and is not suitable for historical alpha claims.
- **Evaluation:** a single chronological split and basic RMSE/correlation only. Research claims still require walk-forward purging/embargo, costs, bootstrap confidence intervals, trial counting/DSR, PBO, and regime/ablation reporting from the larger research design.
- **Graph:** currently one message type has one writer. If multiple model candidates need to produce the same semantic output, wrap them in distinct message types or add named output ports deliberately; do not introduce implicit shared state.
- **No GPU claim:** this example is intentionally too small for GPU acceleration. The GPU belongs in batched model training/inference or large parallel evaluation once workload measurements justify it.
