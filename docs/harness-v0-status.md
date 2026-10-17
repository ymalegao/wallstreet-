# Current status

This branch contains a small research harness scaffold. The old prototype was kept on `research-harness-v0-checkpoint`; this branch is intended to be the easy place to understand and continue the new design.

## What is here

- **Node runtime** (`src/ws/core/`): immutable messages, a simulation clock, a node contract, DAG validation, concurrent execution by graph level, explicit optional-node failures, and JSONL event logging with replay hashes.
- **Point-in-time bars** (`src/ws/store/pit_bars.py`): Parquet storage with `event_time` and `available_at`; the read method takes a clock so callers cannot ask for later data.
- **Training example** (`src/ws/research/demo.py`): synthetic bars → lagged features → scikit-learn gradient-boosting baseline → evaluation and run card. It is CPU-only and intentionally makes no market-performance claim.
- **Trading mechanics**: deterministic allocation and risk components, a transaction-cost function, and a daily-bar simulator. Risk constraints are not yet wired into that simulator.
- **Data conversion**: the universe snapshot and PIT conversion scripts are present. The raw Alpaca bar downloader from the previous prototype was removed, so conversion requires an existing raw-bar cache.

## Try the model example

```bash
uv sync --group dev --group research
uv run python -m ws.research.demo
```

The demo writes generated inputs, the event log, and a run card to ignored `artifacts/research_demo/`.

## Data limits

The checked-in universe file is a dated iShares OEF holdings proxy, not an official historical S&P 100 constituent archive. Applying today's members to older dates creates survivorship bias. The local raw bars and snapshots are ignored by Git and are not included in this branch. Bars are unadjusted; corporate actions and reliable historical `available_at` records are not implemented. Do not use this slice to claim backtest alpha.

## Next useful steps

1. Agree on the first data source and universe, then add a small reproducible data fetcher with source timestamps.
2. Add point-in-time corporate actions and historical membership before making long-horizon claims.
3. Train a chronological baseline on real data and compare it with a zero-return and momentum baseline.
4. Wire portfolio and risk components into the simulator, then cross-check against a second engine.

The complete proposed architecture and open design decisions are in [`system-design.md`](system-design.md).

