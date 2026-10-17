# Harness-first v0 scaffold

This is the first executable slice of `src/ws/spec1.md`. It establishes node composition, replay, and clock-bound daily-bar reads before adding portfolio decisions or broker calls. The existing trading prototype remains alongside it.

## What is implemented

- `ws.core`: immutable ticks and messages, a simulation clock, node and policy contracts, a typed DAG builder, level-by-level concurrent execution, explicit optional-node `Missing` values, fail-closed required-node handling, and JSONL output hashes for replay.
- `ws.store.pit_bars`: raw-price daily bars with separate exchange-close `event_time` and conservative `available_at`, partitioned Parquet persistence, idempotent writes, and reads that take a `Clock` rather than a caller-supplied as-of time.
- `configs/universe/sp100-current.json`: dated U.S. equity tickers from the public iShares OEF holdings file. The ETF is a proxy for S&P 100 membership; its 101 equity tickers include both Alphabet share classes.
- `scripts/prepare_pit_daily_bars.py`: converts the existing Alpaca raw daily-bar cache to the clock-filtered PIT layout.
- `ws.portfolio`: equal-weight top-positive-score construction plus deterministic long-only name, sector, volatility, drawdown, gross, and liquidity constraints.
- `ws.broker.costs`: an explicit one-way commission + half-spread + square-root-impact cost function.
- `ws.backtest.simulate_daily`: a stateful daily simulator that applies prior-close signals at the next session open, separates overnight and intraday returns, and rejects signals unavailable at execution time.
- Focused tests cover the 10-tick constant → double → sink replay and the future-row shuffle property.

## Run it

```bash
uv run --group research python -m ws.research.demo
uv run --group dev pytest -q tests/test_core_runtime.py tests/test_pit_bars.py tests/test_portfolio_risk.py tests/test_daily_backtest.py
uv run --group dev ruff check src/ws/core src/ws/store/pit_bars.py src/ws/portfolio src/ws/broker src/ws/backtest
uv run --group dev mypy src/ws/core src/ws/store/pit_bars.py src/ws/portfolio src/ws/broker src/ws/backtest
```

The model demo is intentionally synthetic and CPU-only. It proves the training node can fit a baseline inside the pipeline; it says nothing about real returns. The current run has 3,860 training rows and 640 evaluation rows; model RMSE is 0.01230 versus 0.01208 for a zero-return forecast, and Pearson correlation is -0.0165. The baseline loses to the naive forecast, which is the null result to preserve. Its inputs, JSONL events, and run card are under `artifacts/research_demo/` locally.

## Data commands

With Alpaca keys in `.env`, refresh the dated universe snapshot and backfill its raw daily bars:

```bash
uv run python scripts/snapshot_oef_universe.py
uv run python scripts/backfill.py daily-bars \
  --start 2016-01-01 \
  --symbols-file configs/universe/sp100-current.json
uv run python scripts/prepare_pit_daily_bars.py
```

The Alpaca endpoint returned no AAPL bars for 2015 in a direct probe, but did return data from January 2016. The current data slice therefore begins in 2016. `available_at` is set to the XNYS session close plus 15 minutes. The source bars are unadjusted raw prices; `adj_factor` stays null until a point-in-time corporate-action source is added. The converter writes immutable Parquet copies and a content-addressed manifest under `data/snapshots/<sha256>/`; the `data/` directory is ignored by Git.

The generated local snapshot is `4a78c64440cdc06f825bee9b74737bb9055eb2398be849cb46bfb85b155ff46b`: 266,145 bars across 101 symbols, from the Jan 4, 2016 close through the Oct 16, 2026 close, in 130 year/month partitions. All 130 copied files match their manifest hashes. A real clock query returned zero rows one minute before the first bar became available and 95 rows exactly at its availability time.

## What this does not establish

- The current S&P 100 proxy applied back to 2016 has survivorship bias. It is a smoke-test universe, not a historical index-membership panel.
- The OEF constituent snapshot is dated but not the index provider's own constituent file.
- There is no historical membership, corporate-action adjustment pipeline, feature parity, vectorized sweep engine, independent engine cross-check, or Alpaca paper broker yet. Risk rules exist as deterministic components but are not yet wired into the simulator.
- Do not interpret the synthetic metrics or this data plumbing as an alpha result.

## Next build steps

1. Add corporate actions and historical membership.
2. Cross-check the daily simulator against an independent engine and benchmark strategy.
3. Wire risk constraints into portfolio execution and add a paper broker behind the same interface.
4. Add stateless feature and LightGBM nodes with chronological training/label cutoffs.
5. Add the broker-parity and paper-forward gates.

## Verification result

The full existing suite passes with `uv run --no-sync --group dev --group research python -m pytest -q` (74 tests). `ruff check src tests scripts` passes, and repository-wide mypy reports no issues in 46 source files. The installed `pytest` script entrypoint does not add the repository root to `sys.path` in this environment; the module invocation does, which is needed by existing tests that import `scripts.*`.

