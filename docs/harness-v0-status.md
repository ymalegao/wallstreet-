# Pipeline v0 status

This branch now has an end-to-end daily research path. The earlier prototype remains on `research-harness-v0-checkpoint`.

## What is implemented

- **Coordinator boundary:** each tick receives typed `MarketSnapshot` and `PortfolioState` messages. `SnapshotProvider` is outside the DAG and reads the preloaded `BarHistory`.
- **Five-node graph:** features → 12-1 momentum scores → equal-weight targets → deterministic risk → broker orders. `Execution` alone receives the mutable broker resource.
- **Replayable runtime:** configured node specs are logged, DataFrame hashes are canonical, and replay compares each node output digest.
- **Daily simulation:** decisions happen at close plus 30 minutes and orders fill at the next XNYS open with transaction costs.
- **Historical ingestion:** the Alpaca downloader pages through adjusted daily bars and writes to the PIT store. The CLI exposes separate `download-bars` and `backtest` commands.
- **Acceptance coverage:** the synthetic fixture exercises formulas, point-in-time filtering, orders, next-open fills, one-year backtest output, replay, and the equal-weight sanity check. Tests make no network calls.

## Run it

See [`pipeline-v0-runbook.md`](pipeline-v0-runbook.md) for the download and backtest commands. Run the offline acceptance suite with:

```bash
uv run pytest -q
```

## Limits to keep visible

The checked-in universe is a dated iShares OEF holdings proxy with 101 tickers. Applying today's proxy to older dates creates survivorship bias. The Alpaca endpoint's historical adjustment is not a point-in-time corporate-action archive, and `available_at` is modeled as close plus 15 minutes rather than taken from a vendor first-seen record. A downloaded history can validate the data and execution path, but does not establish leakage-free alpha. A point-in-time vendor archive and historical universe membership remain necessary before making long-horizon performance claims.

