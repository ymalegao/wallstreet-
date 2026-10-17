# Pipeline v0 runbook

## What is shared and what flows through the graph

The coordinator supplies `MarketSnapshot` and `PortfolioState` for every tick. A `SnapshotProvider` uses one preloaded `BarHistory` to produce bars available by that tick. The graph itself starts at `FeatureBuilder` and sends typed messages through `Alpha`, portfolio construction, risk, and execution.

`BarHistory` is read-only. The broker is mutable shared state and only `Execution` receives it. The rest of the graph uses its declared message inputs, so a replay can replace one input snapshot and compare the downstream node hashes.

## Download adjusted Alpaca daily bars

The downloader reads `ALPACA_KEY` and `ALPACA_SECRET` from the repository-root `.env` file or process environment. It also accepts Alpaca's standard `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` environment names. Download at least 253 trading sessions before the backtest start so 12-1 momentum has its full lookback.

```bash
uv run python -m ws.pipeline_v0.cli download-bars \
  --start 2024-10-14 --end 2026-10-16 \
  --store data/pit-bars
```

The downloader requests Alpaca's `1Day` bars with `adjustment=all`, which adjusts splits, dividends, and spin-offs. It paginates responses and appends rows into the PIT bar store. See the [Alpaca historical bars API](https://docs.alpaca.markets/us/reference/stockbars).

## Run one year of daily decisions

```bash
uv run python -m ws.pipeline_v0.cli backtest \
  --start 2025-10-17 --end 2026-10-16 \
  --store data/pit-bars \
  --out runs/sp100-momentum-2025-2026
```

The command writes `events.jsonl`, `fills.csv`, `cancelled.csv`, and `equity.csv` into a new output directory. It refuses to overwrite those files. A completed run reports its run id, final marked equity, and fill count.

The checked-in universe is a dated S&P 100 ETF holdings proxy with 101 tickers. Applying today's proxy to older dates creates survivorship bias. The downloader also models availability as close plus 15 minutes; Alpaca's historical endpoint does not retain the original first-seen time or a point-in-time adjustment archive. A real-data run checks the ingestion and execution path, but it is not leakage-proof performance evidence. The synthetic fixture acceptance tests remain offline and do not call Alpaca.

## Re-run the acceptance suite

```bash
uv run pytest -q
```
