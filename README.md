# Multi-Agent Trading Research Harness

A small, auditable starting point for researching daily U.S. equity strategies. The main idea is to make each pipeline step a typed `Node`, enforce point-in-time data access through a shared clock, and keep portfolio and risk decisions deterministic.

This repository is scaffolding. It does not contain a validated trading edge, reliable historical constituent data, or a live broker integration.

## Start here

1. Read [`docs/harness-v0-status.md`](docs/harness-v0-status.md) for what currently runs and what is still missing.
2. Read [`docs/system-design.md`](docs/system-design.md) for the larger intended architecture.
3. Look at [`src/ws/research/demo.py`](src/ws/research/demo.py) for the runnable toy pipeline.

## Run the toy training pipeline

```bash
uv sync --group dev --group research
uv run python -m ws.research.demo
```

It generates synthetic data, filters it by availability time, creates lagged features, fits a small scikit-learn baseline, and writes a run card and event log under ignored `artifacts/research_demo/`. Synthetic results are only a mechanics demonstration.

## Structure

```text
src/ws/core/       typed messages, Node contract, clock, DAG runner, replay log
src/ws/store/      point-in-time Parquet bars
src/ws/research/   synthetic train/evaluate example
src/ws/portfolio/  deterministic allocation and risk rules
src/ws/backtest/   daily-bar simulator
src/ws/broker/     transaction-cost model
tests/             focused checks for those components
```

Nodes can be subclassed and connected by declaring message types:

```python
class FeatureNode(Node):
    name = "features"
    reads = (BarsMessage,)
    writes = FeatureMessage

    async def run(self, ctx: NodeContext) -> FeatureMessage:
        bars = ctx.inputs[BarsMessage]
        ...
```

Compose instances with `Pipeline.builder().add(...).build()`. The builder infers edges from `reads`/`writes`, checks the graph, then the runtime executes independent nodes concurrently and records output hashes for replay.

## Local development

```bash
uv run python -m pytest -q
uv run ruff check src tests scripts
uv run mypy
```
