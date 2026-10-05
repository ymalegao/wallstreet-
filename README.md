# wallstreet-

Event-driven US-equity trading system: news and filings → open JEV decision model + Kronos → deterministic risk governor → Alpaca.
Design: `docs/01-research-spec.md` (evidence, metrics, gates) and `docs/02-design-spec.md` (architecture, build order, evaluation).

## Status
- **M0** (repo, CI) done.
- **M1** data-layer code is built and tested, but the milestone is **not complete**. Still required: market-session API probe (news latency, SEC timing, dedupe threshold), the full required historical backfills, and a passing strict `dq_report.py` exit gate. Cached research data and exploratory DQ reports do not satisfy M1.
- **M2** model inference has been exercised locally, but the required JEV memorization probe has **not been implemented or run**. No JEV clean evaluation window is established. See `docs/02-design-spec.md` §3.3 and §6.

## Run on the Spark (M1)
```bash
uv sync
cp .env.example .env        # fill in Alpaca (paper keys are fine), Finnhub, and SEC_USER_AGENT
uv run python scripts/probe_apis.py --ws-seconds 600   # run during market hours; writes docs/data-probe-report.md
# read the report, then record the measured values in configs/sources.yaml
uv run python scripts/backfill.py news --start 2015-01-01
uv run python scripts/backfill.py daily-bars --start 2016-01-01
uv run python scripts/backfill.py intraday-bars --start 2016-01-01
uv run python scripts/backfill.py edgar
uv run python scripts/backfill.py finnhub
uv run python scripts/backfill.py normalize
uv run python scripts/dq_report.py                     # M1 exit gate; writes docs/data-quality-report.md
```

## Development
```bash
uv run pytest -q && uv run ruff check src tests scripts && uv run mypy
```
