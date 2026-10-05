# Overnight readiness review and proposed scope

Reviewed 2026-10-04. This is a proposal, not an activated Goal or implementation authorization.

## What exists

The repository implements the data foundation of an event-driven US equities research system:
Alpaca news/bars, SEC filing metadata, Finnhub news, raw archives, normalized stores, duplicate
detection, trading calendars, point-in-time eligibility, and forward-return labels. The intended
next layers are JEV-9B text scoring, Kronos price forecasts, a deterministic risk governor,
backtests, and Alpaca paper execution. Those layers are not implemented yet.

The later design document takes precedence where it explicitly revises the research document:
one Spark, US stocks, long-only, idle cash, local zero-shot models first, no fine-tuning initially.
Evaluation windows and gates still need reconciliation between the documents.

## Verification performed

| Check | Result |
|---|---|
| Locked dependencies, native ARM Python 3.11 | Installed successfully |
| Existing tests | 28 passed |
| Ruff lint and formatting | Passed; 27 files formatted |
| Mypy | Passed; 19 source files |
| Hardware | NVIDIA GB10; about 1,714 GiB free disk |
| Docker | Daemon available; image `wallstreet-review:local` builds |
| Container probe command, offline | Fails writing report: `/app/docs` does not exist |
| API probe | 22 PASS, 7 FAIL, 9 MEASURE; see `data-probe-report.md` |
| Alpaca news WebSocket | Authenticated and subscribed successfully |
| Live latency | Not measured; needs sufficient actual news observations, preferably a market session |
| Full backfill / production DQ | Not run; DQ has a reproduced crash |
| Models / GPU inference | Not installed or benchmarked |
| Broker orders | None submitted; paper account endpoint not verified |

API probes used existing secrets with variable names mapped only inside the probe process.
The source code, `.env`, and `configs/sources.yaml` were left unchanged. Generated probe samples
are under ignored `data/probe/`; the report is under `docs/`. A local `.venv` and Docker image
were created for testing. Shell commands required the approved fallback because the default
sandbox failed with `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`.

## Findings, in priority order

1. **Credentials are named incorrectly for this app.** `.env` has `ALPACA_KEY`, `ALPACA_SECRET`,
   and `FINN_HUB_API`; `src/ws/config.py:26` expects `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`,
   and `FINNHUB_API_KEY`. All three loaded as absent. `SEC_USER_AGENT` is missing. The existing
   credentials successfully accessed their data APIs after process-local mapping. `.env` is Git-ignored.

2. **The data-quality gate crashes.** A synthetic nonempty dataset reproduced `OverflowError`
   at `scripts/dq_report.py:76`: hour/minute arithmetic retains a narrow integer dtype.
   Cast before multiplying. Add a script-level regression test. Also, by inspection, missing
   intraday bars and an unverified dedupe threshold do not add gate failures; normalization
   rejects are not audited. A passing report must not mean that required checks were skipped.

3. **Historical text availability is not enforced.** `src/ws/ingest/alpaca_news.py:61` attaches
   current historical article text to the original publication time even when it was updated later.
   The sample had 50 revised articles out of 832, with revision-lag p95 around 155 minutes.
   First-write-wins does not undo revisions that happened before our first download. Preserve
   revision metadata and quarantine or conservatively delay revised historical content for evaluation.
   The label API also runs using a hardcoded five-minute latency while config remains UNVERIFIED;
   this was reproduced. Measured availability must be enforced by the research entry point.

4. **Source probes expose timestamp/order assumptions.** Alpaca's returned sample was not
   monotonic in `created_at` despite requesting ascending order. Several cross-source matching
   stories were exactly 240 minutes apart; investigate provider timestamp semantics rather than
   applying an unproven four-hour correction. The SEC probe regex at `scripts/probe_apis.py:233`
   expects `ACCEPTANCE-DATETIME:` but actual headers use `<ACCEPTANCE-DATETIME>`.
   A separate corrected-parser diagnostic found UTC matches for all five sampled filings.
   Retain those fixtures and extend validation across winter/summer before normalizing broadly.

5. **Backfill checkpoints can silently skip later work.** `scripts/backfill.py:76` marks a ticker
   done without recording date range/feed/adjustment. Extending the requested date range therefore
   skips previously completed tickers. Empty fetches also become done. News month iteration does
   not honor an exact historical `--end` within a month. Use interval-aware atomic checkpoints,
   bounded batches, explicit empty/error outcomes, and interrupted-run recovery tests.

6. **Storage is not safe for concurrent writers or interrupted writes.** Two already-open
   `EventStore` instances wrote the same event twice in a temporary reproduction. ID caches are
   per instance. Bar files and checkpoint files are overwritten directly; bars are materialized
   as a full batch in memory. Enforce a single writer initially and use atomic writes; preserve
   provenance/version information for corporate-action-adjusted bar snapshots.

7. **SEC ingestion does not yet provide model-ready filing content.** The backfill archives
   submissions metadata and calls normalization without a body; `fetch_text()` is unused.
   Add relevant document/exhibit retrieval and structured Form 4 parsing before claiming a
   filing-text signal. Current ticker-to-CIK mapping also misses historical rename/delisting issues.
   SEC acceptance time is not universally identical to public dissemination time; preserve that
   uncertainty for historical availability and use observed receipt times in live ingestion.

8. **The universe/coverage design needs an independent denominator.** Daily-bar symbols are
   discovered from downloaded news. Checking news coverage only within that set can miss names
   absent from the news collection. SIVB/FRC sample history exists, but two examples do not prove
   a complete survivorship-free universe. Add explicit symbol history and date-range coverage checks.

9. **Overnight resilience is unfinished.** HTTP retries cover selected status codes but not
   transport failures or HTTP-date Retry-After values. Rate limits are per client, not coordinated
   across processes. The news WebSocket has no reconnect loop or persistent runner. Add provider
   budgets, bounded retries, reconnect/backoff, health logs, and restartable ingestion.
   Use one SEC worker at 2 requests/second initially; retries count toward the same budget.

10. **The container builds but is not operationally complete.** Create report directories at
    runtime or configure mounted output paths. Add `.dockerignore` for secrets/data/environments
    and pin image/tool versions for reproducibility. The current Dockerfile uses selective COPY
    statements; no evidence that `.env` was embedded in the built image.

11. **Research claims need tighter limits.** A memorization test can detect contamination but
    failure to detect it cannot certify an unknown training cutoff. Treat the proposed JEV clean
    window as unestablished. Pin model revisions and document base/student/teacher provenance;
    label uncertain historical runs exploratory and accumulate prospective evidence. Reconcile
    two-month versus six-month holdouts and older G1/G2 periods versus the revised model plan.
    Overlapping multi-day labels need dependence-aware inference. A prior bar close is a label
    reference, not a guaranteed executable price; backtests need subsequent fills and costs.

12. **Model integration has real compatibility work.** JEV's published serving recipe requires
    its decision adapter, bias/calibration files, and specific vLLM capabilities. Its general
    calibration is not calibration of stock returns. Kronos's documented `sample_count` averages
    paths; estimating P(up) requires retaining individual sampled outcomes. Neither GPU path has
    been validated on this machine. Check compatibility before investing in a large backfill.

## Proposed first overnight run

Target: a reliable, resumable research pipeline and a small real-data demonstration. Treat eight
hours as a ceiling and checkpoint time, not a promise that every later milestone will fit.

| Priority | Work | Evidence of completion |
|---|---|---|
| 1 | Correct config loading, probes, DQ crash/gates, source availability handling, rate limits and container output | Regression tests, full CI checks, container smoke test, provider report with explicit unresolved checks |
| 2 | Fix checkpoints/atomic writes; collect and normalize a bounded sample | Restart reproduces the same counts; audit of timestamps, revisions, rejects and bar completeness |
| 3 | Pin and smoke-test JEV-9B and Kronos locally | Real inference on sample inputs; measured speed/memory; clear compatibility blocker if unsuccessful |
| 4 | Add a minimal replay, fixed scoring, deterministic governor and basic SPY/random-entry comparisons | Reproducible sample report with costs, assumptions, and exploratory status; no claims of validated alpha |
| 5 | Package handoff | Commands, tests, changed files, data manifest, logs, remaining blockers, next-session checklist |

Start with the eight probe tickers plus SPY over a recent bounded interval, with earlier bars for
indicator/model warm-up. This is an integration sample, not a representative investment universe
or a final holdout. Choose the historical evaluation interval only after provenance review.
Expand collection after correctness checks pass; do not start by downloading ten years of every bar.
If model installation stalls, continue improving ingestion, fixtures, replay and deterministic
components. Do not fabricate model predictions or claim the integration is complete.

Sunday can validate authentication and historical data but cannot supply a normal US equity
market session's latency/fill evidence. Schedule that measurement for the next open session.
The design's two-week unattended trial and G4's longer paper-evidence requirement cannot complete overnight.

## Decisions needed from Yash

- Accept the priority order above, or choose model demonstration over data reliability as the main deliverable.
- Confirm overnight scope: research/replay only (recommended first), or also allow paper orders
  after controls pass. If paper orders are desired, identify the keys as paper keys and specify
  the simulated account budget; do not paste secrets into chat.
- Confirm GPU availability and download/storage allowance. Proposed defaults: local compute only,
  $0 new paid services/Brev spend, up to 100 GB new downloads/data, respect existing GPU jobs.
- Give the start time and hard stopping time/timezone; configure the desired Codex usage budget.
  Keep the host/session available and resolve the default sandbox issue before an unattended run.
- Confirm the later design's defaults remain intended: US stocks, long-only, idle cash, one Spark,
  JEV-9B + Kronos zero-shot. No need to repeat credentials or supply a paid SEC subscription.

## Draft Goal text (start only after scope is accepted)

Implement the accepted priorities in docs/03-overnight-readiness.md. Deliver a resumable,
auditable real-data research pipeline with regression tests, working container commands,
source validation, and a bounded replay report. Attempt local JEV-9B and Kronos integration
after the data foundation is reliable. Use local compute and existing data access, spend $0
on new services, stay within the agreed storage budget, and submit no broker orders unless
paper execution is explicitly included. Keep secrets out of logs. Preserve raw evidence and
mark unverified assumptions and exploratory results honestly. Maintain a progress/checkpoint
log and finish with reproducible commands and unresolved blockers. Stop at the agreed deadline
or earlier if the accepted completion criteria are met; do not call partial work complete.

## Primary documentation checked

- [SEC scripted access and rate limit](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data)
- [JEV-9B model card and serving requirements](https://huggingface.co/autotrust/JEV-9B)
- [Kronos usage and sampling API](https://github.com/shiyu-coder/Kronos)
- [Alpaca fractional orders](https://docs.alpaca.markets/us/docs/fractional-trading)
- [Alpaca paper simulation limitations](https://docs.alpaca.markets/us/docs/paper-trading)
- [OpenAI Goals](https://developers.openai.com/cookbook/examples/codex/using_goals_in_codex)

The broader bibliography and all performance/legal assertions in the original research document
were not independently re-audited in this quick readiness pass.
