# Overnight implementation log

Accepted scope: research/replay only; no broker orders; no fine-tuning. Local GPU permitted,
$0 new paid services, up to 100 GB new downloads/data. User downloaded the three pinned
models. Permission received to temporarily stop `vllm_qwen38fn`; user later clarified it can remain
stopped when unused.
Check-in requested October 5 at 09:00 America/Los_Angeles. Goal active.

## Initial state

- Qwen endpoint: http://127.0.0.1:8000, model `qwen3.8-flash-next`.
- Existing Qwen occupied about 100 GB; it remains stopped and is not part of this broad run.
- JEV-9B revision b63f651ce8ed64481d3f5e73ecdb05f740042f01.
- Kronos-small revision 901c26c1332695a2a8f243eb2f37243a37bea320.
- Kronos-Tokenizer-base revision 0e0117387f39004a9016484a186a908917e22426.

## Completed data/model foundations

- Credential aliases and SEC identity configured; no secret values in reports.
- Atomic writes and cross-process provider rate budgets implemented.
- Historical revised text delayed to its revision time; original publication retained.
- SEC header parser and container report path repaired.
- Backfill now uses explicit ranges, bounded batches and repeatable checkpoints.

## First verified checkpoint

- 33 tests pass; lint and mypy pass.
- 3,145 Alpaca articles normalized, 308 revised articles delayed to revision time; no rejects.
- 36,609 intraday bars; 15,444 regular-hours bars. Daily warm-up from January 2024.
- Exploratory DQ succeeds with no hard failures; latency and dedupe remain explicitly unverified.
- Qwen: PhraseBank subset n=256 accuracy 0.9570 / macro F1 0.9558;
  FiQA full test n=234 accuracy 0.8462 / macro F1 0.7164. No inference errors.
- All user-downloaded model shards verified present. Beginning authorized Qwen -> JEV switch.

## Current replay run

- Replay inputs are frozen and hashed: 625 ticker-cycle samples, 3,855 event/ticker scores,
  and 362 ticker-session price forecasts. Universe is the eight selected stocks, not a broad
  survivorship-free sample.
- JEV zero-shot benchmark is complete: PhraseBank all-agree n=2,264 (accuracy 0.882, macro-F1
  0.877); FiQA n=234 (accuracy 0.876, macro-F1 0.761). Public-set contamination is unknown.
- Qwen smoke benchmark is currently a PhraseBank subset (n=256) and full FiQA test (n=234).
- Kronos calendar alignment and inference mode were corrected; forecasts are being regenerated
  with pinned revisions, 32 paths per job, dropout disabled, and atomic per-job caching.
- JEV replay scoring is complete: all 625 cycles. The response cache makes each decision
  reproducible without another inference call.
- All 362 Kronos forecast files match the corrected input fingerprint and `eval()` provenance.
- Exploratory replay completed through 2026-09-18. Combined JEV+Kronos return 4.42% (max drawdown
  -5.24%, 31 closed trades); SPY 2.19% (-3.36%); 30 random controls median 3.06%, range
  -3.86% to 14.63%. Standard 12-1 momentum returned 1.28%. See `replay-report.md` for caveats.
- The initial eight-stock run briefly restored Qwen after JEV testing. It was later stopped again
  after the user clarified that Qwen need not be restored while unused. Qwen is a comparison only;
  the replay uses JEV + Kronos.
- Full Qwen zero-shot benchmark completed with zero errors: PhraseBank n=2,264, accuracy 0.953,
  macro-F1 0.949; FiQA n=234, accuracy 0.846, macro-F1 0.716. Both benchmark reports warn that
  public-set contamination is unknown and sentiment accuracy is not trading performance.
- Rebuilt `wallstreet-review:local` and smoke-tested its offline exploratory DQ command against
  the mounted cache. It returned EXPLORATORY and wrote the report successfully.
- Preserved strict DQ's expected failure separately in `docs/data-quality-strict.md`; main DQ
  report remains the successful exploratory status.
- Final source checks: 37 tests, Ruff, and Mypy pass.
- Exploratory DQ rerun: 3,145 events, zero duplicate IDs/acausal times/early revised-text uses/
  rejects; 48,295 intraday bars, 20,358 regular-hours bars. Dedupe and news latency remain
  unverified, so strict research certification is unavailable.
- 37 tests pass; Ruff and Mypy pass. No broker connections/orders, fine-tuning, or paid services.

## Broad-universe follow-up (2026-10-04)

- Expanded the Alpaca cache to 3 years of raw and adjusted daily bars: 3,885,774 rows per store.
  Screened 6,537 listed-stock proxies, with 5,549 having daily bar coverage; 1,932 met the
  point-in-time price/liquidity screen on 2026-10-02. The frozen top-25 liquid watchlist yields 23
  names with enough pre-test history.
- Cached three months of news and 15-minute bars for that watchlist. Broad DQ remains EXPLORATORY;
  the Alpaca universe snapshot is not historical membership and latency/dedupe assumptions remain
  unverified.
- Broad price-only comparison (2024-11 through 2026-10): top-20 12–1 momentum +57.7% after modeled
  costs vs SPY +37.8%; 33/100 random portfolios beat SPY. Survivorship and short-window risks apply.
- Scored 2,433 JEV ticker-cycle samples and generated 1,387 Kronos date forecasts. Sixteen Kronos
  dates produced invalid paths and were marked; 1,773 replay samples remained. The sector map was
  corrected, but the replay still used the biased 400-day Kronos forecast and P(up) >= 0.5 filter.
  All strategy and random-control portfolio figures from this replay are now marked
  INVALIDATED/DIAGNOSTIC ONLY, including JEV-only because rows were filtered by Kronos validity.
- JEV and Kronos were tested together without an allocation error and are currently left loaded.
  Qwen remains stopped. No orders, Reddit data, paid-data calls or fine-tuning were used.
- Final checks before the sector-map correction: 39 tests pass, Mypy passes, Ruff passes, and
  `git diff --check` passes. Rerun all checks after the sector-map and lookback-sweep updates.
- Fixed the broad replay's false `unknown` sector bucket: unclassified names no longer compete
  for a shared sector cap, and the 23-stock watchlist now has explicit sector assignments. Reran
  capped and uncapped controls; reports and replay chart reflect the corrected outcomes.
- Tested four zero-shot Kronos daily configurations across 61 dates and 23 names, including paired
  SPY forecasts. L40/T1 rank IC was +0.091 (5-date block CI +0.003 to +0.179); the 20-day
  reversal control was +0.131. ATR(14) ranked realized volatility better than every Kronos
  dispersion setting. The 400-day setting produced 10 invalid jobs in this 10-path sweep.
- Tested Kronos on bars available at actual news decisions: 1,060 valid morning samples (15m,
  09:45) and 1,034 afternoon samples (30m), each across 61 dates. Raw forecast rank ICs were
  +0.226 and +0.191, but after per-date controls for reversal, long-term mean distance, current
  market-adjusted move, and gap they fell to +0.066 (CI -0.010 to +0.143) and +0.014 (CI -0.072
  to +0.099). This is exploratory evidence of possible morning incremental value, not a validated
  edge. See `broad-universe-experiment.md` and its SVG charts.
- Added strict explanatory handling for invalid forecast values and a replay guard against missing
  bars. All 40 tests pass; Ruff, Mypy across 44 files, and `git diff --check` pass. JEV and Kronos
  remain available; Qwen remains stopped. No Reddit, fine-tuning, paid-data calls, or orders.
- Milestones: M1 code is built, but M1 is still incomplete pending the market-session latency and
  duplicate probes, full required backfills, and strict DQ exit gate. JEV/Kronos serving and basic
  inference ran, but M2's JEV memorization probe was never implemented or run; no JEV clean-window
  claim is established. The Kronos clean-period feature sweep is next, using a historical
  point-in-time liquidity universe rather than today's 2026 watchlist.
