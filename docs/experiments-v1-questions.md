# Experiments v1 — open questions

Spec: `docs/experiments-v1-spec.md`. Only Yash resolves these. "Proposed" is the spec author's suggestion, not a decision.

## Applies to all experiments

- **D-E0 Success test.** Proposed: an improvement requires (1) at or above the 95th placebo percentile overall, (2) positive mean test rank IC in at least 4 of 5 folds, and (3) long-only beating equal-weight across folds at 2× costs. Accept, loosen, or tighten? Unresolved.
- **Q1 (from `fund-v1-questions.md`).** How should the "Placebo pass" field be relabelled? Unresolved.

## E1 — fair px-v1 rerun

- **D-E1 Reading the outcome.** Proposed: if `pxv1f3` passes D-E0, `px-v1` is the baseline for E2/E3; if not, E2/E3 still run because they test a new horizon and new information rather than more tuning. Accept? Unresolved.

## E2 — weekly rebalancing

- **Q-E2.1 Feature set.** Run weekly with `px-v1` only, or also `pxf-v1`? Proposed: `px-v1` only, since `pxf-v1` was noise at monthly and annual fundamentals change too slowly to matter week to week. Unresolved.
- **Q-E2.2 Rebalance day.** Proposed: last XNYS session of each ISO week. Alternative: first session of the week. Unresolved.
- **Q-E2.3 Buffer.** Keep top-35 / hold-25 as at monthly, or widen for weekly to reduce churn? Proposed: keep it unchanged for a like-for-like comparison, and also report buffer-off. Unresolved.
- **Q-E2.4 Features at a weekly horizon.** `px-v1` was designed for a monthly label. Should E2 change no features at all (proposed), so only the horizon differs? Unresolved.

## E3 — news + monthly

- **Q-E3.1 Where the LLM runs.** Proposed: offline batch scoring into a point-in-time store for backtests, with the same `NewsScorer` function used inside the daily graph later for paper trading. Alternative: the LLM runs as a live node inside every backtest tick (slower, but literally the same code path). Unresolved.
- **Q-E3.2 Cutoff strategy.** Yash, 2026-10-22: for now, use a model already on the Spark (Qwen3, JEV-9B or Gemma) rather than downloading ChronoGPT. Consequence to keep in view: these are 2025–2026 models, so every backtest year (2017–2025) is scored by a model that may remember what happened next; E3 backtest results must be labelled as possibly look-ahead-biased, and only post-cutoff dates and paper trading are clean. Still open: whether to hide company names (option B) as well, and whether a ChronoGPT run comes later as a check.
- **Q-E3.3 Text used.** Proposed: headline + summary, no full article content. Or headline only (as Lopez-Lira & Tang)? Unresolved.
- **Q-E3.4 Edited articles.** The API returns the latest version of an article; `updated_at` may be later than `created_at`. Proposed: drop any article whose `updated_at` is more than 1 hour after `created_at`, and report how many are dropped. Alternatives: keep all, or use headline only and assume headlines aren't edited. Unresolved.
- **Q-E3.5 Scoring method.** (a) Lopez-Lira & Tang YES/NO/UNKNOWN prompt (proposed default; note small models did poorly with prompting in their paper); (b) embeddings + per-fold ridge regression (closer to He et al., but adds a second trained model); (c) both. If (a), keep "short term" as published, or change to "next month" to match the label? Unresolved.
- **Q-E3.6 Multi-symbol articles.** Proposed: score per tagged symbol; drop articles tagged with more than 5 symbols. Unresolved.
- **Q-E3.7 Base feature set.** Proposed: `px-v1` + news (`pxn-v1`). Also run `pxf-v1` + news? Unresolved.
- **Q-E3.8 Inference engine.** Finding 2026-10-22: the installed vLLM (0.20.0, AgentCache venv) cannot load JEV-9B's text-only Qwen3.5 class (`Qwen3_5TextConfig` type error; JEV's README says it was tested on a September 2026 vLLM dev build). The `transformers` 5.9 + `peft` path works. Proposed: `transformers`. Unresolved. Update 2026-10-23: with Gemma chosen, vLLM in the local image `ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4:latest` serves it correctly (spec 5.5), so this question no longer blocks anything.
- **Q-E3.9 Missing vintages.** Not applicable while ChronoGPT is not used (Q-E3.2).

- **Q-E3.10 Which local model.** Yash, 2026-10-23: Gemma (Gemma-4-26B-A4B NVFP4 on the Spark). Serving and scoring are specified in spec 5.5. Resolved by Yash.
- **Q-E3.11 Anonymization pitfall.** Replacing the target company's name with "the company" misfires when the headline is about a *different* company but tagged with ours (e.g. "Why Is AMD Stock Falling?" tagged INTC becomes a headline that seems to be about "the company"). In the sample, named and anonymized scores correlate 0.83 but differ by 0.25 on average, mostly on such headlines. If anonymizing, should the question name the headline's subject, or should articles whose headline does not mention the target be dropped or scored separately? Unresolved.

## Feasibility check (2026-10-22, scratch only, nothing added to the repo)

- Alpaca news: 150 articles for 10 symbols, 2026-10-05..16 (capped at 3 pages). `X-RateLimit-Limit: 200` per minute confirmed on the free plan. 22 of 150 articles had `updated_at` ≠ `created_at` (relevant to Q-E3.4). 50 had one symbol; 9 had 10 or more.
- JEV-9B System 1, `choice` over ["good news for the stock", "bad news for the stock", "neither or unclear"], score = P(good) − P(bad): 131 (headline, symbol) pairs × 2 prompts (named and anonymized) in 33 s, about 8 decisions/s unbatched on the GB10. Full-history time depends on the E3.0 article count; batching should speed it up and is untested.
- Scores look sensible on inspection: analyst downgrades and lawsuits near −1, price-target raises near +1, ambiguous headlines near 0 with high "unclear" probability. This is a sanity check only, not evidence of predictive value.
- GPU was idle when checked (no other compute jobs on the GB10); the E1 walk-forward runs on CPU alongside.

- Gemma check 2026-10-23 (same 131 pairs): all parsed; 14.6 scores/s at 32 concurrent; 82 YES / 26 NO / 23 UNKNOWN; correlation 0.61 with JEV-9B. Details in spec 5.5.

## Progress

- 2026-10-23: Yash authorized implementation on Proposed defaults without stopping; the implementer marks each default "Assumed by implementer, <date>" here.
- 2026-10-22: E1 (`pxv1f3-60a47f8a0fcb`) walk-forward started on commit `0646703`. E2 and E3 not started.
