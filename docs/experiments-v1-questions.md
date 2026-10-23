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
- **Q-E3.2 Cutoff strategy.** Proposed: option A (ChronoGPT-instruct yearly vintages) as the main run. Also run option B (a modern open model with company names hidden) as a second, clearly labelled run? If so, which model? Unresolved.
- **Q-E3.3 Text used.** Proposed: headline + summary, no full article content. Or headline only (as Lopez-Lira & Tang)? Unresolved.
- **Q-E3.4 Edited articles.** The API returns the latest version of an article; `updated_at` may be later than `created_at`. Proposed: drop any article whose `updated_at` is more than 1 hour after `created_at`, and report how many are dropped. Alternatives: keep all, or use headline only and assume headlines aren't edited. Unresolved.
- **Q-E3.5 Scoring method.** (a) Lopez-Lira & Tang YES/NO/UNKNOWN prompt (proposed default; note small models did poorly with prompting in their paper); (b) embeddings + per-fold ridge regression (closer to He et al., but adds a second trained model); (c) both. If (a), keep "short term" as published, or change to "next month" to match the label? Unresolved.
- **Q-E3.6 Multi-symbol articles.** Proposed: score per tagged symbol; drop articles tagged with more than 5 symbols. Unresolved.
- **Q-E3.7 Base feature set.** Proposed: `px-v1` + news (`pxn-v1`). Also run `pxf-v1` + news? Unresolved.
- **Q-E3.8 Inference engine.** Proposed: Hugging Face `transformers`; switch to vLLM only if confirmed to load ChronoGPT. Unresolved.
- **Q-E3.9 Missing vintages.** If a yearly vintage is missing (2015 was not visible on Hugging Face on 2026-10-22), proposed: use the most recent *earlier* vintage, never a later one. Accept? Unresolved.

## Progress

- 2026-10-22: E1 (`pxv1f3-60a47f8a0fcb`) walk-forward started on commit `0646703`. E2 and E3 not started.
