# Fundamentals v1 decisions and progress

Yash accepted D1–D7 and all five implementation defaults on 2026-10-18. There are no unresolved implementation choices at this stage.

## Accepted decisions

- **D1:** Annual fundamentals only.
- **D2:** Filing availability is after the next XNYS session close plus 15 minutes.
- **D3:** 50 train-only label permutations.
- **D4:** Keep held names ranked within the top 35 and fill to 25 positions.
- **D5:** Measure top-minus-bottom quintile returns for evaluation only; do not short in the simulator.
- **D6:** Walk-forward only; do not run `final-eval`. Forward paper trading is the next unseen test.
- **D7:** Keep the current S&P 100 proxy and label its survivorship bias.

## Accepted implementation defaults

- **SEC raw layout:** each command adds its immutable file to `data/sec/raw/<date>/` and updates that dated manifest.
- **Ken French raw layout:** `data/ken-french/raw/<date>/`.
- **`asset_growth` FY−1 assets:** choose the visible instant fact with period end closest to the target date 365 days before the current fiscal-year end, within the 350–380-day window; restatements use the latest visible filing.
- **`ivol_21d`:** OLS includes an intercept.
- **`resid_mom`:** use sample standard deviation for the 11 residuals.

## Data inventory

- Alpaca SIP: 102 symbols, 2016-01-04 through 2026-10-16. Current S&P 100 proxy remains survivorship-biased.
- SEC raw snapshot: `data/sec/raw/2026-10-18/`; ticker map and both specified bulk archives downloaded with manifest hashes.
- Ken French snapshot: `data/ken-french/raw/2026-10-18/`; FF12 SIC, five-factor, and momentum archives downloaded with manifest hashes.
- SEC facts snapshot: `data/sec/facts/8ef316af7b2b80b5ecaa697c326e74dede3298f87ab01cf86511dd68aacca3b4.parquet`, 2,530,872 rows. `BRK.B` is missing from the official ticker map and receives null fundamentals; no alternate lookup was used.
- Coverage report: `docs/fund-v1-coverage.md`.
- `px-v1` F2 walk-forward completed: `runs/training/pxv1f2-60a47f8a0fcb/report.md`.
- `pxf-v1` walk-forward stopped during the 2024 fold's portfolio simulation under model id `pxfv1-60a47f8a0f-8ef316af7b`. The trace ends with `ValueError: insufficient pre-fill cost history for GEV`. Fold checkpoints for 2021–2024 were saved, but there is no final report; the run log is `runs/training/pxfv1-60a47f8a0f-8ef316af7b.attempt.log`.
- Portfolio construction now requires 21 close-to-close returns (22 visible closes) and valid 20-session ADV. The broker rechecks pre-fill inputs and records a cancellation rather than failing or assuming zero costs.
- Fresh PXF walk-forward completed on 2026-10-19 under model id `pxfv1-60a47f8a0f-8ef316af7b`: `runs/training/pxfv1-60a47f8a0f-8ef316af7b/report.md`. Console output is saved to `runs/training/pxfv1-60a47f8a0f-8ef316af7b.rerun-20261019.log`.
- An earlier PXF attempt left only the 2021 fold checkpoint and no report or registry entries. Its artifacts are preserved under `.incomplete-20261018`; the reason it ended was not present in the retained output.
- No holdout evaluation has been run.

## Implementation progress

- F1/F2/F3 are implemented in the training, portfolio, and evaluation code.
- SEC point-in-time fact extraction, FF12 SIC mapping, and point-in-time annual selectors are implemented.
- A shared `pxf-v1` feature function is wired into both the training table builder and `FeatureBuilder`.
- The `pxf-v1` table is built from the immutable SIP and SEC snapshots: 218,665 rows across 2,165 dates, ending 2025-08-13 with all labels before the 2025-09-16 holdout.
- `px-v1` F2 walk-forward covers five annual folds (2021 through partial 2025). Mean daily test rank IC is 0.0231 overall, but varies substantially by year; treat this as preliminary predictive evidence, not a demonstrated net trading edge. The report contains fold-level metrics and portfolio baselines.
- `pxf-v1` walk-forward (rerun after the pre-fill eligibility fix) completed all five folds. Mean daily test rank IC is 0.0008; the real model sits at the 46th percentile of the 50 train-only placebo runs, i.e. indistinguishable from noise.
- The `px-v1` F2 report (2026-10-18) predates the placebo fix below and the pre-fill eligibility change, so its 96th-percentile placebo result is not directly comparable to `pxf-v1`. A rerun under the current code is running as `pxv1f3-60a47f8a0fcb`; console output: `runs/training/pxv1f3-60a47f8a0fcb.log`.
- Train-only placebo fits now use the selected real model's per-seed iteration counts, so unpermuted validation labels do not choose each null model's complexity.
- Full test suite: 62 passed after addressing parser/tag/staleness/return-formula expectations, the placebo validation-selection issue, and the new eligibility/backstop behavior.
- No `final-eval` was run.

## Open questions

- **Q1 (placebo "pass" label):** The per-fold and overall "Placebo pass" / "status: pass" fields check only that the placebo mean IC is within ±0.02 of zero (a leakage sanity check). They do not test whether the real model beats the placebos; for example, `pxf-v1` 2022 shows "pass" at the 6th percentile. Should the label be renamed, or a separate real-vs-placebo verdict added? Unresolved.
- **Q2 (px-v1 rerun model id):** Resolved by Yash on 2026-10-22: code change. `px-v1` walk-forwards now write model id `pxv1f3-<snapshot>`; `pxv1f2-60a47f8a0fcb` is kept as the pre-fix run.
