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
- Before rerunning, portfolio construction now requires 21 close-to-close returns (22 visible closes) and valid 20-session ADV. The broker rechecks pre-fill inputs and records a cancellation rather than failing or assuming zero costs. A fresh PXF walk-forward is pending.
- An earlier PXF attempt left only the 2021 fold checkpoint and no report or registry entries. Its artifacts are preserved under `.incomplete-20261018`; the reason it ended was not present in the retained output.
- No holdout evaluation has been run.

## Implementation progress

- F1/F2/F3 are implemented in the training, portfolio, and evaluation code.
- SEC point-in-time fact extraction, FF12 SIC mapping, and point-in-time annual selectors are implemented.
- A shared `pxf-v1` feature function is wired into both the training table builder and `FeatureBuilder`.
- The `pxf-v1` table is built from the immutable SIP and SEC snapshots: 218,665 rows across 2,165 dates, ending 2025-08-13 with all labels before the 2025-09-16 holdout.
- `px-v1` F2 walk-forward covers five annual folds (2021 through partial 2025). Mean daily test rank IC is 0.0231 overall, but varies substantially by year; treat this as preliminary predictive evidence, not a demonstrated net trading edge. The report contains fold-level metrics and portfolio baselines.
- PXF training and 2021–2023 evaluation completed; the 2024 portfolio run stops because GEV lacks 21-session pre-fill cost history. No aggregate PXF result is available until this execution edge case is handled and the walk-forward completes.
- Train-only placebo fits now use the selected real model's per-seed iteration counts, so unpermuted validation labels do not choose each null model's complexity.
- Full test suite: 62 passed after addressing parser/tag/staleness/return-formula expectations, the placebo validation-selection issue, and the new eligibility/backstop behavior.
- No `final-eval` was run.
