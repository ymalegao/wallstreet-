# Training v1 implementation status

## Implemented

- Adjusted Alpaca IEX or SIP snapshots are accepted; the selected feed is carried into the training table, model manifest, and reports. IEX outputs are marked development-only.
- `build-table` stores pre-holdout rows only. Decision dates stop before the holdout, and the final 21-session labels also end before it.
- `final-eval` alone constructs holdout rows and outcomes. It can use pre-holdout bars for feature lookbacks and reserves the model's one-time holdout run before reading bars.
- `fit-final` trains five seeded LightGBM models on pre-holdout rows and records model, data, universe, feature, and artifact hashes.
- Walk-forward evaluation records fixed-grid trials, monthly IC and portfolio tables, LightGBM and momentum portfolios, equal-weight and SPY baselines, 1×/2× costs, and concentration summaries.
- The append-only registry counts configuration, placebo, and final seed fits. Markdown reports are rendered from serialized JSON.
- Equal-weight buy-and-hold baselines include only constituents with a first-fill bar and cost history; the CSV/report records included count and excluded symbols. This handles current constituents that listed after a historical fold began.
- Execution rebalances monthly and permits off-schedule risk-reduction sells. Snapshot and training-table outputs are immutable.
- The calendar start is bounded by actual returned bars and the installed XNYS calendar, so requesting older history than the calendar supports no longer fails before table construction.

## Decisions

- Final refit iteration policy: `median-fold`.
- Placebo scope: `train-only`, with `all-splits` tested as the fallback diagnostic.
- T7 scoring target remains `raw_forward_return`, as specified in Section 8.
- Current development data feed: SIP, bounded to the last completed session; no silent fallback to IEX.
- The remaining lambda tie rule stops and asks if an exact tie occurs.
- Details are in [`training-v1-questions.md`](training-v1-questions.md).

## Current SIP snapshot

- Feed: adjusted Alpaca SIP (`feed=sip`, `adjustment=all`). A bounded AAPL/SPY request through the last completed session, 2026-10-16, succeeded without an entitlement error.
- Snapshot: `60a47f8a0fcbf59d8c260945b6d75c36372f9c364f59c1a06cfc29d78907c01c`.
- 268,858 bars across the 101-symbol S&P 100 proxy and SPY; all 102 requested symbols returned bars.
- Requested range: 2016-01-01 through 2026-10-16; actual overall coverage is 2016-01-04 through 2026-10-16. DELL, GEV, LIN, PLTR, SNDK, and UBER begin later because they were not trading during the full period.
- Built table: `data/training/px-v1-sip.parquet`, 218,665 rows over 2,165 dates (2017-01-03 through 2025-08-13). The latest label exits on 2025-09-15, before the 2025-09-16 holdout boundary; `2026-09-16` is the latest overall labeled date used to determine that boundary.
- Detailed per-symbol coverage: [`data-coverage.md`](data-coverage.md).

## SIP walk-forward result

- Full run report and JSON are local outputs under `runs/training/pxv1-60a47f8a0fcb/` and remain gitignored; the key metrics are recorded below.
- Walk-forward used expanding training windows, the preceding year for validation, and the next year for testing, with 21-session label purging/embargo. It evaluated five folds: 2021–2024 and 2025 through 2025-08-13. The locked holdout starts 2025-09-16 and is not included.
- Each fold tried 8 LightGBM configurations × 5 seeds. The five-seed predictions were averaged. The current modal fold-selected final configuration is `num_leaves=31`, `min_data_in_leaf=200`, `lambda_l2=0`.
- At 1× modeled costs, annual portfolio returns were: 2021 +32.4% (equal weight +31.9%, SPY +30.7%); 2022 −13.6% (equal weight −12.8%, SPY −18.9%); 2023 +46.8% (equal weight +31.6%, SPY +25.9%); 2024 +23.7% (equal weight +27.0%, SPY +26.2%); and 2025 through Aug 13 +8.5% (equal weight +14.0%, SPY +10.4%). The model beat both baselines in 2021 and 2023, and lagged equal weight in 2022, 2024, and 2025.
- Monthly cross-sectional rank-IC means by fold were +0.0066 (t=+0.15), −0.0585 (t=−0.61), +0.0617 (t=+1.56), −0.0146 (t=−0.34), and +0.0747 (t=+0.75), respectively. None is strong, stable evidence of predictive skill; 2023 is the strongest fold but still has only 12 monthly observations and a t-stat below 2.
- Turnover remains high: the LightGBM portfolio traded about 8–17 times its starting equity in the tested windows, versus roughly 0.85–1.13 for equal-weight buy-and-hold. The simulator uses monthly rebalancing and reports both 1× and 2× modeled commission, spread, and impact costs.
- The walk-forward run completed its folds, models, and backtests, but the required SIP placebo gate failed. The report records train-only mean test rank IC −0.0770 (t=−1.42) over 12 months. A separate all-splits diagnostic is in `placebo-all-splits.json`: mean −0.0638 (t=−1.97). These single-permutation results fail the ±0.02 acceptance bound but are noisy diagnostics, not proof of leakage. The owner later requested the final fit and holdout score; those actions and results are recorded in the following section.
- The report's 255 trial count is registry-wide, including earlier experiment records; this SIP walk-forward itself ran 200 grid fits (8 configurations × 5 seeds × 5 folds) plus 5 train-only placebo fits. The all-splits diagnostic added five more registered placebo fits after the report was written.

## SIP final model and one-time holdout result

- At the owner's explicit request, `fit-final` and `final-eval` were run on 2026-10-17 despite the unresolved T7 placebo failure. This records the requested score; it does not satisfy the failed placebo gate or resolve its acceptance rule.
- Final model: `pxv1-60a47f8a0fcb`, five seeds, 212,315 pre-holdout symbol-date training rows, training dates 2017-01-03 through 2025-08-13. The selected configuration is `num_leaves=31`, `min_data_in_leaf=200`, `lambda_l2=0`; the `median-fold` policy selected only 2 boosting rounds. That low round count is a material model limitation.
- One-time holdout: 2025-09-16 through 2026-09-16, 25,452 symbol-date rows and 12 month-end rank-IC observations. LightGBM mean monthly rank IC was +0.0597 (t=+0.92; 8/12 positive). Momentum's was +0.0369 (t=+0.54; 6/12 positive).
- At 1× modeled costs, total returns were LightGBM +30.4% (Sharpe 1.91, max drawdown −9.0%, turnover 8.38, costs $2.37 on $1,000 initial cash); 12–1 momentum +29.0% (Sharpe 1.81); equal-weight buy-and-hold +44.2% (Sharpe 1.97); SPY +15.5% (Sharpe 1.19). At 2× costs, returns were +30.1%, +28.8%, +44.2%, and +15.5%, respectively. LightGBM beat momentum and SPY but lagged equal weight by 13.8 percentage points.
- The five largest LightGBM stock contributions were DELL, MU, AMD, SNDK, and LRCX, totaling about $232 of the $304 model profit. The result is concentrated in a handful of semiconductor and memory names, while equal weight performed better across the same window.
- This calendar window was already examined by the earlier IEX final evaluation and the v0 momentum smoke run. Therefore the SIP score is not a clean, project-wide unseen holdout, even though this SIP model ID was reserved and evaluated only once. The registry now locks `final-eval` for this model; do not rerun it.

## Original IEX development snapshot

- Feed: adjusted Alpaca IEX (`feed=iex`, `adjustment=all`), approved for this development run only.
- Snapshot: `63efd37cabd093594d4ee7b245a88e5a503dce6084d0c5ddf9e33e4368f146f4`.
- 157,322 bars across the 101-symbol S&P 100 proxy and SPY; every requested symbol returned bars.
- Requested range: 2000-01-01 through 2026-10-16. Actual per-symbol starts range from 2018-04-25 through 2025-02-13; 93 symbols begin on 2020-07-27.
- Built table: `data/training/px-v1-iex.parquet`, 102,616 rows over 1,016 dates (2021-07-28 through 2025-08-13). Its last label exits 2025-09-15, before the 2025-09-16 holdout; it contains no holdout decision dates or outcomes.
- IEX is a single-venue feed. Do not use this model or these results for live trading decisions.

## T7 blocker

The required ±0.02 T7 fixture check fails against the specified real-return target under both authorized label scopes:

- `train-only`: mean test rank IC +0.1592 over 12 monthly dates.
- `all-splits`: mean test rank IC +0.1669 over 12 monthly dates.
- In the all-splits run, the permuted training labels have only +0.0048 mean monthly IC with the planted feature, while the resulting model predictions correlate +0.1734 with that feature in the test partition. This is consistent with a feature selected by chance from randomized training data that remains genuinely predictive of test returns in this fixture.
- Scoring those predictions against a separately shuffled test label gives −0.0077, but that is diagnostic only and does not replace T7's specified `raw_forward_return` target.

The pass threshold has not been relaxed. The SIP real-data walk-forward placebo also fails under both tested scopes (results above). This is not by itself evidence of leakage, but together with the deterministic fixture failure it leaves T7 unresolved. The owner explicitly requested the final model and holdout score; that run is documented above. T7 remains formally failed and needs an agreed acceptance rule before future model-selection or deployment claims.

## Original IEX walk-forward result

- Report: `runs/training/pxv1-63efd37cabd0/report.md` and `report.json`.
- The only available test fold is 2025-01-02 through 2025-08-13 (seven month-end observations), after training on 2021-07-28 through 2023-11-28 and validating through 2024-11-27.
- The model's mean monthly rank IC is +0.0121 (t-stat +0.08); 4 of 7 observations are positive.
- At 1× costs, LightGBM returned +1.6% (Sharpe 0.25), versus +4.8% for 12-1 momentum (Sharpe 0.59), +14.0% for equal-weight buy-and-hold (Sharpe 1.18), and +10.4% for SPY (Sharpe 0.85).
- At 2× costs, LightGBM returned +1.4%, momentum +4.7%, equal-weight +13.9%, and SPY +10.4%.
- Equal-weight held 100 available symbols and excluded SNDK because its first bar was after the test start. These short-window numbers are development measurements on single-venue IEX data, not evidence of deployable alpha.

## Final holdout result (evaluated once)

- The locked evaluation ran once for `pxv1-63efd37cabd0`, covering 2025-09-16 through 2026-09-16. The registry contains both `holdout_started` and `holdout_result` records; do not rerun it.
- This window overlaps the v0 momentum smoke test already seen (2025-10-17 through 2026-10-16). That result was not used to select the LightGBM configuration, but this is not a completely unseen period for the project.
- The holdout contains 25,452 symbol-date rows and 12 monthly IC observations. LightGBM mean rank IC is +0.0736 (t-stat +0.90); 7 of 12 monthly observations are positive.
- At 1× costs, LightGBM returned +46.4% (Sharpe 2.53), versus +28.7% for 12-1 momentum (Sharpe 1.79), +44.2% for equal weight (Sharpe 1.97), and +15.5% for SPY (Sharpe 1.19). At 2× costs, returns were +46.0%, +28.5%, +44.2%, and +15.5%, respectively.
- This single-year result is concentrated: the five largest stock contributions are SNDK, MU, DELL, AMD, and LRCX, about $324 combined from $464 final P&L. The stronger holdout follows a weak 2025 walk-forward fold, so treat it as an unstable result needing forward validation, not proof of persistent alpha.
- Report: `runs/training/pxv1-63efd37cabd0/report.md` and `report.json`.

## Validation

- `uv run --offline pytest -q -k 'not test_t7_training_label_placebo_has_no_test_signal'`: 46 passed, 1 deselected in 98.74 seconds.
- `uv run --offline mypy src/ws`: passed, 46 source files.
- Targeted Ruff checks and `git diff --check`: passed.
- Focused late-listing baseline tests: 2 passed. The strict T7 fails under both scopes above and was separately reproduced.

## Workflow commands used

```bash
uv run python -m ws.pipeline_v0.cli download-bars --start 2000-01-01 --end 2026-10-16 --store data/pit-bars-iex --feed iex
uv run python -m ws.training.cli build-table --store data/pit-bars-iex --out data/training/px-v1-iex.parquet
uv run python -m ws.training.cli walk-forward --table data/training/px-v1-iex.parquet --out runs/training
uv run python -m ws.training.cli fit-final --table data/training/px-v1-iex.parquet
uv run python -m ws.training.cli final-eval --model <model-id>  # once
```

This sequence completed for snapshot `63efd37cabd093594d4ee7b245a88e5a503dce6084d0c5ddf9e33e4368f146f4`, producing model `pxv1-63efd37cabd0`. Do not rerun `final-eval` for that model id; its one-time holdout record is locked. The synthetic T7 acceptance issue remains unresolved.
