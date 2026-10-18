# Training v1 — Implementation Spec

Status: confirmed by Yash. Items D1–D7 in Section 12 record the selected decisions.
Owner: Yash. Implementer: coding agent.

## 0. Rules for the implementer

1. **Implement only what this spec says.** No extra features, models, hyperparameters, metrics or dependencies beyond those listed.
2. **If something is ambiguous or missing, stop and ask.** Write the question into `docs/training-v1-questions.md`. Do not record a guess as a "resolved" decision; only Yash resolves questions.
3. **Never read the holdout period (Section 6.3)** in any code path except the single final evaluation command, and never print, plot or log holdout metrics anywhere else.
4. **Every feature formula exists once** and is used by both the training-table builder and the pipeline's `FeatureBuilder` node (Section 5.3).
5. **No network access in tests.** Tests use the synthetic fixture.
6. Allowed new dependency: `lightgbm`. If no wheel installs on Linux aarch64 (the GB10 Spark), stop and ask; do not build from source or substitute a library.

## 1. Goal

Train a LightGBM model that ranks the universe's stocks by expected forward return, evaluate it honestly against simple baselines, register it, and plug it into the existing pipeline as `Alpha(model_id="lgbm@<id>")` through config.

The output of this milestone is a **measurement**: does LightGBM on price/volume features beat 12-1 momentum and equal-weight after costs, out of sample? A "no" is an acceptable, useful result.

## 2. Non-goals

- News, filings, LLM features
- Deep learning models, Kronos or any foundation model
- Hyperparameter search beyond the fixed grid in Section 7.2
- Porting all of Alpha158 (later; v1 uses the feature set in Section 5)
- Short selling, intraday data
- Deflated Sharpe and PBO (research phase; v1 only records the trial count they need)

## 3. Prerequisites (fix before training)

These come from the review of the v0 smoke run.

- **P1 Data feed.** Use Alpaca `feed=sip` for this development run. A request ending at the last completed session succeeded on the free account, so use SIP as the consolidated source and record the feed in manifests and reports. End historical daily-bar requests at a completed session; never silently switch feeds. If SIP becomes unavailable, stop and ask before changing sources.
- **P2 Longer history.** Download from the earliest date Alpaca returns for this universe through the latest completed session. Report the actual earliest date per symbol in `docs/data-coverage.md`.
- **P3 Benchmarks.** Also download `SPY` bars into the store. `SPY` is a benchmark only and is never part of the tradable universe.
- **P4 Baselines in run outputs.** Every backtest writes, next to `equity.csv`, a `baselines.csv` with daily equity for: (a) equal-weight buy-and-hold of the universe symbols that have bars and cost history on the first fill session, (b) `SPY` buy-and-hold, both starting at the same cash on the same first fill session and paying the same cost model on their initial buys. Record any excluded symbols in the CSV and report.
- **P5 Rebalance rule.** Implement the rebalance rule chosen in **D1** in `Execution` (or a new node, if D1 says so).

## 4. Prediction target (label)

- **Decision time:** tick t (session close + 30 minutes), as in v0.
- **Entry:** the next session's open, `open[t+1]`, matching how `SimBroker` fills.
- **Horizon:** `H` sessions **(D2)**. Exit price: `open[t+1+H]`.
- **Raw forward return:** `r = open[t+1+H] / open[t+1] - 1`, on adjusted prices.
- **Label:** cross-sectional percentile rank of `r` among universe symbols that have a valid `r` on date t, scaled to [0, 1]. Ties get the average rank.
- A row has no label (excluded from training, kept for inference) if `open[t+1+H]` is not yet available.

Labels are computed from future prices by design. They live only in the training table and are never passed to any pipeline node.

## 5. Features

### 5.1 Feature set `px-v1`

All computed per symbol from adjusted daily bars with `available_at <= tick t`. `s` = sessions back from the latest visible session (s=0 is today's close). Returns use `close`.

| Name | Formula |
| --- | --- |
| `ret_1d` | close[0]/close[1] − 1 |
| `ret_5d` | close[0]/close[5] − 1 |
| `ret_21d` | close[0]/close[21] − 1 |
| `ret_63d` | close[0]/close[63] − 1 |
| `ret_126d` | close[0]/close[126] − 1 |
| `mom_12_1` | close[21]/close[252] − 1 (existing helper) |
| `vol_21d` | existing helper |
| `vol_63d` | same formula over 63 returns |
| `vol_ratio` | vol_21d / vol_63d |
| `dist_ma_20` | close[0] / mean(close[0..19]) − 1 |
| `dist_ma_200` | close[0] / mean(close[0..199]) − 1 |
| `dist_high_252` | close[0] / max(high[0..251]) − 1 |
| `adv_dollars_20d` | existing helper |
| `volume_ratio` | mean(volume[0..4]) / mean(volume[0..59]) |
| `range_21d` | mean((high − low) / close) over the last 21 sessions |
| `beta_252` | slope of the symbol's 252 daily log returns on SPY's, same sessions |

A feature whose window is not fully available is null. LightGBM handles nulls natively; do not impute.

### 5.2 Cross-sectional normalization

For each date t and each feature, replace the value with its percentile rank among symbols on that date, scaled to [0, 1]. Nulls stay null. Keep `adv_dollars_20d` both raw (`adv_dollars_20d_raw`, for liquidity filtering only, not a model input) and ranked.

### 5.3 One implementation

- `src/ws/features/px_v1.py`: `compute_px_v1(bars: pl.DataFrame, spy_bars: pl.DataFrame) -> pl.DataFrame` returning one row per symbol for the latest visible session.
- The training-table builder calls it once per historical tick on that tick's visible bars (from `BarHistory.visible`).
- `FeatureBuilder` calls the same function when configured with `feature_set="px-v1"`. The existing `"mom-v0"` stays available.
- `FeatureBuilder` must also receive SPY bars: add `SPY` as a benchmark symbol to `SnapshotProvider` output in a separate field `benchmark_bars`, not in `universe` or `bars`.

## 6. Training table and splits

### 6.1 Table

`src/ws/training/table.py` builds one Parquet file per data snapshot: one row per (date t, symbol) with columns `date`, `symbol`, all `px-v1` features (ranked), `adv_dollars_20d_raw`, `label`, `raw_forward_return`.

- Dates: every XNYS session from the first date where at least 80% of universe symbols have a non-null `mom_12_1`, through the last pre-holdout session whose label exit is before the holdout starts.
- `build-table` must not build feature or label rows for holdout decision dates. It must also drop any earlier row whose label exit session is on or after the holdout start, so no training label uses a holdout outcome. `fit-final` reads only this pre-holdout table.
- `final-eval` alone constructs the holdout rows and outcomes. It may read pre-holdout bars for feature lookbacks, including the 252-session beta and momentum windows. It evaluates the selected model and backtests it exactly once.
- The table records the data snapshot id and the feature-set version in its Parquet metadata.

### 6.2 Walk-forward folds

Expanding window, one fold per calendar year **(D3)**:

```
fold k:   train = [first_date, Y_k − embargo)   valid = last 12 months of train
          test  = [Y_k, Y_{k+1})
```

- **Purge:** drop training rows whose label window `[t+1, t+1+H]` reaches into the test period.
- **Embargo:** additionally drop the last `H` sessions before the test start from training.
- The validation segment is used only for early stopping and for choosing among the grid in Section 7.2. It is cut from the end of the training window and is purged the same way.
- The first test year is the first calendar year with at least 3 years of training data before it.

### 6.3 Locked holdout

- The holdout is the most recent 12 months of labeled data **(D4)**.
- No fold's test period overlaps the holdout.
- The holdout is evaluated exactly once, by `ws.training.cli final-eval --model <id>`, which refuses to run twice for the same model id (it checks the registry) and refuses to run on a model whose config was changed after any holdout run.

## 7. Model

### 7.1 LightGBM settings

- `objective="regression"` on the percentile label, `metric="l2"`.
- Fixed: `learning_rate=0.05`, `n_estimators=2000` with early stopping (patience 100) on the validation segment, `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=1`, `deterministic=True`, `force_row_wise=True`, `num_threads` fixed per run and recorded.

### 7.2 Grid (the only tuning allowed)

`num_leaves ∈ {15, 31}` × `min_data_in_leaf ∈ {50, 200}` × `lambda_l2 ∈ {0, 10}` = 8 configurations.

- Each configuration is trained with 5 seeds **(D5)**; its prediction is the mean over seeds.
- The configuration is chosen per fold by mean validation rank IC.
- Every trained configuration and seed counts as a trial and is recorded in the registry (Section 9), including ones that were not selected.

## 8. Evaluation

Computed on fold test periods (and, once, on the holdout):

**Prediction quality** (on rebalance dates, see D1):
- Rank IC per date (Spearman between prediction and `raw_forward_return`); report mean, standard deviation, t-stat, and fraction of dates with IC > 0.
- Same metrics for the 12-1 momentum score as a baseline predictor.

**Portfolio** (through the existing v0 pipeline and `SimBroker`, one backtest per fold test year and one for the holdout):
- Strategies: LightGBM, 12-1 momentum, equal-weight buy-and-hold, SPY.
- Report per strategy: total return, annualized return and volatility, Sharpe (daily returns, annualized, rf = 0), max drawdown, turnover (sum of absolute traded notional / mean equity), total costs paid.
- Report each at 1× and 2× the cost model.

**Placebo** (sanity check): rerun one fold with labels randomly permuted within each date. Its mean test rank IC must be within ±0.02 of zero; otherwise the pipeline has a leak and the run fails.

**Monthly reporting and concentration**:
- Include a monthly table of IC and portfolio return with the model, 12-1 momentum, equal-weight buy-and-hold, and SPY results for each test period and the holdout.
- Report the five largest stock contributions and five largest monthly contributions to portfolio return.

All outputs go to `runs/training/<model_id>/report.json` and `report.md` (the Markdown is rendered from the JSON by a template, not written freehand).

## 9. Registry

- Append-only `models/registry.jsonl`, one line per trained (configuration, seed, fold).
- Fields: `model_id`, `created_at`, `data_snapshot_id`, `feature_set`, `label` (`{"horizon": H, "type": "xs_rank_open_to_open"}`), `fold`, `train_range`, `valid_range`, `test_range`, `params`, `seed`, `best_iteration`, `valid_rank_ic`, `test_rank_ic`, `selected` (bool), `git_commit`.
- The selected model for inference is trained on all data before the holdout with the configuration that won most folds (ties: smaller `num_leaves`, then larger `min_data_in_leaf`), 5 seeds averaged. Saved as LightGBM text files under `models/<model_id>/seed_<n>.txt` with a `manifest.json`.
- `Alpha(model_id="lgbm@<model_id>")` loads the manifest and seed files, predicts with `px-v1` features, averages over seeds. It never trains.

## 10. Tests

- **T1 Label alignment.** On a hand-built series, `raw_forward_return` for date t equals `open[t+1+H]/open[t+1] − 1`.
- **T2 Feature point-in-time.** Appending bars with `available_at` after tick t to the store does not change any feature row for date t.
- **T3 Parity.** For 5 random fixture dates, the training-table feature row equals `FeatureBuilder(feature_set="px-v1")` output for the same tick, exactly.
- **T4 Purge and embargo.** For each fold, no training row's label window overlaps the test period, and the last `H` sessions before test start are absent from training.
- **T5 Holdout lock.** No function other than `final-eval` can read dates inside the holdout; `final-eval` refuses a second run for the same model id.
- **T6 Planted signal.** On a fixture where next-period returns depend on one feature plus noise, test rank IC > 0.1.
- **T7 Placebo.** On the same fixture with permuted labels, mean test rank IC is within ±0.02 of zero.
- **T8 Determinism.** Training the same configuration and seed twice gives identical model files.
- **T9 Pipeline integration.** The v0 backtest runs end to end on the fixture with `Alpha(model_id="lgbm@<fixture model>")` and `feature_set="px-v1"`, and replay digests match.

## 11. Commands

```bash
uv run python -m ws.training.cli build-table --store data/pit-bars --out data/training/px-v1.parquet
uv run python -m ws.training.cli walk-forward --table data/training/px-v1.parquet --out runs/training --placebo-label-scope train-only
uv run python -m ws.training.cli fit-final --table data/training/px-v1.parquet --iteration-policy median-fold
uv run python -m ws.training.cli final-eval --model <model_id>     # once
```

The workflow flags have defaults matching the selected v1 decisions and can be overridden for documented experiments.

## 12. Decisions for Yash

- **D1 Rebalance rule.** Proposed: rebalance on the last trading session of each month; between rebalances, trade only to sell positions that fail risk limits. Alternative: daily band rebalancing (trade a name only when its weight is more than 25% away from target). The v0 run's ~14.5 fills per day came from daily rebalancing with a $1 threshold.
- **D2 Label horizon `H`.** Proposed: 21 sessions, matching monthly rebalancing. If D1 changes to daily bands, consider 5.
- **D3 Fold size.** Proposed: one calendar year per test fold.
- **D4 Holdout.** Proposed: the most recent 12 months. The 2025-10-17 → 2026-10-16 window was already used for the momentum smoke run; it was not used to select anything, so it can still serve as the holdout, but note that you have seen momentum's result on it.
- **D5 Seeds.** Proposed: 5 per configuration.
- **D6 Liquidity filter.** Proposed: none for the S&P 100 (all are liquid); keep `adv_dollars_20d_raw` in the table for later universes.
- **D7 Survivorship.** The universe is today's S&P 100 proxy applied to all past dates. Longer history (P2) makes this bias larger. Proposed: accept and label every report "survivorship-biased universe"; fix with point-in-time membership and delisted-stock data in a later milestone.


D1	Monthly rebalance; weekly as a later experiment (confirmed)
D2	21-session label (confirmed)
D3	One year per fold (confirmed)
D4	Last 12 months as holdout; the v0 momentum result has been seen (confirmed)
D5	Seeds 0–4 (confirmed)
D6	No liquidity filter (confirmed)
D7	Accept and label survivorship bias (confirmed)
