# Fundamentals v1 (`pxf-v1`) — Implementation Spec

Status: implementation authorized; D1-D7 and all five implementation defaults accepted by Yash on 2026-10-18. Download and training progress is tracked in `docs/fund-v1-questions.md`.
Owner: Yash. Implementer: coding agent.

## 0. Rules for the implementer

1. **Implement only what this spec says.** No extra features, data sources, models or dependencies.
2. **If something is ambiguous, missing, or a download fails, stop and ask.** Write the question in `docs/fund-v1-questions.md`. Do not substitute another source, guess a tag name, or mark a question "resolved"; only Yash resolves questions.
3. **No `final-eval` in this milestone.** The 2025-09-16 → 2026-09-16 holdout has been viewed. Evaluation here is walk-forward only (Section 8). Forward paper trading is the next unseen test.
4. **Every feature formula exists once** and is shared by the training-table builder and `FeatureBuilder`, as in training v1.
5. **Downloaded data is never committed.** Everything under `data/` stays git-ignored. Raw downloads are immutable once written.
6. **No network access in tests.** Tests use small checked-in JSON/CSV fixtures under `tests/fixtures/`.

## 1. Goal

Add published, replicated stock characteristics built from free data (SEC EDGAR fundamentals and existing Alpaca prices), and measure whether they improve the LightGBM ranker over `px-v1` on the same walk-forward folds.

Definitions follow the original papers cited per feature, as catalogued by Chen & Zimmermann (Open Source Asset Pricing) and Jensen, Kelly & Pedersen. Where this spec simplifies a definition for free data, it says so.

## 2. Prerequisite fixes (do these first)

- **F1 Permutation placebo (replaces T7).**
  - Run the placebo `N = 50` times, each with a different permutation seed (labels shuffled within each date, train rows only).
  - Use the selected real model's per-seed best iteration for the corresponding placebo seed. Do not early-stop each placebo against the unpermuted validation labels; that would use the validation signal to select a null model's complexity.
  - Record the mean test rank IC of each run.
  - **Pass** if the mean of the 50 means is within ±0.02 of zero.
  - Report the real model's mean test rank IC and its **percentile within the 50 placebo means**. Report it; do not gate on it.
  - Placebo fits count as trials in the registry.
- **F2 Early stopping on rank IC.** Replace l2 early stopping with mean daily Spearman rank IC on the validation segment (custom `feval`, higher is better). Keep patience 100 and `n_estimators=2000`. Record `best_iteration` per fold.
- **F3 Holding buffer.** In `PortfolioConstruction`: keep a currently held symbol if its rank is within the top `buffer_rank` (default 35); fill remaining slots up to 25 holdings from the highest-ranked non-held symbols. Report turnover with and without the buffer.

## 3. Data downloads

All downloads go through `uv run python -m ws.data.cli <command>`. Each command writes to a new dated directory and a `manifest.json` with URL, download time, byte size and sha256 of each file.

### 3.1 SEC EDGAR (free, no API key)

- **User-Agent is required by the SEC.** Read it from the environment variable `SEC_USER_AGENT` (format: `"Full Name email@example.com"`), also accepted from `.env`. Refuse to run if it is missing. Never commit it.
- **Rate limit:** at most 10 requests per second (SEC fair-access policy). The bulk files below need only a few requests.
- Commands and sources:

| Command | URL | Output |
| --- | --- | --- |
| `sec-tickers` | `https://www.sec.gov/files/company_tickers.json` | `data/sec/raw/<date>/company_tickers.json` |
| `sec-bulk-facts` | `https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip` | `data/sec/raw/<date>/companyfacts.zip` |
| `sec-bulk-submissions` | `https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip` | `data/sec/raw/<date>/submissions.zip` |

- `companyfacts.zip` and `submissions.zip` are large (on the order of gigabytes). Stream them to disk; do not load them whole into memory.
- **Extract only the universe's companies**: map each universe ticker to its CIK with `company_tickers.json`, then read only `CIK##########.json` for those CIKs from each zip.
- A ticker not found in `company_tickers.json` is listed in `docs/fund-v1-coverage.md` and gets null fundamentals. Do not look it up elsewhere.
- Multiple tickers can share one CIK (e.g. two share classes). They share fundamentals; see 5.2 for market equity.

### 3.2 Ken French Data Library (free, for validation only)

Base URL: `https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/`

| Command | File | Use |
| --- | --- | --- |
| `ff-industries` | `Siccodes12.zip` | SIC → Fama-French 12 industry map |
| `ff-factors` | `F-F_Research_Data_5_Factors_2x3_CSV.zip` | monthly HML, RMW, CMA, SMB for validation |
| `ff-momentum` | `F-F_Momentum_Factor_CSV.zip` | monthly UMD for validation |

If any URL returns an error, stop and ask (rule 2).

### 3.3 Prices

Reuse the existing Alpaca SIP snapshot. No new price download.

## 4. Point-in-time fundamentals table

`src/ws/fundamentals/store.py` builds `data/sec/facts/<snapshot>.parquet`, one row per XBRL fact for universe CIKs, with columns:

`cik, tag, unit, value, period_start (nullable), period_end, form, fiscal_year, fiscal_period, accession, filed, available_at`

- Keep only facts with `form` in {`10-K`, `10-K/A`, `10-Q`, `10-Q/A`} and unit `USD` (or `shares` for share counts).
- **`available_at` = close of the first XNYS session strictly after `filed`, plus 15 minutes.** `filed` is a date without a time, and filings can arrive after the close, so this is the conservative choice.
- **Restatements:** keep every fact. At decision time t, for a given (tag, period_start, period_end), use the fact with the latest `available_at <= t`.
- **Annual value of a flow tag at time t:** among visible facts with `form` in {10-K, 10-K/A} and duration 350–380 days, take the one with the latest `period_end` (restatement rule applies).
- **Annual value of a stock tag (balance sheet) at time t:** the instant fact at that same latest fiscal year end, visible at t.
- **Staleness:** if the latest visible annual `period_end` is more than 18 months before t, all fundamentals for that company are null at t.
- **SIC code:** from that company's `submissions` JSON field `sic`. Map to FF12 with `Siccodes12`; unmapped → `Other`.

### 4.1 Tag priority lists

Use the first tag in each list that has a visible value. Never sum or mix tags within one value.

| Item | Tags, in priority order |
| --- | --- |
| Book equity | `StockholdersEquity`, `StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest` |
| Net income | `NetIncomeLoss`, `ProfitLoss` |
| Operating cash flow | `NetCashProvidedByUsedInOperatingActivities`, `NetCashProvidedByUsedInOperatingActivitiesContinuingOperations` |
| Total assets | `Assets` |
| Revenue | `Revenues`, `RevenueFromContractWithCustomerExcludingAssessedTax`, `SalesRevenueNet` |
| Cost of revenue | `CostOfRevenue`, `CostOfGoodsAndServicesSold`, `CostOfGoodsSold` |
| Gross profit | `GrossProfit` (used if present; else Revenue − Cost of revenue) |
| Shares outstanding | `dei:EntityCommonStockSharesOutstanding` |

All tags are in the `us-gaap` taxonomy unless prefixed. Coverage per item and company goes in `docs/fund-v1-coverage.md`.

## 5. Features

### 5.1 New features

All ranked cross-sectionally per date to [0, 1] like `px-v1`; nulls stay null. `ME` = market equity (5.2). Fundamentals use annual values (Section 4).

| Name | Definition | Source paper |
| --- | --- | --- |
| `log_me` | log(ME) | Banz (1981), size |
| `bm` | book equity / ME; null if book equity ≤ 0 | Fama & French (1992) |
| `ep` | net income / ME | Basu (1977) |
| `cfp` | operating cash flow / ME | Lakonishok, Shleifer & Vishny (1994) |
| `gp_at` | gross profit / total assets | Novy-Marx (2013) |
| `roe` | net income / book equity; null if book equity ≤ 0 | Haugen & Baker (1996) |
| `asset_growth` | assets(FY) / assets(FY−1) − 1; FY−1 = instant 350–380 days before FY end | Cooper, Gulen & Schill (2008) |
| `ind_mom` | equal-weight mean of `ret_126d` over universe symbols in the same FF12 industry (including itself); null if fewer than 3 such symbols | Moskowitz & Grinblatt (1999), simplified to the universe |
| `resid_mom` | see 5.3 | Blitz, Huij & Martens (2011), simplified |
| `ivol_21d` | std of the last 21 daily residuals from regressing the symbol's daily log returns on SPY's over those 21 sessions | Ang, Hodrick, Xing & Zhang (2006), SPY instead of FF3 |
| `max_ret_21d` | maximum daily simple return over the last 21 sessions | Bali, Cakici & Whitelaw (2011) |

### 5.2 Market equity

- `shares` = sum of all `dei:EntityCommonStockSharesOutstanding` values in the latest visible filing (same accession), across share classes.
- `ME = shares × close` of the symbol on decision date t.
- Known limitation: for multi-class companies with different class prices, this is an approximation. List affected tickers in `docs/fund-v1-coverage.md`.

### 5.3 Residual momentum (simplified)

- Monthly returns from month-end closes for the symbol and SPY.
- At decision month m: regress the symbol's monthly returns on SPY's over months m−36 … m−1 (OLS with intercept). Null if fewer than 36 months.
- `resid_mom` = (sum of residuals over months m−12 … m−2) / (std of residuals over months m−12 … m−2).

### 5.4 Feature set

`pxf-v1` = all `px-v1` features + the 11 features above. `FeatureBuilder(feature_set="pxf-v1")` and the table builder use the same functions. `px-v1` stays unchanged.

## 6. Training table

- Same label, dates, purge, embargo, folds and holdout boundary as training v1.
- New table `data/training/pxf-v1-sip.parquet`; records the price snapshot id and the fundamentals snapshot id in its metadata.

## 7. Model and grid

Identical to training v1 (same 8 configurations, seeds 0–4), with the F2 early-stopping change. Train both `px-v1` and `pxf-v1` with the F2 change so the comparison is like for like.

## 8. Evaluation (walk-forward only)

For each fold and in total, for `px-v1` and `pxf-v1`:

1. **Rank IC:** mean, t-stat, fraction positive (as training v1).
2. **Long-short spread:** each rebalance date, sort symbols by prediction into quintiles; report the mean `raw_forward_return` of the top quintile minus the bottom quintile, its mean over dates and t-stat. This needs no shorting in the simulator.
3. **Long-only portfolio** through `SimBroker` with the F3 buffer, against equal-weight and SPY, at 1× and 2× costs.
4. **Permutation placebo (F1)** for `pxf-v1`, and the real model's percentile.
5. **Feature importance:** LightGBM gain importance per fold, averaged.
6. **Concentration and monthly tables** as training v1.

### 8.2 Cost-history eligibility

At a close-time decision, a symbol is eligible for ranking into the portfolio only when 20-session ADV and 21 close-to-close returns are available from bars known by that decision. This requires 22 daily closes and ensures the next-open fill has pre-fill cost inputs. `SimBroker` independently rechecks those inputs at the fill open and cancels and records any order that lacks them; it never substitutes an estimated zero cost.

### 8.1 Construction check (reported, not gated)

Using the universe only, build monthly long-short returns (top third minus bottom third, equal weight) for `bm`, `gp_at`, `asset_growth` (bottom minus top) and `mom_12_1`. Report the correlation of each with Fama-French HML, RMW, CMA and UMD over the same months. Low correlation is expected with 100 stocks; the table is for Yash to judge whether construction looks right.

## 9. Tests

- **T1 Filing-date availability.** A fact filed on date d is invisible to a tick on d and visible to the tick after the next session close.
- **T2 Restatement.** Two facts for the same period with different `filed` dates: a tick between them sees the first value, a later tick sees the second.
- **T3 Annual selection.** Given 10-Q and 10-K facts, the annual value comes from the 10-K with the latest visible period end; a 10-Q value is never used as annual.
- **T4 Tag priority.** When the first tag is absent, the second is used; values from two tags are never mixed.
- **T5 Staleness.** A company whose latest annual period end is more than 18 months old has null fundamentals.
- **T6 Feature formulas.** Hand-computed values for every new feature on fixture data match exactly.
- **T7 Parity.** Training-table rows equal `FeatureBuilder(feature_set="pxf-v1")` output on 5 fixture dates.
- **T8 Permutation placebo** on the planted-signal fixture: mean of 50 placebo means within ±0.02; real model above the 95th percentile.
- **T9 Buffer.** A held symbol ranked 30th is kept; one ranked 40th is sold.
- **T10 Downloader.** Missing `SEC_USER_AGENT` refuses to run; a mocked download writes the manifest with the correct sha256.

## 10. Commands

```bash
export SEC_USER_AGENT="Your Name your@email"   # or put it in .env
uv run python -m ws.data.cli sec-tickers
uv run python -m ws.data.cli sec-bulk-facts
uv run python -m ws.data.cli sec-bulk-submissions
uv run python -m ws.data.cli ff-industries
uv run python -m ws.data.cli ff-factors
uv run python -m ws.data.cli ff-momentum
uv run python -m ws.fundamentals.cli build --raw data/sec/raw/<date> --out data/sec/facts
uv run python -m ws.training.cli build-table --feature-set pxf-v1 --store data/pit-bars --facts data/sec/facts/<snapshot>.parquet --out data/training/pxf-v1-sip.parquet
uv run python -m ws.training.cli walk-forward --table data/training/pxf-v1-sip.parquet --out runs/training
uv run python -m ws.training.cli walk-forward --table data/training/px-v1-sip.parquet --out runs/training   # rerun with F2 for comparison
```

## 11. Decisions for Yash

- **D1 Annual fundamentals only.** Quarterly trailing-twelve-month values are more current but harder to build correctly from XBRL (Q4 is not reported separately). Proposed: annual for v1.
- **D2 Filing-date lag.** Proposed: available after the next session close following `filed`.
- **D3 Placebo runs.** Proposed: 50.
- **D4 Holding buffer.** Proposed: keep holdings ranked within the top 35.
- **D5 Long-short.** Proposed: measured as a quintile spread in evaluation only; no shorting in the simulator yet.
- **D6 No holdout run.** Proposed: walk-forward only; paper trading is the next unseen test.
- **D7 Survivorship.** Unchanged: today's S&P 100 proxy applied to past dates, labeled in every report.
