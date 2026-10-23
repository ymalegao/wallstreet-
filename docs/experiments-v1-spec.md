# Experiments v1 — Spec (draft)

Status: **draft, not authorized.** Written 2026-10-22 for Yash to review. Every item marked **Proposed** is a suggestion, not a decision. Open questions are in `docs/experiments-v1-questions.md`; only Yash resolves them.
Owner: Yash. Implementer: coding agent.

## 0. Rules for the implementer

1. **Implement only what this spec says**, one experiment at a time, in order (E1 → E2 → E3). Do not start an experiment until Yash has resolved its questions.
2. **If something is ambiguous, missing, or a download fails, stop and ask** in `docs/experiments-v1-questions.md`. Never mark a question resolved yourself.
3. **No `final-eval`.** The 2025-09-16 → 2026-09-16 holdout has been viewed. Walk-forward only. No training label may end on or after 2025-09-16.
4. **Every feature formula exists once**, shared by the table builder and `FeatureBuilder` (as in training v1 and fund-v1).
5. **Downloaded data and model weights are never committed.** Raw downloads are immutable, with sha256 manifests.
6. **No network access in tests.** Use small fixtures under `tests/fixtures/`.
7. **Every run gets its own model id** and never overwrites an earlier run. Every fit, including placebo fits, counts as a trial in `models/registry.jsonl`.
8. **Keep the survivorship label** (today's S&P 100 proxy applied to the past) in every report.

## 1. Goal

Answer three questions, in this order:

| # | Question | Why it matters |
|---|---|---|
| E1 | Does `px-v1` still look like a real (if small) signal once it faces the same, fair placebo test as `pxf-v1`? | Its 96th-percentile result came from an easier, since-fixed test. Everything after builds on whether this baseline is real. |
| E2 | Does rebalancing **weekly** with a matching 5-session label do better than monthly, after costs? | Shorter horizons give more independent bets per year, and news (E3, later) is known to matter over days, not months. |
| E3 | Does adding **news features scored by a language model** improve the monthly ranker? | First test of the LLM-as-a-node idea, at the existing monthly horizon, to see whether news helps there or only at shorter horizons. |

Not in this spec: paper trading, a larger universe, a sector-relative label, the LLM-as-agent tool interface. Each gets its own spec.

## 2. Comparison rules (apply to all three)

So that "better" means the same thing everywhere, decide these **before** seeing results. Fixing the success test in advance prevents moving the goalposts after the fact (Harvey, Liu & Zhu 2016, "…and the Cross-Section of Expected Returns").

- Same expanding annual folds (2021–2025 test years), same 8-config × 5-seed grid, F2 rank-IC early stopping, 50 train-only placebos with fixed iterations (F1), same cost model at 1× and 2×, same 25-holding / top-35 buffer unless an experiment says otherwise.
- **Proposed success test (D-E0):** an experiment counts as an improvement over its baseline only if **all** of these hold:
  1. the real model sits at or above the **95th percentile** of its placebo distribution overall;
  2. mean test rank IC is positive in **at least 4 of 5** folds;
  3. the long-only portfolio beats equal-weight buy-and-hold in total return across folds **at 2× costs**.
  Anything weaker is reported as "inconclusive", not as a win.
- Every comparison is reported side by side in one table: rank IC (mean, t-stat), placebo percentile, long-short quintile spread, long-only vs equal-weight and SPY at 1× and 2×, turnover, costs paid.
- **Reporting fix (Q1 in `fund-v1-questions.md`):** the current "Placebo pass" field only checks that placebos average near zero. Reports for these experiments should label that field as a **leakage check**, and separately state whether the real model cleared the 95th percentile. Exact wording waits on Yash's answer to Q1.

## 3. E1 — Fair `px-v1` rerun

**Status:** already running (authorized 2026-10-22) as model id `pxv1f3-60a47f8a0fcb`, on the current code. Log: `runs/training/pxv1f3-60a47f8a0fcb.log`.

**What changed since `pxv1f2-60a47f8a0fcb`:**
- Placebo fits now use the real model's per-seed tree counts, instead of early-stopping themselves on the unshuffled validation labels.
- Portfolio eligibility now needs 22 visible closes and 20-session ADV.

The model itself (data, features, grid, early stopping) is unchanged, so mean test rank IC should come out ≈ 0.0231 again. The placebo percentile and portfolio numbers may move.

**Deliverable:** `docs/e1-px-vs-pxf.md`, a one-page, plain-language side-by-side of `pxv1f3` vs `pxfv1` (Section 2 table), plus a short note on what each number means.

**Proposed reading of the outcome (D-E1):**
- `pxv1f3` passes the D-E0 test → `px-v1` stays the baseline for E2 and E3.
- It doesn't → no feature set has shown an edge on this universe; E2 and E3 still run, because they test different things (horizon, new information), not more tuning on the same features.

## 4. E2 — Weekly rebalancing with a 5-session label

### 4.1 What changes

| Piece | Monthly (today) | Weekly (E2) |
|---|---|---|
| Label | percentile rank of open(t+1) → open(t+22) return | percentile rank of open(t+1) → open(t+6) return, i.e. 5 sessions |
| Rebalance day | last XNYS session of the month | **Proposed:** last XNYS session of each ISO week (usually Friday) |
| Purge and embargo | 21 sessions | 5 sessions (follows the label horizon; `build_folds` already takes `horizon`) |
| Rank IC and quintile spread | measured on month-end dates (~12 per year) | measured on weekly rebalance dates (~52 per year) |
| Features | `px-v1`, unchanged | `px-v1`, unchanged (see Q-E2.1) |
| Model id | `pxv1f3-…` | `pxv1w5-<snapshot>` |
| Table | `data/training/px-v1-sip.parquet` | `data/training/px-v1-w5-sip.parquet` |

Everything else is identical: folds, grid, early stopping (mean daily rank IC on validation), placebos, cost model, holdings, buffer.

### 4.2 Code changes needed

These are the places that hard-code monthly today, found by reading the code on 2026-10-22:

- `src/ws/training/table.py`: `LABEL_HORIZON = 21`, and `build_table` raises if `horizon != 21`. Allow `horizon ∈ {5, 21}` and record it in table metadata (already a field).
- `src/ws/pipeline_v0/nodes.py` `Execution`: `rebalance_rule` only accepts `"monthly"`. Add `"weekly"` (definition from 4.1).
- `src/ws/training/metrics.py`: `rank_ic_by_date` and the quintile-spread function filter with `is_month_end`. Replace with the rebalance rule recorded in the table metadata.
- `src/ws/training/evaluation.py`: sets `rebalance_rule = "monthly"` in the backtest config. Take it from table metadata.
- `src/ws/training/trainer.py`: model id prefix `pxv1w5-` when horizon is 5.
- `src/ws/training/cli.py`: `build-table --horizon 5`.

### 4.3 Extra reporting for E2

- Turnover and costs paid per year, weekly vs monthly. Weekly will trade more; the question is whether the signal pays for it.
- The same comparison with the buffer **off** (top-25 only), to show how much the buffer is saving (as F3 in fund-v1).
- The monthly table of returns, so weekly and monthly can be compared month by month.

### 4.4 Tests

- **T-E2.1:** a 5-session table's label equals the hand-computed open(t+1) → open(t+6) return rank on fixture data, and no label exits on or after the holdout start.
- **T-E2.2:** `rebalance_rule="weekly"` rebalances on the last session of each ISO week, including a holiday week where that day is Thursday.
- **T-E2.3:** with horizon 5, no training row's `label_exit_date` falls inside the 5-session embargo before validation or test.
- **T-E2.4:** the monthly path gives byte-identical results to before (regression test on the existing fixture).

### 4.5 Commands

```bash
uv run python -m ws.training.cli build-table --feature-set px-v1 --horizon 5 --store data/pit-bars --out data/training/px-v1-w5-sip.parquet
uv run python -m ws.training.cli walk-forward --table data/training/px-v1-w5-sip.parquet --out runs/training
```

## 5. E3 — News features scored by a language model, monthly rebalancing

### 5.1 Shape of the experiment

Of the three ways people combine news with a stock model, this uses the most common one: the language model turns each headline into a number, and those numbers become new columns that LightGBM is **retrained** with. The language model itself is not trained on returns.

```
Alpaca News API ──► raw news (immutable JSONL) ──► point-in-time news store
                                                        │  (available_at per article)
                                                        ▼
                                    NewsScorer (local LLM, offline batch)
                                                        │  score per (article, symbol)
                                                        ▼
                         news features per (date, symbol) ──► FeatureBuilder / table
                                                        ▼
                                    LightGBM walk-forward (same protocol)
```

**LLM as a node:** the scorer sits behind a small `NewsScorer` interface (`score(headline, summary, symbol) -> float | None`), so a different model can be swapped in without touching the rest. For backtests the scores are computed once in a batch and stored point-in-time, like the SEC fundamentals store. Later, in daily paper trading, the same scorer function would run inside the daily graph. Whether that's the design is Q-E3.1.

### 5.2 Step E3.0 — Download and coverage audit (gate)

Nothing is scored until Yash has read the coverage report.

- **Source:** Alpaca News API, `GET https://data.alpaca.markets/v1beta1/news`. Benzinga is the only provider. History is advertised back to 2015.
  - Parameters: `symbols`, `start`, `end`, `limit` (max 50), `sort`, `include_content`, `page_token`.
  - Fields: `id, headline, author, created_at, updated_at, summary, content, symbols, source, url`.
- **Rate limit:** 200 calls/minute on the free plan (`X-RateLimit-Limit: 200`, observed 2026-10-22). Read `X-RateLimit-*` headers and back off on HTTP 429.
- **Command:** `uv run python -m ws.data.cli alpaca-news --start 2015-01-01 --end 2026-10-16`. It writes `data/news/raw/<date>/` (one JSONL file per symbol) plus `manifest.json`.
  - Uses the existing Alpaca credentials from `.env`.
  - Proposed: headline and summary only, no `content` (Q-E3.3).
- **Coverage report** `docs/news-v1-coverage.md`: articles per symbol per year; share of (date, symbol) pairs with any article in the last 21 sessions; median articles per article-bearing day; share of articles tagged with more than 5 symbols; earliest date per symbol.
  - The report also lists any year that looks thin. An Alpaca forum thread reports missing 2016 history, so this must be measured, not assumed.

### 5.3 Point-in-time rule

- An article is visible to a tick at time `now` only if **`created_at <= now`** (both UTC; the tick is the session close + 30 minutes, as today).
- `updated_at` is ignored for visibility. If the API returns a later-edited headline, that edit could leak information; how to handle edited articles is Q-E3.4.
- Raw files are immutable; the news store records the raw snapshot sha256, and the training table records the news snapshot id (as fund-v1 does for SEC).

### 5.4 The language-model cutoff problem (most important design choice)

A language model trained on text through year Y "knows" what happened to companies before Y. Scoring a 2018 headline with a 2025 model quietly uses the future and makes the backtest look better than reality. Options from the literature:

| Option | How it works | Clean backtest? | Source |
|---|---|---|---|
| **A. Point-in-time model vintages** | Score each headline with a model trained only on text **before** that headline's year. ChronoGPT-instruct has yearly vintages (e.g. `chrono-gpt-instruct-v1-20161231` … `-20241231`), ~1.5B parameters, MIT license. A headline dated in year Y is scored by the vintage ending Dec 31 of Y−1. | Yes, by construction | He, Lv, Manela & Wu (2025), "Chronologically Consistent Large Language Models" |
| **B. Modern model + company names hidden** | Use a strong current open model; replace company names and tickers with "the company". Reduces, does not remove, the model's memory of what happened. | Partly | Glasserman & Lin (2023), "Assessing Look-Ahead Bias in Stock Return Predictions Generated by GPT Sentiment Analysis" |
| **C. Modern model, post-cutoff dates only** | Only use news after the model's training cutoff. | Yes, but the history is too short for 5 walk-forward folds | Lopez-Lira & Tang (2023) |

- **Yash, 2026-10-22:** for now, use a model already on the Spark (JEV-9B, Qwen3 or Gemma), i.e. options B/C rather than A. E3 backtest results are therefore labelled "possibly look-ahead-biased"; the clean test is post-cutoff dates and paper trading. Whether to anonymize, and whether to run A later as a check, remain open (Q-E3.2, Q-E3.11).
- **Caveat on A:** Lopez-Lira & Tang (Section 7.1) found that small models (GPT-1, GPT-2, BERT) scored headlines much less accurately, their scores were not significantly correlated with later drift, and the resulting strategies had negative Sharpe ratios. He et al. report ChronoGPT competitive with Llama 3.1 in their asset-pricing test, but that is from embeddings fed to a regression, not prompting. So how a 1.5B model produces a score is an open question (Q-E3.5).
- **Verify before use:** confirm which yearly vintages exist (the Hugging Face listing seen on 2026-10-22 showed 2014 and 2016–2024; 2015 was not visible), that the model loads on the Spark, and record each vintage's revision hash.

### 5.5 Scoring

**Proposed default (Q-E3.5):** the Lopez-Lira & Tang prompt per (headline, symbol):

> "Forget all your previous instructions. Pretend you are a financial expert. You are a financial expert with stock recommendation experience. Answer 'YES' if good news, 'NO' if bad news, or 'UNKNOWN' if uncertain in the first line. Then elaborate with one short and concise sentence on the next line. Is this headline good or bad for the stock price of _company_name_ in the short term? Headline: _headline_"

(Wording as given in the paper, Section 5. "Short term" is kept as published even though E3 is monthly; see Q-E3.5.)

- Map YES → +1, NO → −1, UNKNOWN → 0. Greedy decoding (temperature 0).
- Record the full prompt template hash, model id, vintage and revision, and the raw output for every scored article.
- **Alternative (Q-E3.5):** use the model's embedding of the headline and fit a small regression (ridge) on each fold's training rows only. This is closer to how He et al. got their results, but it is a second trained model and needs its own purge and placebo handling.
- **Multi-symbol articles:** proposed: score once per tagged symbol; drop articles tagged with more than 5 symbols, which are usually market round-ups (Q-E3.6).

### 5.6 News features

Computed per (date, symbol) from articles visible at the tick, then ranked per date like every other feature. Days with no articles give null, which LightGBM handles natively.

| Name | Definition | Source |
|---|---|---|
| `news_score_21d` | mean article score over the last 21 sessions | Lopez-Lira & Tang (2023), aggregated to the monthly horizon |
| `news_score_5d` | mean article score over the last 5 sessions | same |
| `news_count_21d` | log(1 + number of articles in the last 21 sessions) | Fang & Peress (2009), "Media Coverage and the Cross-Section of Stock Returns" |
| `news_neg_frac_21d` | share of articles scored −1 in the last 21 sessions | Tetlock, Saar-Tsechansky & Macskassy (2008), "More Than Words" |

Why monthly might still work: Chan (2003), "Stock Price Reaction to News and No-News", finds prices keep drifting for months after bad-news headlines. The strongest LLM evidence is for next-day returns, though. If E3 is null at monthly, that is informative and points to a weekly E3 variant, not a failure of the setup.

### 5.7 Feature set, table, model id

- **Proposed:** `pxn-v1` = `px-v1` + the 4 news features (Q-E3.7 asks whether to also run `pxf-v1` + news).
- Table: `data/training/pxn-v1-sip.parquet`, 21-session label, records price, news and scorer snapshot ids.
- Model id: `pxnv1-<price snapshot[:10]>-<news snapshot[:10]>`.
- Report gain importance for the 4 news features per fold.

### 5.8 Compute

- ChronoGPT-instruct is ~1.5B parameters and fits easily in the Spark's 128 GB.
- Total article count is unknown until E3.0. The scoring time estimate goes in the coverage report, before scoring starts.
- Inference engine: proposed Hugging Face `transformers` (the model uses the GPT-2 tokenizer); vLLM only if it is confirmed to load these weights (Q-E3.8).

### 5.9 Tests

- **T-E3.1 Visibility:** an article with `created_at` one second after the tick is invisible to it and visible to the next tick.
- **T-E3.2 Vintage:** a headline dated 2019-03-01 is scored by the vintage ending 2018-12-31, never a later one; scoring refuses if that vintage is missing.
- **T-E3.3 Parse:** fixture model outputs "YES…", "NO…", "UNKNOWN…", and garbage map to +1, −1, 0, null.
- **T-E3.4 Features:** hand-computed values for all four features on fixture articles.
- **T-E3.5 Parity:** table rows equal `FeatureBuilder(feature_set="pxn-v1")` output on 5 fixture dates.
- **T-E3.6 Downloader:** a mocked paginated download writes the manifest with correct sha256 and handles a 429 with backoff.
- Tests use a stub scorer, never real model weights.

### 5.10 Commands (after E3.0 is approved)

```bash
uv run python -m ws.data.cli alpaca-news --start 2015-01-01 --end 2026-10-16
uv run python -m ws.news.cli build --raw data/news/raw/<date> --out data/news/store
uv run python -m ws.news.cli coverage --store data/news/store/<snapshot>.parquet --out docs/news-v1-coverage.md
# stop: Yash reviews coverage
uv run python -m ws.news.cli score --store data/news/store/<snapshot>.parquet --scorer chronogpt-instruct --out data/news/scores
uv run python -m ws.training.cli build-table --feature-set pxn-v1 --store data/pit-bars --news data/news/scores/<snapshot>.parquet --out data/training/pxn-v1-sip.parquet
uv run python -m ws.training.cli walk-forward --table data/training/pxn-v1-sip.parquet --out runs/training
```

## 6. References

- Gu, Kelly & Xiu (2020), "Empirical Asset Pricing via Machine Learning."
- Harvey, Liu & Zhu (2016), "…and the Cross-Section of Expected Returns."
- Lopez-Lira & Tang (2023), "Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models."
- He, Lv, Manela & Wu (2025), "Chronologically Consistent Large Language Models," arXiv:2502.21206. Models: huggingface.co/manelalab.
- Glasserman & Lin (2023), "Assessing Look-Ahead Bias in Stock Return Predictions Generated by GPT Sentiment Analysis," arXiv:2309.17322.
- Tetlock, Saar-Tsechansky & Macskassy (2008), "More Than Words: Quantifying Language to Measure Firms' Fundamentals."
- Fang & Peress (2009), "Media Coverage and the Cross-Section of Stock Returns."
- Chan (2003), "Stock Price Reaction to News and No-News: Drift and Reversal after Headlines."
- Alpaca News API reference: docs.alpaca.markets/reference/news-3.
