# 01 — Research Spec: Event-Driven Trading System on Local Compute

**Status:** Research phase (precedes design spec `02-design-spec.md`)
**Date:** 2026-10-04
**Scope:** What the research says about where an edge can come from, a fact-check of the original concept doc, what the hardware can do, and the evaluation protocol plus pass/fail metrics the system has to meet before real money is used.

> **Update (2026-10-04, after review):** decisions are recorded in `02-design-spec.md` §0. **GDELT and prediction markets are dropped entirely.** Open-weight JEV replicas exist (JEV-9B, OpenJev, Von), so the System-1 filter uses one of them locally instead of a home-built logit-readout model (§4.2 remains the fallback). Kronos is the adopted price model.

---

## 0. TL;DR

1. **The core idea holds up. Several details in the original draft do not.** Keep the layered "driving stack" design (fast filter → reasoning/world model → deterministic risk governor → execution). Change the alpha source, the data feeds, the JEV dependency, and the regulatory assumptions (§1).
2. **The original benchmark table can't be used.** The "Hybrid JEV-TKG: 21.3% / Sharpe 1.78 / 63.4%" row cites MMF-Trans, a CSI-300 (Chinese market) paper that does not report those numbers. JEV launched on 2026-09-15, so no backtest of it can exist. Treat every number in that table as unsupported.
3. **Where research finds a real edge:** firm-specific news (company news, filings, earnings language) that markets absorb slowly, held for **1–10 days**. The effect is stronger in smaller stocks and on negative news, and **it is shrinking as more LLMs trade it** (Lopez-Lira & Tang; Chen, Kelly & Xiu). **Where the edge does not exist for us:** reacting to macro headlines like "Hormuz closed → buy USO" on a 15-minute GDELT delay. Those moves are priced in seconds.
4. **Everything can run locally on 1–2 DGX Sparks.** JEV itself is API-only with closed weights, but its mechanism (one forward pass, typed and calibrated answers, no decoding) can be rebuilt on an open model by reading answer-token logits or adding trained heads. Brev covers heavier fine-tunes.
5. **The biggest threat to validity is look-ahead bias, not model quality.** Standard LLMs "remember" what happened to prices. In Look-Ahead-Bench, their alpha fell 15–22 percentage points once tested outside their training window. Point-in-time (PiT) models stayed stable. Historical backtests have to use PiT models. Modern models can only be judged on data after their knowledge cutoff, or live.
6. **"Beats the market" has to be defined statistically.** A 30-day paper test cannot show it. To get t ≥ 2 on excess Sharpe (information ratio) of 1.0 takes about 4 years of daily returns. At the trade level, it takes roughly 250 or more independent trades. The evaluation is built around **event-level abnormal returns**, because they reach statistical power far sooner than portfolio Sharpe does.

---

## 1. Fact-check of the original concept doc

| Claim in draft | Finding | Impact |
|---|---|---|
| PDT rule (FINRA 2520, $25k) forces swing trading | **The SEC approved removing the PDT designation and the $25k minimum on 2026-04-14. The change took effect 2026-06-04.** Brokers have until **2027-10-20** to implement it, and it is replaced by real-time intraday margin requirements. | PDT is no longer the binding constraint. Swing horizons are still preferred for **cost and edge** reasons (§2), not legal ones. Still check what Alpaca currently enforces, and that margin at all generally requires ≥$2k equity. A $500 account is effectively a cash account (T+1 settlement, Good Faith Violation risk). |
| Alpaca fractional/notional orders are market + DAY only | **Out of date.** Fractional and notional orders now support **market, limit, stop, and stop-limit** (DAY TIF). **Brackets/OCO are still not supported** for fractional orders. | Limit entries are possible, which cuts slippage. A client-side synthetic bracket is still needed, but it can rest real **stop orders** on the server instead of watching ticks. |
| JEV can be the System-1 filter | JEV is real (TypeSafe AI, launched 2026-09-15, $0.042/M input tokens). It is **closed-weight with no self-hosting** and early access is by waitlist. | Conflicts with "all compute local." Rebuild the mechanism locally (§4.2). Keep JEV as an optional **comparison baseline**. |
| Benchmark table (Hybrid 21.3%, Sharpe 1.78, 63.4% accuracy) | Can't be traced to a source. MMF-Trans covers CSI-300 and reports only relative gains (−23.7% RMSE, +32.6% Sharpe vs. its own baseline). | Throw it out. Our own baselines replace it (§5.2). |
| GDELT drives trades | GDELT updates every 15 minutes. The published equity evidence mostly concerns **volatility**, not direction. | Demote GDELT to a regime/volatility feature. Directional signals should come from firm-level sources (§3). |
| JEPA world model for prices | TS-JEPA (2025) matches baselines on generic benchmarks (ETT, Electricity), with strength at long horizons. It has no published finance edge. | An optional research branch, not core. **Kronos** (AAAI 2026, MIT license, trained on 12B+ K-line records) is the stronger off-the-shelf price model to test first. |
| 30-day paper test before going live | Statistically meaningless for "beats the market" (§5.4). | Swap in staged gates that use event-level statistics plus a minimum trade count. |

---

## 2. What the research says about where an edge can exist

### 2.1 Evidence for news-driven predictability
- **Lopez-Lira & Tang** ("Can ChatGPT Forecast Stock Price Movements?"): LLM scores on headlines published after the model's cutoff predict **next-day drift**. The effect is strongest for **small stocks and negative news**, and returns **fall as LLM adoption rises**. The long-short strategy kept about 50% outperformance even at 25 bps costs. The initial reaction (around a 90% hit rate) **cannot be traded**: it happens before you can get in.
- **Chen, Kelly & Xiu** ("Expected Returns and LLMs"): LLM **embeddings** of news fed into a simple penalized regression predict returns across 16 markets and 13 languages. Prices absorb news "with an inefficient delay." Fresh news alerts give higher Sharpe. *Design lesson:* a frozen LLM embedding plus a linear model is a strong, cheap baseline that must be beaten.
- **"Fast numbers, slow language"** (2026): the numeric earnings surprise is arbitraged away by the next open. **Signals from the language** of earnings releases peak at the next open and **persist**. This favors interpreting text over racing on headlines.
- **PEAD (post-earnings announcement drift):** contested. Martineau (2022) finds it gone from non-microcaps by about 2006. Two papers accepted in 2025 find it alive, and the disagreement comes down to research design. Treat it as a **hypothesis to test**, not an assumption.

### 2.2 Evidence against "LLM bot beats the market"
- **StockBench** (2025, contamination-free, multi-month): most LLM agents, including GPT-5, Claude 4, Qwen3, and Kimi-K2, **fail to beat buy-and-hold**.
- **Alpha Arena S1** (Oct–Nov 2025, $10k real money each, crypto perps): 4 of 6 frontier models lost money, and GPT-5 lost about 63%. Only Qwen3-Max (+22%) and DeepSeek finished positive, over 17 days, which is too short to separate skill from luck.
- **LiveTradeBench** (2025, 50 days live, 21 LLMs): high general-ability scores (LMArena) **do not predict** trading results.
- **SPIVA 2025:** 79% of active large-cap funds trailed the S&P 500 in 2025. Over 15 years, **no** US equity category had a majority of managers beating its benchmark.

**Conclusion:** using an end-to-end "LLM as trader" is the documented failure mode. The original concept already avoids it with the deterministic governor. The edge has to come from a **narrow, measurable signal**: firm-specific text → calibrated forward abnormal return. The LLM's job is **measurement**, not decision-making.

### 2.3 Implications for a $500 account
- **Trade where the edge is slow:** firm-level text events in liquid mid/large caps, held 1–10 days. Avoid microcaps. That is where the edge is largest, but 0.5–2% spreads would consume it at our order sizes.
- **Don't race:** our latency (seconds to minutes) is hopeless against HFTs on headline reactions. Enter at the **next open or via a limit order**, and size for the drift, not the jump.
- **Core–satellite option (recommended to evaluate):** park idle cash in SPY/VOO and move only a satellite slice (for example 30–60%) into event positions. Then "beat the market" means "the satellite beats SPY," which is easier to measure and limits the damage when the model is wrong.

---

## 3. Data sources (point-in-time correctness is mandatory)

Every record must carry the **time the information became publicly available** (not the event date). The backtest may only see a record once its timestamp is ≤ the decision time.

| Source | Cost | History | Latency | Role | Point-in-time timestamp |
|---|---|---|---|---|---|
| **Alpaca News API** (Benzinga) | Free (200 req/min) | 2015→ | Real-time WS, 600–900 items/day | **Primary firm-level news** | `created_at` |
| **SEC EDGAR** (8-K, 10-Q/K, Form 4) | Free | 1990s→ | Seconds to minutes after acceptance | **Primary material-event feed**, insider trades | **Acceptance datetime** (not filing date) |
| Earnings releases / transcripts | 8-K Ex-99.1 free; transcripts mostly paid | Varies | Minutes | "Slow language" signal | Release time |
| **Alpaca market data** (bars, quotes) | Free IEX feed; SIP is paid | 2016→ | Real-time | Prices, spreads, labels | Bar time |
| GDELT 2.0 Events/GKG | Free | 2015→ | 15 min | Macro regime / volatility feature only | `DATEADDED` |
| Kalshi / Polymarket | Free APIs | 2021→ (thin early) | Real-time | Macro event probabilities (Fed, CPI); research suggests they are well calibrated | Trade time |
| FRED / ALFRED | Free | Decades | Daily | Macro context. **Use ALFRED vintages** to avoid revised data | Vintage date |

**Labels** (what models learn to predict): forward **abnormal** returns over h ∈ {1, 3, 5, 10} trading days, measured from the **next tradable price** after the timestamp. Abnormal = minus a market/sector-beta-adjusted benchmark (SPY + sector ETF). Questions like "is this a supply shock?" are not labels. Markets supply the ground truth.

---

## 4. Compute: what 1–2 DGX Sparks and Brev can do

### 4.1 Hardware facts
| | DGX Spark (GB10) | Implication |
|---|---|---|
| Memory | 128 GB unified, **273 GB/s** | Large models fit; decoding is bandwidth-bound and slow |
| gpt-oss-120b (MXFP4) | ~1,200–1,700 tok/s prefill, ~41–55 tok/s decode | Fits on one Spark. Good for System-2 reasoning on the few events that reach that tier |
| gpt-oss-20b (MXFP4) | ~3,200 tok/s prefill, ~58 tok/s decode | Candidate backbone for the System-1 filter |
| Llama 3.1 8B (FP4) | ~924 tok/s | — |
| 2× Spark (ConnectX-7 link) | e.g. 70B ≈ 12 tok/s decode; interconnect dominates | Run **two independent roles** (filter / reasoner) on separate boxes instead of tensor-parallel over the link |
| Fine-tuning on Spark | LoRA 8B ≈ 54k tok/s; QLoRA 70B ≈ 5k tok/s; full FT 3B ≈ 83k tok/s | Iterative LoRA works locally. Big runs go to Brev |

**Key point:** the System-1 workload is **prefill-only**: a classification question needs no decoding. The Spark is strong at prefill. At around 1–3k tok/s, a 500-token news item plus 5 questions takes roughly 0.2–0.5 s, which is in JEV's latency range, locally and at zero marginal cost. Expected volume (about 1k items/day) is a tiny load.

### 4.2 Rebuilding the JEV mechanism locally ("typed decision heads")
JEV's useful properties are a single forward pass, multiple typed questions over one shared state, calibrated probabilities, and no parsing failures. All of them come from standard techniques:
1. **Logit readout:** prompt with "…Answer: " and read P(token ∈ {Yes, No}), or P over option letters for a choice, or over level indices for a score (expected value gives the [0,1] score, the same as JEV's formula). One prefill, no sampling.
2. **Packed multi-question inference:** encode the state once in the KV cache and branch each question off it (prefix caching in vLLM/SGLang). This matches JEV's "questions can't attend to siblings" mask.
3. **Trained heads (the better version):** add linear/ordinal heads on the backbone's hidden state, trained directly on **forward abnormal returns**, with optional LoRA. This replicates the Chen-Kelly-Xiu design with a better backbone.
4. **Calibration:** temperature/isotonic scaling on a held-out window. Report Brier score (and its reliability/resolution split), ECE, and log loss.

### 4.3 Model candidates per tier (to be ablated, not assumed)
| Tier | Candidates | Notes |
|---|---|---|
| System-1 filter (backtest) | **PiT models**: Kelly et al. PIT-1B/4B (monthly checkpoints 2013–2024, HF `Diamegs/PIT-*`), ChronoBERT/ChronoGPT | Only these are valid for historical backtests |
| System-1 filter (live) | gpt-oss-20b, Qwen3.x ~8–30B, FinBERT (baseline) | Valid only on data after each model's cutoff |
| Price world model | **Kronos** (zero-shot, then fine-tuned), plus GARCH/HAR-RV for volatility | Kronos reports +93% RankIC over general time-series foundation models |
| System-2 reasoner | gpt-oss-120b on Spark #2 | Explains and vetoes; **does not** size positions |
| Optional research | TS-JEPA over joint price+news latents | Only if it beats Kronos in ablation |

---

## 5. Evaluation spec

### 5.1 Leakage controls (any failure invalidates a result)
1. **Point-in-time data:** every feature is gated on its availability timestamp. Use ALFRED for macro data. Use acceptance time for EDGAR.
2. **Model-memory leakage:** historical backtests only use PiT models with checkpoint date < test date. Modern LLMs only use data after their cutoff. Run a **memorization probe** per model: ask it for the return of a stock in a past window and compare with the truth. Score above chance = contaminated.
3. **Walk-forward with purging + embargo** (López de Prado): train on [t0, t1], purge any samples whose label windows overlap the test period, embargo ≥ 10 trading days, then test on (t1, t2]. Roll the window forward.
4. **Survivorship-free universe:** include delisted tickers and use the index membership as of each date.
5. **Execution realism:** fill at the next tradable price, plus half-spread, plus slippage (10 bps default; sweep 5–30). Size fractional orders realistically. Apply T+1 settlement and cash-account Good Faith Violation rules.
6. **Frozen final holdout:** the most recent 6 months are touched **once**, at the end of research.

### 5.2 Baselines (all evaluated with identical costs and universe)
| ID | Baseline | Why |
|---|---|---|
| B0 | **SPY buy-and-hold** | The target to beat |
| B1 | Equal-weight universe buy-and-hold | Universe-selection effect |
| B2 | **Random-entry control**: same number of trades, holding periods, and sizes on random dates and tickers | Separates signal from exposure/beta. **The single most important control** |
| B3 | 12-1 momentum and SMA trend | Classical quant |
| B4 | FinBERT sentiment → same governor | Standard NLP baseline |
| B5 | **Frozen LLM embedding + ridge (Chen-Kelly-Xiu)** | Strong academic baseline |
| B6 | Zero-shot LLM yes/no score (Lopez-Lira style) | Baseline without fine-tuning |
| B7 | End-to-end LLM agent (StockBench-style) | The documented failure mode, to show our architecture beats it |
| B8 | JEV API filter (optional) | Comparison against the original concept |

### 5.3 Metrics
**Signal level** (computed per event; this is the primary evidence):
- Mean **cumulative abnormal return (CAR)** at h = 1, 3, 5, 10 days for top- vs. bottom-scored events, with a t-stat (clustered by date).
- **Rank IC** (Spearman of score vs. forward abnormal return), mean and t-stat of daily IC, and IC decay over horizons.
- Hit rate, and precision@k for the gating threshold.
- **Calibration:** Brier score (reliability/resolution/uncertainty split), ECE, and log loss for every probabilistic output.

**Portfolio level:**
- CAGR, annual volatility, **Sharpe, Sortino**, max drawdown, Calmar.
- **Information ratio vs. SPY** and **CAPM / FF5+momentum alpha with t-stat** (shows whether returns are just beta).
- Turnover, cost drag (bps/year), average holding period, exposure %.
- **Deflated Sharpe Ratio** (Bailey & López de Prado), corrected for the number of configurations tried. We log every trial.
- **Probability of Backtest Overfitting** (CSCV) across the parameter sweep.

**System level:** latency p50/p99 per tier, data-gap rate, order rejection rate, fill slippage vs. modeled slippage, and reconciliation of governor vetoes.

### 5.4 Statistical power: how much evidence "beats the market" needs
- Portfolio: t ≈ IR × √years. IR 0.5 → **16 years** for t = 2. IR 1.0 → **4 years**. A live month proves nothing at the portfolio level.
- Trade level: t ≈ (mean excess / σ) × √N. With a mean excess return per trade of 0.5% and σ = 4%, t = 2 needs **N ≈ 256 independent trades**. That is why the gates below count trades.
- Multiple testing: Harvey-Liu-Zhu suggest **t ≥ 3** for newly discovered factors. We use the Deflated Sharpe Ratio to account for this.

### 5.5 Stage gates (proposed thresholds, to be confirmed with you)
| Gate | Environment | Pass criteria |
|---|---|---|
| **G1 Signal** | Walk-forward PiT backtest 2016–2024 | Top-decile 5-day CAR > 0 with t ≥ 3. Mean daily Rank IC t ≥ 2. Beats B5 (embedding+ridge) on IC. Brier score better than the base rate |
| **G2 Strategy** | Same, with full costs | **Beats B2 (random entry) at p < 0.05**. IR vs. SPY ≥ 0.5. **DSR ≥ 0.95**. PBO ≤ 0.2. Max DD ≤ 1.5× SPY's over the same period. Survives 2× the slippage assumption |
| **G3 Post-cutoff** | Modern models on data after their cutoff (about 2025-H2 → 2026-Q3) | Signal-level results within 1 standard error of G1. No significant alpha decay (Look-Ahead-Bench style) |
| **G4 Shadow live** | Alpaca paper, real-time | ≥ **100 closed trades** or 3 months (whichever is later). Live per-trade excess return within the backtest's 90% CI. Realized slippage ≤ modeled. Zero governor/broker reconciliation errors |
| **G5 Live micro** | $500 real | Start the satellite at $100–250. Kill switch at −15% drawdown or if live results fall outside the CI. Continue G4-style monitoring |

---

## 6. Risks and unknowns
- **Alpha decay:** LLM-news edges are being competed away (Lopez-Lira & Tang). Monitor IC decay continuously.
- **PiT model quality:** PiT-4B is weaker than modern models. Backtests may **understate** what modern models do live. G3/G4 measure that gap.
- **Data cost:** Alpaca's free tier uses IEX quotes (a partial view of the market), so spreads may look worse than they are. Transcripts need a paid source.
- **Regulatory flux:** the PDT replacement is being phased in until 2027-10-20. Check Alpaca's current rules before any intraday mode.
- **Small-sample live results** will be noisy. The gates are designed around this.

---

## 7. Decisions needed from you before the design spec
1. **Universe:** US equities plus ETFs only, or also crypto (24/7, no settlement friction, but a different edge) or prediction markets?
2. **Core–satellite** (idle cash in SPY) vs. **all-cash between trades**?
3. **Account type:** cash (T+1, no leverage, shorts impossible) or margin (needs ≥ $2k, allows shorts)? At $500 the plan assumes **long-only cash**.
4. **Sparks:** one or two? (Two → filter and reasoner on separate boxes.)
5. **Brev budget** for fine-tunes (rough $/month)?
6. Are the **G1–G5 thresholds** acceptable, especially "≥ 100 paper trades before live"?

---

## Sources
- TypeSafe AI, *Introducing System One Models & Jev* — https://typesafe.ai/blog/introducing-system-one-models-and-jev ; launch/no-self-hosting: https://startuphub.ai/ai-news/artificial-intelligence/2026/typesafe-jev-model-kills-chat , https://www.orcarouter.ai/ko/blog/jev-open-source
- SEC approval of FINRA Rule 4210 amendments (SR-FINRA-2025-017) — https://www.sec.gov/files/rules/sro/finra/2026/34-105226.pdf ; effective-date coverage — https://www.stocktitan.net/articles/pattern-day-trader-rule-eliminated-2026
- Alpaca fractional trading docs — https://docs.alpaca.markets/docs/fractional-trading ; changelog — https://docs.alpaca.markets/changelog/support-for-fractional-usd-stop-stop-limit-lct-stop-stop-limit-limit-with-extended-hours-orders
- Alpaca News API — https://docs.alpaca.markets/us/docs/historical-news-data
- Lopez-Lira & Tang — https://arxiv.org/abs/2304.07619
- Chen, Kelly & Xiu, *Expected Returns and LLMs* (discussion) — https://jacobslevycenter.wharton.upenn.edu/wp-content/uploads/2024/09/ExpectedReturnAndLLM_Discussion_SophiaZhengziLi_NoPause.pdf
- *Fast Numbers, Slow Language* — https://arxiv.org/pdf/2606.29734
- PEAD debate — https://anderson-review.ucla.edu/is-post-earnings-announcement-drift-a-thing-again/
- StockBench — https://arxiv.org/abs/2510.02209
- LiveTradeBench — https://arxiv.org/abs/2511.03628
- Alpha Arena S1 — https://www.gncrypto.news/news/qwen-wins-alpha-arena-season-1-with-22-percent-returns/
- SPIVA US Year-End 2025 — https://www.spglobal.com/spdji/en/spiva/article/spiva-us-year-end-2025/
- Look-Ahead-Bench — https://arxiv.org/abs/2601.13770 , code: https://github.com/benstaf/lookaheadbench
- Kelly, Malamud, Schwab, Xu, *Scaling Point-in-Time Language Models* — https://www.nber.org/papers/w35247 ; models: https://huggingface.co/Diamegs/PIT-4B-FT-202412
- ChronoBERT/ChronoGPT — https://arxiv.org/abs/2502.21206
- FinCAD (parametric look-ahead bias) — https://arxiv.org/abs/2605.24564
- Kronos — https://arxiv.org/abs/2508.02739
- TS-JEPA — https://arxiv.org/abs/2509.25449
- MMF-Trans (source of the cited row) — https://arxiv.org/abs/2501.16621
- GDELT news sentiment & volatility — https://er.ucu.edu.ua/items/1a7a61f9-eff9-48bb-af2d-09f11eeefe18/full
- Kalshi macro markets — https://www.nber.org/papers/w34702
- DGX Spark performance — https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks , https://registry.ollama.ai/blog/nvidia-spark-performance , https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark , 2-node: https://storagereview.com/review/nvidia-dgx-spark-cluster-review-distributed-inference-on-dell-gigabyte-and-hp
- Backtest overfitting: Bailey & López de Prado, *The Deflated Sharpe Ratio* (2014); Bailey et al., *The Probability of Backtest Overfitting* (2016); López de Prado, *Advances in Financial ML* (2018, purged CV/embargo); Harvey, Liu & Zhu, *…and the Cross-Section of Expected Returns* (2016).
