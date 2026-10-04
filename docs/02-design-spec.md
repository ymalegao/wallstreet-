# 02 — Design Spec: Architecture, Training, Evaluation

**Status:** Draft v1 for review
**Date:** 2026-10-04
**Depends on:** `01-research-spec.md` (evidence base, metrics, and gates G1–G5)
**Principle:** do not invent anything. Assemble proven, open components (open JEV replicas, Kronos, LightGBM, Alpaca) carefully, and spend the effort on **data correctness, evaluation, and risk control**. That is where trading systems actually fail.

---

## 0. Decisions locked (from review of 01)

| # | Decision |
|---|---|
| D1 | **Universe:** US stocks first. Crypto (BTC/ETH/SOL on Alpaca) is a phase-2 sleeve on the same pipeline. **No prediction markets.** |
| D2 | **Idle capital:** **cash** in v1. A cash account makes "SPY between trades" cause Good Faith Violation (GFV) problems: an SPY sale settles T+1, so a same-day stop-out of the new position is a violation. The backtest still reports a *synthetic* "idle cash in SPY" variant. If that variant wins, we switch on a flag later. |
| D3 | **Account:** $500 **cash account, long-only**. The $2k margin account (shorts, no settlement lag) is unlocked **only after G5** passes. |
| D4 | **Brev budget:** **$200 total** (about 65–80 H100-hours). Everything else runs on the Spark(s). |
| D5 | **No GDELT.** News comes from free firm-level sources (§3). |
| D6 | **System-1 filter:** an **open JEV replica**, run locally and fine-tuned on market outcomes. No TypeSafe API dependency. |
| D7 | **Price model:** **Kronos**, used as published plus fine-tuned. No custom world model and no JEPA. |
| D8 | **Gates G1–G5** from 01 §5.5 accepted. |
| D9 | **Compute:** designed to run on **1 Spark**. A second Spark, if available, takes the System-2 reasoner and backtests. |

---

## 1. System overview

```mermaid
flowchart LR
  subgraph Ingest["1 · Ingest (always on)"]
    A1[Alpaca News WS<br/>Benzinga] --> N
    A2[SEC EDGAR<br/>8-K / Form 4 poller] --> N
    A3[Finnhub company-news<br/>REST poller] --> N
    A4[Alpaca bars/quotes<br/>stocks + crypto] --> P[(Price store)]
    N[Normalize · dedupe ·<br/>ticker map · first-seen ts] --> E[(Event store)]
  end
  subgraph S1["2 · System-1 perception (Spark)"]
    E --> J[Open-JEV decision model<br/>typed questions + trained heads]
    P --> K[Kronos forecaster<br/>return / vol distribution]
  end
  subgraph Fuse["3 · Fusion"]
    J --> F[LightGBM meta-model<br/>→ calibrated P(win), E[r]]
    K --> F
    P --> F
  end
  subgraph S2["4 · System-2 review (optional)"]
    F -->|top candidates only| R[Reasoner LLM<br/>veto / explain]
  end
  subgraph Gov["5 · Deterministic governor"]
    R --> G[Edge > cost? · sizing ·<br/>limits · settled-cash · kill switches]
    F --> G
  end
  G --> X[Alpaca execution<br/>limit entries · DAY stops · time exits]
  X --> L[(Ledger · fills · audit)]
  L --> G
```

**Cadence (v1):** ingestion runs continuously and records every item with its *first-seen* timestamp. **Decisions** happen in two batch cycles per trading day: **09:40 ET** (overnight and pre-market news; avoids the opening auction chaos) and **15:30 ET** (intraday news). The edge being targeted is multi-day drift (01 §2), so faster cycles add cost and complexity without adding edge. Crypto (phase 2) runs a cycle every 4 hours, 24/7.

---

## 2. Components

### 2.1 Ingestion and event store
| Source | Access | Used for | Timestamp used |
|---|---|---|---|
| **Alpaca News API** (Benzinga) | WebSocket live, REST history back to 2015, free | Main news stream, stocks and crypto | `created_at` |
| **SEC EDGAR** | Free JSON/RSS, 10 req/s limit, needs a User-Agent | 8-K items, Form 4 insider buys/sells | **Acceptance datetime** |
| **Finnhub** company-news | Free REST (about 60/min, ~1 year history) | Second source: coverage check and dedupe | `datetime` |
| **FNSPID** dataset | Free download (15.7M articles, 1999–2023) | **Training corpus only** for the decision heads. Some timestamps are date-only, so it is never used for timing-sensitive backtests | date (+ time where present) |
| Alpaca market data | Free (IEX feed); SIP is paid | Daily/hourly bars, quotes for spread estimates, labels | bar time |

**Processing steps:**
1. **Normalize** each item to `{event_id, source, first_seen_ts, tickers[], headline, body, url}`.
2. **Dedupe** across sources with MinHash (Jaccard ≥ 0.8 within 48 h). Keep the **earliest** `first_seen_ts` and link duplicates. A story syndicated five times is still one event.
3. **Ticker map:** use the vendor symbols. For EDGAR, map CIK → ticker using the point-in-time SEC mapping.
4. **Novelty score:** 1 − max cosine similarity against the same ticker's events in the prior 7 days. Chen-Kelly-Xiu find "fresh" news carries the signal.
5. **Storage:** Parquet partitioned by date plus a DuckDB catalog. Records are append-only and never rewritten, and every record keeps `ingested_at`. This is what makes backtests point-in-time honest.

### 2.2 System-1: open-JEV decision model
**Choice of replica (evaluated in the M2 bake-off, §5):**

| Candidate | Backbone | Size | Why |
|---|---|---|---|
| **autotrust/JEV-9B** *(default)* | Qwen3.5-9B | 9B | Claims KL-parity with closed Jev 1.13. LoRA-tunable on one Spark. Can also act as a System-2 generator |
| **Von** *(fast baseline)* | ModernBERT encoder | 395M | Ships with `von calibrate` and fine-tuning docs. Full fine-tunes take minutes; runs on CPU |
| **OpenJev** | DiffusionGemma 26B-A4B | 26B MoE (4B active) | Jev wire-compatible, about 30 ms. Fine-tuning not documented |
| **PIT-4B + heads** *(backtest-only)* | Kelly et al. point-in-time GPT | 4B | **The only leak-free backbone** for the 2016–2024 history track (§6) |

All replicas are used behind one interface (the Jev request/response wire format), so they can be swapped, and the closed Jev API could also serve as a reference baseline if desired.

**Typed questions** (zero-shot features, packed into one forward pass per event):
| Key | Type | Content |
|---|---|---|
| `material` | noul | Does this contain material, new, firm-specific information (not a rehash or recap)? |
| `event_type` | choice | earnings · guidance · M&A · legal/regulatory · product/contract · management · capital-structure (buyback/offering) · analyst action · insider trade · macro/sector · other |
| `direction` | score (5 levels) | strongly negative → strongly positive for the shareholders of `ticker` |
| `surprise` | score (4 levels) | expected/known → completely unexpected |
| `persistence` | score (3 levels) | one-off → structural, multi-quarter impact |

**Trained heads** (the important part): small MLP heads on the backbone's pooled hidden state, predicting the **forward abnormal-return quintile** at h ∈ {1, 5, 10} days. The training loss is cross-entropy, a proper scoring rule, so it optimizes calibration the same way JEV's RLCD/Brier objective does. Head outputs are used as features alongside the zero-shot answers.

**Serving:** vLLM (NVIDIA's DGX Spark container) with prefix caching, so each event's state is encoded once and branched per question. Expected load is about 1–2k events per day; at roughly 0.3 s per event that is under 10 minutes of compute per day.

### 2.3 Kronos price forecaster
- **Models:** Kronos-small (24.7M) and Kronos-base (102.3M), MIT license, 512-bar context. Kronos-large is not released.
- **Use:** for each universe ticker, feed the last 400 daily bars (plus hourly bars for crypto). Draw `sample_count = 32` sampled paths for h = 1/5/10. Derive features from them: median return, P(up), forecast volatility, and the 10th/90th percentiles.
- **Fine-tune:** walk-forward on US daily bars using the repo's `finetune_csv` pipeline. Runs on the Spark (102M parameters, so cheap). One model per walk-forward fold.
- **Standalone check:** Kronos alone is baseline **B9**. If its RankIC is near zero on US large caps (plausible, since price-only signals there are weak), it stays in as a **volatility/risk feature** for sizing and stops rather than as a direction signal. Either way that is a valid, useful result.

### 2.4 Fusion meta-model
- **Model:** LightGBM, one model per horizon. Inputs: System-1 answers and head outputs, novelty, source count, event age, Kronos features, simple context (ticker's 20-day volatility, 1- and 3-month momentum, SPY 50/200-day trend, VIX level, sector ETF return, days to earnings).
- **Targets:** (a) P(abnormal return over h > cost hurdle); (b) E[abnormal return].
- **Calibration:** isotonic regression on the most recent fold. Brier score and ECE are tracked live.
- **Why trees and not another neural net:** few thousand events per year, tabular features, and the need for monotone constraints and interpretability (SHAP). This is the standard choice.

### 2.5 System-2 reasoner (optional, gated)
- **Model:** JEV-9B's generator mode or gpt-oss-120b (MXFP4, about 45 tok/s on a Spark). It runs **only on the top ~5 candidates per cycle**.
- **Role:** produce a short thesis plus a **veto** flag. Typical veto reasons: already priced in, stale, misattributed ticker, or a known pending catalyst. **It never sizes and never initiates trades.**
- It is kept only if ablation shows the veto improves G1/G2 metrics. Otherwise it stays as an explanation log for human review.

### 2.6 Deterministic governor (the safety layer)
All rules are code, configured in YAML and unit-tested. Every intent → decision → order is logged with the reason.

| Rule | v1 value |
|---|---|
| Entry filter | E[r] − est. cost ≥ 50 bps **and** P(win) ≥ 0.55 (both tuned in G1/G2 and recorded as trials) |
| Sizing | ¼-Kelly on the calibrated edge, scaled so each position's forecast volatility ≈ 1% of equity per day |
| Caps | Each position ≤ 25% of equity; ≤ 5 open positions; ≤ 2 per sector; gross exposure ≤ 100% (cash account) |
| **Settled cash only** | Buy only with **settled** cash. GFVs become impossible by construction, at the cost of some idle T+1 capital |
| Entry order | Marketable limit at mid + ½ spread, cancelled if unfilled within 5 min. No market orders at the open |
| Stop | 2 × ATR(14) stop. Fractional stops are **DAY-only**, so the governor re-submits them every morning at 09:31. Overnight gaps are not protected; this is accepted and sized for |
| Take-profit / time exit | Exit at horizon h (default 5 days), or a take-profit at +3 × ATR |
| Liquidity | Price ≥ $5, 20-day average daily volume ≥ $20M, quoted spread ≤ 10 bps at decision time |
| Kill switches | Day P&L ≤ −4%; drawdown from peak ≤ −15%; data feed stale > 15 min; live ECE > 0.10 over the last 50 trades; any broker/ledger reconciliation mismatch → **flatten and halt**, with an alert |

### 2.7 Execution and operations
- `alpaca-py`; orders use **idempotent** `client_order_id = hash(decision_id)`. A trade-updates WebSocket feeds the ledger, plus a reconciliation job every 5 minutes.
- **Paper and live are the same code**; the only difference is the API keys and `paper=True`.
- **Alerts** via ntfy/Discord (fills, vetoes, kill switch). A Streamlit dashboard shows P&L vs. SPY, open positions, the calibration plot, and the latest signals.

### 2.8 Crypto sleeve (phase 2)
- Same pipeline. Universe: BTC/ETH/SOL. Alpaca crypto news plus Kronos on hourly bars.
- Cost hurdle is higher: Alpaca tier-1 fees are 15 bps maker / 25 bps taker, so **limit (maker) orders only** and an entry filter E[r] − cost ≥ 100 bps.
- 24/7 cycles. No settlement constraint.
- It must pass its **own** G1–G4 gates against a BTC buy-and-hold benchmark before receiving capital (cap: 20% of equity).

---

## 3. Labels and data hygiene (non-negotiable)
- **Label:** abnormal return r_i − β_i·r_SPY − γ_i·r_sector over h ∈ {1, 5, 10} trading days. Measured **from the first tradable price after `first_seen_ts` + 5 min** (next open if the market is closed), to a close.
- **Point-in-time everywhere:** features as-of the decision timestamp; S&P 500 / Nasdaq-100 **membership as of each date**; delisted tickers kept; ALFRED for anything macro.
- **Purged walk-forward:** expanding training window, 1-year test folds rolled quarterly, **purge** samples whose label window overlaps the test period, **10-day embargo**.
- **Trial registry:** every configuration evaluated is logged in MLflow (thresholds, horizons, features). The number of trials feeds the Deflated Sharpe Ratio (DSR) and the Probability of Backtest Overfitting (PBO).

---

## 4. Training plan

| Stage | What | Where | Cost |
|---|---|---|---|
| T0 | **Backfill**: Alpaca news 2015→now (~600k items), EDGAR 8-K/Form 4 2015→now, FNSPID, daily bars for about 1,000 tickers plus crypto hourly | Spark (I/O-bound) | $0 |
| T1 | **Embedding pass**: run each backbone over all events once and cache pooled states plus zero-shot answers (~300M tokens per backbone) | Spark: about 1 day per 9B backbone. **Brev burst** if the Spark is busy | ~$20–40 |
| T2 | **Decision heads** (SFT on return quintiles): trained per walk-forward fold on cached states. Seconds to minutes per fold | Spark | $0 |
| T3 | **LoRA fine-tune** of the winning backbone end-to-end on the same labels. Only if T2 heads plateau and LoRA beats them out-of-sample | Spark (LoRA 9B ≈ 50k tok/s) | $0 |
| T4 | **Kronos fine-tune** per fold on US daily bars (and crypto hourly) | Spark | $0 |
| T5 | **Fusion LightGBM** per fold plus isotonic calibration | Spark CPU | $0 |
| T6 | **RL (GRPO) experiment**, optional: train the System-2 reasoner (9B) to emit a thesis plus probability. **Reward = negative Brier/log score** against the realized abnormal-return bucket, plus format validity (reinforcement learning with verifiable rewards). The design follows Trading-R1 (SFT → GRPO), but the reward is calibration rather than P&L, because P&L rewards are too noisy at our sample size | **Brev** H100, about 30–40 h | ~$100–120 |
| — | Reserve | | ~$40 |

**Why RL is optional and last:** with roughly 10³–10⁴ labelled events per year, supervised heads trained with a proper scoring rule already optimize the target directly. RL earns its place only if it beats T2/T3 on G1 metrics out-of-sample. We do **not** use RL for the trading policy itself (FinRL-style PPO): the governor stays deterministic, as the original concept required.

---

## 5. Milestones

| M | Deliverable | Exit criterion |
|---|---|---|
| **M0** | Repo skeleton, config, CI (ruff, pytest, mypy), Docker image for Spark (aarch64) | CI green |
| **M1** | Ingestion plus event store plus backfill (T0); data-quality report (coverage per ticker, timestamp audit, duplicate rate) | ≥ 95% of S&P 500 tickers with news. **Zero** records where `first_seen_ts` > `ingested_at` |
| **M2** | Decision-model bake-off: JEV-9B vs. Von vs. OpenJev vs. PIT-4B, all on cached states (T1/T2), plus baselines B4–B6 | Report: IC, CAR, Brier per backbone; memorization probe per backbone |
| **M3** | Kronos zero-shot plus fine-tuned (B9, T4); fusion model (T5) | Report: standalone and fused IC |
| **M4** | Backtester (daily, event-driven, full governor plus cost model) with the B0–B9 baselines | **Gates G1 + G2** on the history track |
| **M5** | **Gate G3** on the post-cutoff track | G3 pass |
| **M6** | Live shadow: paper trading with the full stack, dashboard, alerts | **G4**: ≥ 100 closed trades or 3 months |
| **M7** | Live $100–250 satellite, then the full $500 | **G5** ongoing |
| M8 | Phase 2: crypto sleeve; optional GRPO reasoner (T6); margin account | Each passes its own gates |

---

## 6. Evaluation: can we verify on year-long history, or only live?

**Both, in three tracks.** Each track rules out a different way of fooling ourselves, and all three must pass.

| Track | Data window | Models allowed | What it proves | Gates |
|---|---|---|---|---|
| **H: History** | 2016 → 2024 (about 8 years, walk-forward) | **PIT-4B** (monthly checkpoint < test date), Von **retrained** from ModernBERT with no financial pretraining, Kronos **fine-tuned per fold** (pretraining is still a leak risk, so Kronos zero-shot is excluded here) | That the **pipeline and edge exist** across regimes (2018, 2020, 2022) without model-memory leakage | G1, G2 |
| **P: Post-cutoff** | Day after each production model's cutoff → now (target ≥ 12 months; Kronos after its Aug 2025 release) | Production models (JEV-9B, Kronos zero-shot and fine-tuned) | That the **actual production models** perform within error of track H, so there is no hidden look-ahead boost | G3 |
| **L: Live paper** | Now → +3 months / 100 trades | Full stack | Execution realism: real fills, slippage, latency, uptime | G4 |

**Leak detection (every production backbone):**
1. Look up the published cutoff, then **verify it**: run a memorization probe asking for past monthly returns and S&P 500 events. Above-chance accuracy marks the actual contamination horizon.
2. Run the Look-Ahead-Bench "alpha decay" test: compare performance inside vs. outside the cutoff. A large drop means the model is leaking.
3. If the clean post-cutoff window is < 9 months, track P is advisory and promotion relies on tracks H + L.

**Reporting:** each run writes an HTML tearsheet (equity curve vs. SPY and the random-entry control, drawdowns, CAR by decile, IC decay, calibration plot, DSR/PBO, cost breakdown) and a frozen `results.json`. Gates are checked by a script, not by eye.

---

## 7. Repository layout
```
wallstreet-/
├── docs/                    # 01 research, 02 design, ADRs
├── configs/                 # universe.yaml, governor.yaml, models.yaml, gates.yaml
├── src/ws/
│   ├── ingest/              # alpaca_news.py, edgar.py, finnhub.py, fnspid.py, bars.py
│   ├── store/               # event_store.py (Parquet+DuckDB), pit.py (as-of joins, membership)
│   ├── perception/          # jev_client.py (wire format), backbones/, heads.py, questions.yaml
│   ├── kronos/              # forecaster.py, finetune.py
│   ├── fusion/              # features.py, meta_model.py, calibrate.py
│   ├── reasoner/            # review.py, grpo/ (phase 2)
│   ├── governor/            # rules.py, sizing.py, settlement.py, killswitch.py
│   ├── exec/                # alpaca_broker.py, ledger.py, reconcile.py
│   ├── backtest/            # engine.py, costs.py, baselines.py, walkforward.py
│   └── eval/                # metrics.py (IC, CAR, DSR, PBO, Brier/ECE), tearsheet.py, gates.py
├── services/                # docker-compose: ingest, perception (vLLM), scheduler, dashboard
└── tests/                   # unit (governor rules, PIT joins!), integration (paper broker)
```
**Stack:** Python 3.12, uv, Polars + DuckDB, PyTorch, vLLM (NVIDIA Spark container), LightGBM, alpaca-py, APScheduler, MLflow, Streamlit, Docker Compose.

---

## 8. Open items / risks
1. **JEV-9B model card couldn't be fetched** (network block during research): its license, training-data cutoff, and whether it was distilled from closed-Jev outputs need checking in M2. Distillation from a 2026 API would push its contamination horizon to 2026, which makes track P shorter.
2. **Kronos pretraining date range is undisclosed.** It is treated as contaminated before Aug 2025 and evaluated zero-shot only after that.
3. **IEX-only quotes** on the free Alpaca tier understate liquidity. Spreads are estimated conservatively (2× IEX quoted spread) until live fills calibrate the cost model.
4. **FNSPID timestamp quality** is mixed. It is used only for head pretraining, never for timing.
5. **PDT phase-in:** irrelevant for the cash account. Re-check before enabling margin (D3).
6. **Expectation setting:** a realistic success is **positive, statistically supported alpha over the random-entry control and SPY after costs**. The odds of large returns are low; SPIVA and StockBench say most attempts fail. The gates exist so that we learn this cheaply if it is true.

---

## Sources (new in this doc; see 01 for the rest)
- Open JEV replicas: OpenJev — https://github.com/razorback16/openjev ; Von — https://github.com/wfzyx/von ; JEV-9B — https://huggingface.co/autotrust/JEV-9B ; collection — https://huggingface.co/collections/Ferr0/open-jev-typed-decision-models
- Kronos repo — https://github.com/shiyu-coder/Kronos ; weights — https://huggingface.co/NeoQuasar/Kronos-base
- Trading-R1 (SFT → GRPO for trading LLMs) — https://arxiv.org/abs/2509.11420
- FNSPID — https://github.com/Zdong104/FNSPID_Financial_News_Dataset
- Alpaca crypto fees — https://docs.alpaca.markets/docs/crypto-fees
- Finnhub free tier — https://www.interactivebrokers.com/campus/ibkr-quant-news/exploring-the-finnhub-io-api/
- H100 rental pricing (budget estimate) — https://jarvislabs.ai/blog/h100-price
