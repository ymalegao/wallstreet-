# Exploratory zero-shot market replay

**EXPLORATORY; not G1/G2 certification**

625 ticker-cycle samples; 2026-07-01 through 2026-09-18. Initial equity $500.

| Strategy | Return | Max drawdown | Closed trades | Open positions |
|---|---:|---:|---:|---:|
| jev_kronos | 4.42% | -5.24% | 31 | 0 |
| text_only | 4.26% | -3.13% | 48 | 0 |
| kronos_only | 4.42% | -4.57% | 39 | 0 |
| 12_1_momentum | 1.28% | -5.26% | 50 | 0 |
| SPY buy-and-hold | 2.19% | -3.36% | — | — |

Random-entry controls (30 seeds): median return 3.06%, range -3.86% to 14.63%.

## Signal diagnostics

{
  "spearman_text_vs_market_adjusted_5d": 0.015833035089857905,
  "spearman_kronos_vs_return_5d": 0.12359355085623648
}

## Costs

Per-side cost sensitivity for combined strategy:
- 5.0 bps: 5.61%
- 15.0 bps: 4.42%
- 30.0 bps: 2.67%

## Limitations

- Only eight selected stocks and a short recent window; model training overlap unknown.
- Historical revised text conservatively delayed; assumed five-minute news latency remains unmeasured.
- Adjusted bars provide a total-return proxy, not actual broker fills, quotes or dividend cash accounting.
- Fixed thresholds were not optimized. This interval is exploratory and is not a final holdout.
- Random controls match candidate count and use identical rules; realized trade counts/exposure can differ.
- Five-day overlapping outcomes are dependent; descriptive IC is not a significance test.
- Kronos daily close forecasts and intraday exits have different exact horizons.
- No broker, order book, partial fill or market impact simulation; no live/paper orders.
- Text scores select the strongest absolute event per cycle; contradictory events can cancel in reality.
