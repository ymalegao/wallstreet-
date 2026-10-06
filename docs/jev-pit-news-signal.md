# JEV news signal on the point-in-time daily top 25

**EXPLORATORY JEV-only feature test; no Kronos forecasts, gates, or exclusions**

2024-07-01 to 2026-07-01 (exclusive): 25,050/25,050 complete top-25 ticker-cycles, 19,687 with news scored and 84,219 event texts.

The table reports JEV by itself, without any Kronos feature, filter, or Kronos-based sample exclusion. Rank IC is calculated cross-sectionally per decision cycle against five-session SPY-relative returns; 95% intervals resample five-session blocks. With beta fixed at 1, subtracting the same SPY return from every stock in a cycle does not change that cycle's ranks; SPY is the market benchmark in this stock test, not an options signal test.

| Feature | Pooled Spearman (descriptive) | Mean cycle rank IC | Median cycle IC | Positive cycles | 5-session block 95% CI |
|---|---:|---:|---:|---:|---:|
| JEV news signal | -0.005 | -0.001 | +0.006 | 51.2% | [-0.018, +0.016] |
| 20-session reversal | +0.008 | -0.001 | -0.003 | 49.8% | [-0.046, +0.043] |
| 12–1 momentum | +0.047 | +0.052 | +0.083 | 57.7% | [-0.001, +0.105] |

## Signal groups

| Fixed group | Samples | Mean 5-session abnormal return | Median | Positive fraction |
|---|---:|---:|---:|---:|
| negative_signal | 5,582 | +0.37% | -0.02% | 49.8% |
| zero_signal | 10,329 | +0.22% | +0.06% | 50.7% |
| positive_signal | 9,139 | +0.23% | -0.01% | 49.9% |
| fixed_entry_rule_signal_ge_1_3 | 9,139 | +0.23% | -0.01% | 49.9% |

## Limits

- The candidate pool is the earlier current-asset snapshot; delisted-security survivorship bias remains until the revised universe is used.
- Near-duplicate news threshold and historical news latency remain unverified. Revised text is delayed until updated_at.
- Five-session outcomes overlap. Confidence intervals use a five-session block bootstrap; pooled correlations are descriptive only.
- This feature test is not a live execution backtest, and it does not establish an options strategy.
- Supplemental Massive bars use their own adjustment rules and are recorded in the report; adjustment differences around corporate actions may affect mixed-source labels and price features.
