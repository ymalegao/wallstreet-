# Kronos daily feature sweep

**EXPLORATORY**

This uses only data through the prior daily close. It tests market-adjusted cross-sectional ranking and volatility features, not the intraday news-time setup.

| Setting | Features | Samples | Dates | Mean daily rank IC | Median IC | Positive dates | 5-day block CI | First/second half IC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| L40_T1 | forecast_excess | 1281 | 61 | +0.091 | +0.083 | 67% | [+0.003, +0.179] | +0.140/+0.044 |
| L40_T1 | reversal20 | 1281 | 61 | +0.131 | +0.125 | 64% | [+0.052, +0.211] | +0.151/+0.112 |
| L40_T1 | momentum12_1 | 1281 | 61 | -0.031 | -0.003 | 48% | [-0.201, +0.142] | -0.141/+0.076 |
| L40_T1 | below_ma400 | 1281 | 61 | +0.048 | +0.072 | 52% | [-0.124, +0.217] | +0.130/-0.030 |
| L40_T1 | forecast_dispersion | 1281 | 61 | +0.651 | +0.677 | 100% | [+0.584, +0.711] | +0.712/+0.592 |
| L40_T1 | atr14 | 1281 | 61 | +0.753 | +0.775 | 100% | [+0.710, +0.795] | +0.777/+0.731 |
| L90_T0.6 | forecast_excess | 1281 | 61 | -0.047 | -0.018 | 46% | [-0.155, +0.063] | -0.122/+0.026 |
| L90_T0.6 | reversal20 | 1281 | 61 | +0.131 | +0.125 | 64% | [+0.052, +0.211] | +0.151/+0.112 |
| L90_T0.6 | momentum12_1 | 1281 | 61 | -0.031 | -0.003 | 48% | [-0.201, +0.142] | -0.141/+0.076 |
| L90_T0.6 | below_ma400 | 1281 | 61 | +0.048 | +0.072 | 52% | [-0.124, +0.217] | +0.130/-0.030 |
| L90_T0.6 | forecast_dispersion | 1281 | 61 | +0.522 | +0.519 | 100% | [+0.469, +0.577] | +0.562/+0.483 |
| L90_T0.6 | atr14 | 1281 | 61 | +0.753 | +0.775 | 100% | [+0.710, +0.795] | +0.777/+0.731 |
| L90_T1 | forecast_excess | 1281 | 61 | +0.015 | -0.006 | 48% | [-0.085, +0.118] | -0.038/+0.066 |
| L90_T1 | reversal20 | 1281 | 61 | +0.131 | +0.125 | 64% | [+0.052, +0.211] | +0.151/+0.112 |
| L90_T1 | momentum12_1 | 1281 | 61 | -0.031 | -0.003 | 48% | [-0.201, +0.142] | -0.141/+0.076 |
| L90_T1 | below_ma400 | 1281 | 61 | +0.048 | +0.072 | 52% | [-0.124, +0.217] | +0.130/-0.030 |
| L90_T1 | forecast_dispersion | 1281 | 61 | +0.598 | +0.619 | 100% | [+0.549, +0.650] | +0.634/+0.564 |
| L90_T1 | atr14 | 1281 | 61 | +0.753 | +0.775 | 100% | [+0.710, +0.795] | +0.777/+0.731 |
| L400_T1 | forecast_excess | 1271 | 61 | +0.028 | +0.075 | 54% | [-0.148, +0.197] | +0.098/-0.040 |
| L400_T1 | reversal20 | 1271 | 61 | +0.136 | +0.125 | 64% | [+0.056, +0.218] | +0.156/+0.117 |
| L400_T1 | momentum12_1 | 1271 | 61 | -0.030 | -0.003 | 48% | [-0.200, +0.143] | -0.138/+0.075 |
| L400_T1 | below_ma400 | 1271 | 61 | +0.048 | +0.072 | 52% | [-0.124, +0.218] | +0.127/-0.028 |
| L400_T1 | forecast_dispersion | 1271 | 61 | +0.395 | +0.431 | 90% | [+0.301, +0.475] | +0.290/+0.496 |
| L400_T1 | atr14 | 1271 | 61 | +0.754 | +0.784 | 100% | [+0.711, +0.795] | +0.775/+0.733 |

## Limitations

- Forecasts use daily bars through the prior session, omitting the current news-day price reaction.
- Within a date, subtracting the same SPY return from every ticker cannot change rank IC. Interpret rank metrics as stock-return ranking; market adjustment matters for absolute forecasts and thresholds.
- Date-level rank ICs use a 23-name proxy watchlist and overlapping 5-session outcomes.
- Temperature/lookback settings were informed by a preceding 12-date sweep; this comparison is not a pristine holdout.
- No portfolio thresholds were tuned and no live or paper trading was performed.
