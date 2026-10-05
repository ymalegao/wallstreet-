# Kronos at the news decision time

**EXPLORATORY**

1034 valid ticker-cycle samples; 61 decision cycles; 0 invalid/missing model comparisons.

| Feature | Mean cycle rank IC | Median IC | Positive cycles | 5-session block CI | First/second half |
|---|---:|---:|---:|---:|---:|
| forecast_excess | +0.191 | +0.262 | 71% | [+0.086, +0.289] | +0.236/+0.148 |
| reversal20 | +0.098 | +0.143 | 64% | [+0.019, +0.175] | +0.118/+0.079 |
| momentum12_1 | -0.009 | -0.030 | 47% | [-0.166, +0.151] | -0.108/+0.086 |
| below_ma400 | +0.035 | -0.032 | 49% | [-0.134, +0.207] | +0.132/-0.059 |
| current_day_return | -0.133 | -0.173 | 36% | [-0.208, -0.048] | -0.186/-0.082 |
| current_day_vs_spy | -0.133 | -0.173 | 36% | [-0.208, -0.048] | -0.186/-0.082 |
| open_gap | -0.098 | -0.123 | 40% | [-0.197, +0.004] | -0.070/-0.125 |
| intraday_return_to_cycle | -0.094 | -0.086 | 38% | [-0.186, -0.008] | -0.164/-0.026 |
| forecast_dispersion | +0.565 | +0.547 | 100% | [+0.514, +0.621] | +0.592/+0.539 |
| atr14 | +0.693 | +0.696 | 100% | [+0.641, +0.744] | +0.714/+0.673 |
| forecast_excess_after_price_controls | +0.014 | +0.044 | 56% | [-0.072, +0.099] | -0.041/+0.066 |

## Limitations

- Kronos inputs include completed intraday bars through each news-decision time; 15m data is used at 09:45 and 30m at the afternoon cycle.
- Returns run from the decision-bar close to the close five sessions later, with simple SPY subtraction (beta=1).
- Within a date, subtracting the same SPY return from every ticker cannot change rank IC. Interpret rank metrics as stock-return ranking; market adjustment matters for absolute forecasts and thresholds.
- The 23-name watchlist, overlapping outcomes and candidate-only news cycles limit inference; no thresholds were optimized.
- Forecast paths and prices are vendor bars, not executable fills or quoted spreads.
