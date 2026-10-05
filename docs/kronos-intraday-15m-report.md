# Kronos at the news decision time

**EXPLORATORY**

1060 valid ticker-cycle samples; 61 decision cycles; 0 invalid/missing model comparisons.

| Feature | Mean cycle rank IC | Median IC | Positive cycles | 5-session block CI | First/second half |
|---|---:|---:|---:|---:|---:|
| forecast_excess | +0.226 | +0.311 | 75% | [+0.137, +0.308] | +0.249/+0.204 |
| reversal20 | +0.129 | +0.193 | 63% | [+0.050, +0.203] | +0.115/+0.143 |
| momentum12_1 | -0.043 | -0.029 | 47% | [-0.228, +0.131] | -0.163/+0.072 |
| below_ma400 | +0.057 | +0.082 | 54% | [-0.118, +0.237] | +0.158/-0.040 |
| current_day_return | -0.134 | -0.175 | 33% | [-0.211, -0.057] | -0.152/-0.116 |
| current_day_vs_spy | -0.134 | -0.175 | 33% | [-0.211, -0.057] | -0.152/-0.116 |
| open_gap | -0.078 | -0.125 | 44% | [-0.166, +0.010] | -0.056/-0.100 |
| intraday_return_to_cycle | -0.035 | +0.011 | 53% | [-0.123, +0.041] | -0.067/-0.003 |
| forecast_dispersion | +0.521 | +0.554 | 100% | [+0.466, +0.577] | +0.561/+0.483 |
| atr14 | +0.700 | +0.713 | 100% | [+0.649, +0.750] | +0.740/+0.661 |
| forecast_excess_after_price_controls | +0.066 | +0.062 | 58% | [-0.010, +0.143] | +0.037/+0.093 |

## Limitations

- Kronos inputs include completed intraday bars through each news-decision time; 15m data is used at 09:45 and 30m at the afternoon cycle.
- Returns run from the decision-bar close to the close five sessions later, with simple SPY subtraction (beta=1).
- Within a date, subtracting the same SPY return from every ticker cannot change rank IC. Interpret rank metrics as stock-return ranking; market adjustment matters for absolute forecasts and thresholds.
- The 23-name watchlist, overlapping outcomes and candidate-only news cycles limit inference; no thresholds were optimized.
- Forecast paths and prices are vendor bars, not executable fills or quoted spreads.
