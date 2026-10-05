# Kronos at the intraday decision time

**EXPLORATORY**

12094 valid ticker-cycle samples; 497 decision cycles; 0 invalid/missing model comparisons; 331 excluded for fewer than 400 prior daily sessions; 0 missing five-session labels.
Prepared decision sessions: 2024-07-08 to 2026-06-30
Paired SPY subtraction changed rank IC by at most 0 across evaluated cycles.

| Feature | Mean cycle rank IC | Median IC | Positive cycles | 5-session block CI | First/second half |
|---|---:|---:|---:|---:|---:|
| forecast_return | +0.001 | -0.010 | 48% | [-0.034, +0.035] | +0.007/-0.005 |
| reversal20 | +0.006 | -0.001 | 50% | [-0.038, +0.051] | +0.030/-0.017 |
| momentum12_1 | +0.050 | +0.095 | 57% | [-0.005, +0.101] | +0.036/+0.063 |
| below_ma400 | -0.029 | -0.060 | 44% | [-0.077, +0.022] | +0.005/-0.063 |
| current_day_return | +0.014 | +0.039 | 55% | [-0.009, +0.037] | +0.008/+0.020 |
| open_gap | -0.008 | +0.002 | 50% | [-0.033, +0.018] | -0.018/+0.002 |
| intraday_return_to_cycle | +0.026 | +0.034 | 57% | [+0.002, +0.049] | +0.039/+0.013 |
| forecast_dispersion | +0.519 | +0.543 | 100% | [+0.495, +0.542] | +0.555/+0.483 |
| atr14 | +0.703 | +0.733 | 100% | [+0.681, +0.723] | +0.725/+0.681 |
| forecast_excess | +0.001 | -0.010 | 48% | [-0.034, +0.035] | +0.007/-0.005 |
| forecast_excess_after_price_controls | -0.004 | -0.015 | 49% | [-0.035, +0.027] | -0.023/+0.014 |

## Absolute SPY-relative forecasts

Mean forecast excess: -0.264%; mean realized excess: +0.204%; MAE: 5.434%; zero-forecast MAE: 4.799%; directional accuracy: 50.3%.
These fixed summaries use no tuned buy threshold and do not include transaction costs.

## Limitations

- Kronos inputs include completed intraday bars through each decision time; no news or JEV features are used. 15m data is used at 09:45 ET and 30m data at 15:30 ET.
- Both raw and SPY-relative stock returns are reported. Subtracting one common SPY value per date cannot change cross-sectional rank IC, though paired SPY forecasts are retained for absolute market-relative thresholds and future portfolio decisions.
- The point-in-time candidate universe uses a current asset snapshot, so it has survivorship bias and is not a complete historical listing universe. Outcomes overlap; no thresholds were optimized. Rows with fewer than 400 prior daily sessions are omitted to keep the 400-session moving-average control comparable.
- Alpaca all-adjusted bars are used for model inputs and labels, so outcomes are adjusted returns rather than raw spot-price returns or executable fills. They cannot stand in for historical option-contract pricing.
- Forecast paths and prices are vendor bars, not executable fills or quoted spreads.
