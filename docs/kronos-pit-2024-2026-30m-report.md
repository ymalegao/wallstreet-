# Kronos at the intraday decision time

**EXPLORATORY**

12019 valid ticker-cycle samples; 494 decision cycles; 0 invalid/missing model comparisons; 331 excluded for fewer than 400 prior daily sessions; 0 missing five-session labels.
Prepared decision sessions: 2024-07-11 to 2026-06-30
Paired SPY subtraction changed rank IC by at most 0 across evaluated cycles.

| Feature | Mean cycle rank IC | Median IC | Positive cycles | 5-session block CI | First/second half |
|---|---:|---:|---:|---:|---:|
| forecast_return | -0.020 | -0.033 | 45% | [-0.055, +0.015] | -0.024/-0.016 |
| reversal20 | -0.001 | -0.022 | 48% | [-0.046, +0.046] | +0.020/-0.021 |
| momentum12_1 | +0.047 | +0.084 | 57% | [-0.006, +0.099] | +0.032/+0.063 |
| below_ma400 | -0.030 | -0.052 | 45% | [-0.079, +0.022] | +0.001/-0.060 |
| current_day_return | -0.009 | -0.007 | 48% | [-0.033, +0.015] | -0.008/-0.010 |
| open_gap | -0.007 | +0.018 | 52% | [-0.033, +0.019] | -0.025/+0.011 |
| intraday_return_to_cycle | -0.006 | -0.004 | 49% | [-0.030, +0.019] | -0.001/-0.011 |
| forecast_dispersion | +0.544 | +0.558 | 100% | [+0.522, +0.567] | +0.573/+0.516 |
| atr14 | +0.689 | +0.713 | 100% | [+0.666, +0.711] | +0.714/+0.665 |
| forecast_excess | -0.020 | -0.033 | 45% | [-0.055, +0.015] | -0.024/-0.016 |
| forecast_excess_after_price_controls | -0.004 | +0.002 | 50% | [-0.034, +0.028] | -0.022/+0.014 |

## Absolute SPY-relative forecasts

Mean forecast excess: -0.751%; mean realized excess: +0.239%; MAE: 5.167%; zero-forecast MAE: 4.633%; directional accuracy: 49.4%.
These fixed summaries use no tuned buy threshold and do not include transaction costs.

## Limitations

- Kronos inputs include completed intraday bars through each decision time; no news or JEV features are used. 15m data is used at 09:45 ET and 30m data at 15:30 ET.
- Both raw and SPY-relative stock returns are reported. Subtracting one common SPY value per date cannot change cross-sectional rank IC, though paired SPY forecasts are retained for absolute market-relative thresholds and future portfolio decisions.
- The point-in-time candidate universe uses a current asset snapshot, so it has survivorship bias and is not a complete historical listing universe. Outcomes overlap; no thresholds were optimized. Rows with fewer than 400 prior daily sessions are omitted to keep the 400-session moving-average control comparable.
- Alpaca all-adjusted bars are used for model inputs and labels, so outcomes are adjusted returns rather than raw spot-price returns or executable fills. They cannot stand in for historical option-contract pricing.
- Forecast paths and prices are vendor bars, not executable fills or quoted spreads.
