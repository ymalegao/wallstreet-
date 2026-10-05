# Broad-universe price-only baseline

**EXPLORATORY price-only baseline; not model alpha or certification**

2024-11-01 through 2026-10-02; 24 monthly windows; 6537 current listed-stock proxy symbols; 1932 pass the point-in-time price/liquidity screen on the last date. Daily bars exist for 5549 of the proxy symbols.

| Baseline | Return | Max drawdown |
|---|---:|---:|
| 12–1 momentum, top 20, gross | 69.4% | -34.0% |
| 12–1 momentum, top 20, 15 bps/side | 57.7% | -34.9% |
| All eligible with 12–1 history, equal weight | 33.7% | -13.1% |
| SPY buy-and-hold | 37.8% | -18.8% |
| Random top-20 median (100 trials) | 29.0% | — |

This is a price-only benchmark. It does not test JEV, Kronos, intraday execution, or a learned strategy.
Drawdown is measured from monthly portfolio marks for rebalanced stock strategies and daily closes for SPY.
Random portfolios ranged from -4.1% to 76.5%; 33/100 beat SPY.

- Alpaca's current active/inactive security master is not a historical point-in-time security master.
- SIVB, FRC and BBBY are absent from both the current asset snapshot and this bar cache; historical delisted coverage is incomplete.
- The listed-stock proxy excludes obvious funds/derivatives by exchange, ticker syntax and name; some may remain.
- Daily OHLC bars omit quotes, intraday volatility, fills, borrow costs, taxes and market impact.
- No feature thresholds were tuned here, but this is still one exploratory period and one fixed rule.
- The random controls are descriptive, and monthly observations are not independent significance tests.
