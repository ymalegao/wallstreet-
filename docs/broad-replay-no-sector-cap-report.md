# INVALIDATED: legacy zero-shot market replay (no sector cap)

**Do not interpret the strategy returns or compare the strategies from this report.** This replay uses the legacy 400-daily-bar Kronos forecast and `P(up) >= 0.5` gate. That forecast setting produced severely biased outputs and invalid forecast paths. The dataset excluded exactly 28 affected ticker-cycle samples, selecting the replay universe based on Kronos validity. All strategy and random-control portfolio results below are therefore diagnostic artifacts only. Decision-time Kronos features have not been passed through this portfolio replay.

1773 ticker-cycle samples; 2026-07-01 through 2026-10-02. Initial equity $500.

| Strategy | Return | Max drawdown | Closed trades | Open positions |
|---|---:|---:|---:|---:|
| jev_kronos | 9.87% | -5.26% | 48 | 0 |
| text_only | 10.36% | -5.10% | 63 | 0 |
| kronos_only | 1.17% | -4.82% | 55 | 0 |
| 12_1_momentum | -13.81% | -15.35% | 29 | 0 |
| SPY buy-and-hold | 3.26% | -3.36% | — | — |

Random-entry controls (30 seeds): median return 5.31%, range -15.80% to 27.82%.

## Signal diagnostics

{
  "spearman_text_vs_market_adjusted_5d": -0.06218099204467237,
  "spearman_kronos_vs_return_5d": 0.08627724231838377,
  "spearman_text_vs_same_day_market_adjusted_return": -0.013807790385419818,
  "spearman_kronos_vs_same_day_return": 0.030808539988088135
}

## Same-day 09:45 ET basket (toy day-trading comparison)

At most five positions, 20% equity each, entered at the next 15-minute open and exited at the regular-session close; 15 bps per side.

| Strategy | Return | Max drawdown |
|---|---:|---:|
| jev_kronos | -2.97% | -5.58% |
| text_only | -9.60% | -11.97% |
| kronos_only | -5.29% | -7.59% |
| 12_1_momentum | -16.56% | -17.51% |
| SPY same-time intraday | -18.01% | -18.01% |
| Random controls median | -3.77% | — |
Random-control range: -7.60% to 5.76%; 30/30 beat the SPY intraday baseline.

## Costs

Per-side cost sensitivity for combined strategy:
- 5.0 bps: 11.79%
- 15.0 bps: 9.87%
- 30.0 bps: 6.96%

## Limitations

- Only 23 liquid selected stocks and a short recent window; model training overlap unknown.
- Samples missing exact intraday bars, ATR warm-up, or valid Kronos forecasts are excluded; see excluded_samples.
- Historical revised text conservatively delayed; assumed five-minute news latency remains unmeasured.
- Adjusted bars provide a total-return proxy, not actual broker fills, quotes or dividend cash accounting.
- Fixed thresholds were not optimized. This interval is exploratory and is not a final holdout.
- Random controls match candidate count and use identical rules; realized trade counts/exposure can differ.
- Five-day overlapping outcomes are dependent; descriptive IC is not a significance test.
- Kronos daily close forecasts and intraday exits have different exact horizons.
- No broker, order book, partial fill or market impact simulation; no live/paper orders.
- Text scores select the strongest absolute event per cycle; contradictory events can cancel in reality.
- Same-day return diagnostics and the capped toy basket omit sector caps, stops, quotes, and measured market impact.
- The same-day toy replay uses only 09:45 ET candidates, a five-position 20%-equity cap, and exits at the close. It has no sector cap, stop, quotes or measured market impact.
