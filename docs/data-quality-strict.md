# Data quality report

**Result: FAIL**

## Metrics

```json
{
  "events": 3145,
  "sources": {
    "alpaca_news": 3145
  },
  "duplicate_ids": 0,
  "acausal_timestamps": 0,
  "revised_articles": 308,
  "revisions_used_early": 0,
  "outside_clock_hours_fraction": 0.5672496025437201,
  "eligible_ticker_months": 306,
  "news_ticker_month_coverage": 1.0,
  "intraday_bars": 48295,
  "regular_hours_bars": 20358,
  "normalization_rejects": 0
}
```

## Failures

- unverified dedupe.jaccard_threshold
- unverified latency.news_minutes

## Limitations

- Symbol coverage is limited to the declared sample, not a survivorship-free market universe
