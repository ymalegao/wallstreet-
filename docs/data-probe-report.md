# Data probe report

Run at 2026-10-04T07:01:56.692739+00:00 by scripts/probe_apis.py.


**Summary:** {'PASS': 22, 'FAIL': 7, 'MEASURE': 9}

## Alpaca News API (REST)

- **PASS** articles on 2026-10-01: 832
- **PASS** field `id` populated: 832/832
- **PASS** field `headline` populated: 832/832
- **PASS** field `created_at` populated: 832/832
- **PASS** field `updated_at` populated: 832/832
- **PASS** field `summary` populated: 261/832
- **PASS** field `content` populated: 261/832
- **PASS** field `symbols` populated: 811/832
- **PASS** field `url` populated: 832/832
- **PASS** field `source` populated: 832/832
- **FAIL** sort=asc respected across pages
- **PASS** pagination exercised: 17 requests
- **MEASURE** revised after publication (updated_at > created_at): 50/832; lag minutes p50=0.0 p95=155.0
- **MEASURE** symbols per article: mean=1.82, zero=21
- **MEASURE** vendors: [('benzinga', 832)]

History depth (articles on one weekday in mid-March of each year, all symbols):

- 2015-03-16: 623
- 2016-03-16: 574
- 2017-03-16: 466
- 2018-03-16: 342
- 2019-03-18: 412
- 2020-03-16: 800
- 2021-03-16: 722
- 2022-03-16: 1087
- 2023-03-16: 996
- 2024-03-18: 919
- 2025-03-17: 819
- 2026-03-16: 895

## Alpaca bars

- **PASS** feed=sip SPY 15Min bars on 2026-09-29: 64 bars, pre-market=22, after-hours=16
- **MEASURE** feed=sip first RTH bar `t` (bar START expected at 09:30 ET): 2026-09-29T09:30:00-04:00
- **PASS** feed=iex SPY 15Min bars on 2026-09-29: 28 bars, pre-market=2, after-hours=0
- **MEASURE** feed=iex first RTH bar `t` (bar START expected at 09:30 ET): 2026-09-29T09:30:00-04:00
- **MEASURE** NVDA closes adjustment=raw: 2024-06-06:1209.98, 2024-06-07:1208.88, 2024-06-10:121.79, 2024-06-11:120.91
- **MEASURE** NVDA closes adjustment=all: 2024-06-06:120.65, 2024-06-07:120.54, 2024-06-10:121.44, 2024-06-11:120.58
- **PASS** history for delisted SIVB (survivorship): 20 daily bars Jan 2023
- **PASS** history for delisted FRC (survivorship): 20 daily bars Jan 2023
- **PASS** crypto BTC/USD 1Hour bars: 25

## SEC EDGAR

- **PASS** ticker->CIK map: 10434 tickers; AAPL=320193
- **PASS** field `acceptanceDateTime` present: 1588 8-K/4 rows
- **PASS** field `items` present: 1588 8-K/4 rows
- **PASS** field `primaryDocument` present: 1588 8-K/4 rows
- **PASS** field `accessionNumber` present: 1588 8-K/4 rows
- **FAIL** header for 0000320193-26-000018: ACCEPTANCE-DATETIME not found
- **FAIL** header for 0000320193-26-000011: ACCEPTANCE-DATETIME not found
- **FAIL** header for 0001140361-26-015711: ACCEPTANCE-DATETIME not found
- **FAIL** header for 0001140361-26-006577: ACCEPTANCE-DATETIME not found
- **FAIL** header for 0000320193-26-000005: ACCEPTANCE-DATETIME not found
- **FAIL** acceptanceDateTime timezone -> configs/sources.yaml edgar.acceptance_tz: {}

## Finnhub company news

- **PASS** articles last 7 days across probe tickers: 1343
- **MEASURE** AAPL articles ~13 months ago (free-tier history limit): 0
- **MEASURE** vendors: [('Yahoo', 1071), ('Benzinga', 165), ('SeekingAlpha', 45), ('ChartMill', 37), ('CNBC', 24), ('Fintel', 1)]

## Cross-source near-duplicate study (headline + summary, same ticker, within 48h)

- 0.0-0.2: 45084 pairs
- 0.2-0.4: 23 pairs
- 0.4-0.6: 15 pairs
- 0.6-0.8: 11 pairs
- 0.8-1.0: 127 pairs

Examples per band (read these and choose `dedupe.jaccard_threshold`):

- [0.2-0.4] J=0.38 (MSFT, finnhub first by 240 min)
  - alpaca: 'OpenAI Executive Backs Out of Second $25M Donation to A.I. Super PAC'- NY Times
  - finnhub: 'OpenAI Executive Backs Out of Second $25M Donation to A.I. Super PAC'- NY Times https://www.nytimes.com/2026/09/30/technology/openai-brockman-super-pac-leading
- [0.2-0.4] J=0.38 (JPM, finnhub first by 240 min)
  - alpaca: SEC Censures JPMorgan Over Alleged Failure To Stop Swap Trader Trading In U.S.; No Monetary Fine Imposed
  - finnhub: SEC Censures JPMorgan Over Alleged Failure To Stop Swap Trader Trading In U.S.; No Monetary Fine Imposed https://www.sec.gov/files/litigation/admin/2026/34-1065
- [0.2-0.4] J=0.37 (AAPL, finnhub first by 1680 min)
  - alpaca: Inquiry Into Apple's Competitor Dynamics In Technology Hardware, Storage &amp; Peripherals Industry In today&#39;s rapidly changing and fiercely competitive bus
  - finnhub: Competitor Analysis: Evaluating Apple And Competitors In Technology Hardware, Storage &amp; Peripherals Industry In today&#39;s rapidly changing and highly comp
- [0.4-0.6] J=0.59 (TSLA, finnhub first by 240 min)
  - alpaca: President Trump Says Will Be Naming AI Czar In Next Three Or Four Days
  - finnhub: President Trump Says Will Be Naming AI Czar In Next Three Or Four Days https://www.youtube.com/watch?v=CqlX1TZwdVE
- [0.4-0.6] J=0.59 (MSFT, finnhub first by 240 min)
  - alpaca: President Trump Says Will Be Naming AI Czar In Next Three Or Four Days
  - finnhub: President Trump Says Will Be Naming AI Czar In Next Three Or Four Days https://www.youtube.com/watch?v=CqlX1TZwdVE
- [0.4-0.6] J=0.58 (MSFT, finnhub first by 240 min)
  - alpaca: Anthropic invests $100M to train 10,000 engineers and tackle the enterprise AI talent gap
  - finnhub: Anthropic invests $100M to train 10,000 engineers and tackle the enterprise AI talent gap https://www.anthropic.com/news/claude-frontier-academy
- [0.6-0.8] J=0.78 (JPM, finnhub first by 240 min)
  - alpaca: BAE Weighs Bid For Robin Radar In Possible $2.3B Deal
  - finnhub: BAE Weighs Bid For Robin Radar In Possible $2.3B Deal -Reuters Exclusive.
- [0.6-0.8] J=0.74 (MSFT, finnhub first by 240 min)
  - alpaca: OpenAI Announces It Identified, Disrupted Coordinated Campaign Designed To Extract Protected Reasoning From Its Models; Operators Did Not Break Our Encryption,
  - finnhub: OpenAI Announces It Identified, Disrupted Coordinated Campaign Designed To Extract Protected Reasoning From Its Models; Operators Did Not Break Our Encryption,
- [0.6-0.8] J=0.73 (PFE, finnhub first by 1680 min)
  - alpaca: 10 Health Care Stocks With Whale Alerts In Today’s Session This whale alert can help traders discover the next big trading opportunities.
Whales are entities wi
  - finnhub: 10 Health Care Stocks Whale Activity In Today’s Session This whale alert can help traders discover the next big trading opportunities.
Whales are entities with
- [0.8-1.0] J=1.00 (MSFT, finnhub first by 240 min)
  - alpaca: Micron, Nike, Meta and More: 5 Stocks Investors Couldn't Stop Buzzing About This Week Retail investors talked up five hot stocks during the week (Sept. 28 to Oc
  - finnhub: Micron, Nike, Meta and More: 5 Stocks Investors Couldn't Stop Buzzing About This Week Retail investors talked up five hot stocks during the week (Sept. 28 to Oc
- [0.8-1.0] J=1.00 (NVDA, finnhub first by 240 min)
  - alpaca: Benzinga Bulls and Bears: Alphabet, Micron, Nike Benzinga examined the prospects for many investors’ favorite stocks over the last week — here’s a look at some
  - finnhub: Benzinga Bulls and Bears: Alphabet, Micron, Nike Benzinga examined the prospects for many investors’ favorite stocks over the last week — here’s a look at some
- [0.8-1.0] J=1.00 (NVDA, finnhub first by 240 min)
  - alpaca: Nvidia-Backed Robotics Startup FieldAI Hits $10 Billion Valuation After $700 Million Raise, More Than Quadrupling Value in Just Over a Year: Report FieldAI repo
  - finnhub: Nvidia-Backed Robotics Startup FieldAI Hits $10 Billion Valuation After $700 Million Raise, More Than Quadrupling Value in Just Over a Year: Report FieldAI repo
