# Fundamentals v1 coverage

- Fundamentals snapshot: `8ef316af7b2b80b5ecaa697c326e74dede3298f87ab01cf86511dd68aacca3b4`
- Universe: `S&P 100` (101 current constituents; survivorship-biased historically).
- SEC facts rows: 2,530,872
- Tickers missing from `company_tickers.json`: BRK.B
- Dual share-class CIK mappings share issuer fundamentals; market equity uses each ticker's close with issuer-level shares, as specified.

## Per-ticker coverage

| Ticker | CIK | SIC | FF12 industry | SEC facts | Book equity | Net income | Operating cash flow | Assets | Revenue | Cost of revenue | Gross profit | Shares outstanding | Shared CIK tickers |
|---|---:|---:|---|---|---|---|---|---|---|---|---|---|---|
| AAPL | 0000320193 | 3571 | BusEq | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| ABBV | 0001551152 | 2834 | Hlth | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| ABT | 0000001800 | 2834 | Hlth | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| ACN | 0001467373 | 7389 | Other | present | yes | yes | yes | yes | yes | yes | no | yes | — |
| ADBE | 0000796343 | 7372 | BusEq | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| AMAT | 0000006951 | 3674 | BusEq | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| AMD | 0000002488 | 3674 | BusEq | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| AMGN | 0000318154 | 2836 | Hlth | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| AMT | 0001053507 | 6798 | Money | present | yes | yes | yes | yes | yes | yes | no | yes | — |
| AMZN | 0001018724 | 5961 | Shops | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| ANET | 0001596532 | 3576 | BusEq | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| AVGO | 0001730168 | 3674 | BusEq | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| AXP | 0000004962 | 6199 | Money | present | yes | yes | yes | yes | yes | no | no | yes | — |
| BA | 0000012927 | 3721 | Manuf | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| BAC | 0000070858 | 6021 | Money | present | yes | yes | yes | yes | yes | no | no | yes | — |
| BKNG | 0001075531 | 4700 | Other | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| BLK | 0002012383 | 6211 | Money | present | yes | yes | yes | yes | yes | no | no | yes | — |
| BMY | 0000014272 | 2834 | Hlth | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| BNY | 0001390777 | 6022 | Money | present | yes | yes | yes | yes | yes | no | no | yes | — |
| BRK.B | — | — | Other | missing ticker mapping | — | — | — | — | — | — | — | — | — |
| C | 0000831001 | 6021 | Money | present | yes | yes | yes | yes | yes | yes | no | yes | — |
| CAT | 0000018230 | 3531 | Manuf | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| CMCSA | 0001166691 | 4841 | Telcm | present | yes | yes | yes | yes | yes | no | no | yes | — |
| COF | 0000927628 | 6021 | Money | present | yes | yes | yes | yes | yes | no | no | yes | — |
| COP | 0001163165 | 2911 | Enrgy | present | yes | yes | yes | yes | yes | yes | no | yes | — |
| COST | 0000909832 | 5331 | Shops | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| CRM | 0001108524 | 7372 | BusEq | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| CSCO | 0000858877 | 3576 | BusEq | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| CVS | 0000064803 | 5912 | Shops | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| CVX | 0000093410 | 2911 | Enrgy | present | yes | yes | yes | yes | yes | yes | no | yes | — |
| DE | 0000315189 | 3523 | Manuf | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| DELL | 0001571996 | 3571 | BusEq | present | yes | yes | yes | yes | yes | yes | yes | no | — |
| DHR | 0000313616 | 3823 | BusEq | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| DIS | 0001744489 | 7990 | Other | present | yes | yes | yes | yes | yes | no | no | yes | — |
| DUK | 0001326160 | 4931 | Utils | present | yes | yes | yes | yes | yes | yes | no | yes | — |
| EMR | 0000032604 | 3600 | Manuf | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| FDX | 0001048911 | 4513 | Other | present | yes | yes | yes | yes | yes | yes | no | yes | — |
| GD | 0000040533 | 3730 | Manuf | present | yes | yes | yes | yes | yes | yes | no | yes | — |
| GE | 0000040545 | 3600 | Manuf | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| GEV | 0001996810 | 3600 | Manuf | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| GILD | 0000882095 | 2836 | Hlth | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| GM | 0001467858 | 3711 | Durbl | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| GOOG | 0001652044 | 7370 | BusEq | present | yes | yes | yes | yes | yes | yes | no | no | GOOG, GOOGL |
| GOOGL | 0001652044 | 7370 | BusEq | present | yes | yes | yes | yes | yes | yes | no | no | GOOG, GOOGL |
| GS | 0000886982 | 6211 | Money | present | yes | yes | yes | yes | no | no | no | yes | — |
| HD | 0000354950 | 5211 | Shops | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| IBM | 0000051143 | 3570 | BusEq | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| INTC | 0000050863 | 3674 | BusEq | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| INTU | 0000896878 | 7372 | BusEq | present | yes | yes | yes | yes | yes | yes | no | yes | — |
| ISRG | 0001035267 | 3842 | Hlth | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| JNJ | 0000200406 | 2834 | Hlth | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| JPM | 0000019617 | 6021 | Money | present | yes | yes | yes | yes | yes | no | no | yes | — |
| KO | 0000021344 | 2080 | NoDur | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| LIN | 0001707925 | 2810 | Chems | present | yes | yes | yes | yes | yes | no | no | yes | — |
| LLY | 0000059478 | 2834 | Hlth | present | yes | yes | yes | yes | yes | yes | no | yes | — |
| LMT | 0000936468 | 3760 | Manuf | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| LOW | 0000060667 | 5211 | Shops | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| LRCX | 0000707549 | 3559 | Manuf | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| MA | 0001141391 | 7389 | Other | present | yes | yes | yes | yes | yes | no | no | yes | — |
| MCD | 0000063908 | 5812 | Shops | present | yes | yes | yes | yes | yes | yes | no | yes | — |
| MDLZ | 0001103982 | 2000 | NoDur | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| MDT | 0001613103 | 3845 | Hlth | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| META | 0001326801 | 7370 | BusEq | present | yes | yes | yes | yes | yes | yes | no | no | — |
| MMM | 0000066740 | 3841 | Hlth | present | yes | yes | yes | yes | yes | yes | no | yes | — |
| MO | 0000764180 | 2111 | NoDur | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| MRK | 0000310158 | 2834 | Hlth | present | yes | yes | yes | yes | yes | yes | no | yes | — |
| MS | 0000895421 | 6211 | Money | present | yes | yes | yes | yes | yes | no | no | yes | — |
| MSFT | 0000789019 | 7372 | BusEq | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| MU | 0000723125 | 3674 | BusEq | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| NEE | 0000753308 | 4911 | Utils | present | yes | yes | yes | yes | yes | no | no | yes | — |
| NFLX | 0001065280 | 7841 | Other | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| NOW | 0001373715 | 7372 | BusEq | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| NVDA | 0001045810 | 3674 | BusEq | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| ORCL | 0001341439 | 7372 | BusEq | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| PANW | 0001327567 | 3577 | BusEq | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| PEP | 0000077476 | 2080 | NoDur | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| PFE | 0000078003 | 2834 | Hlth | present | yes | yes | yes | yes | yes | yes | no | yes | — |
| PG | 0000080424 | 2840 | Chems | present | yes | yes | yes | yes | yes | yes | no | yes | — |
| PLTR | 0001321655 | 7372 | BusEq | present | yes | yes | yes | yes | yes | yes | yes | no | — |
| PM | 0001413329 | 2111 | NoDur | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| QCOM | 0000804328 | 3663 | BusEq | present | yes | yes | yes | yes | yes | yes | no | yes | — |
| RTX | 0000101829 | 3724 | Manuf | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| SBUX | 0000829224 | 5810 | Shops | present | yes | yes | yes | yes | yes | yes | no | yes | — |
| SCHW | 0000316709 | 6211 | Money | present | yes | yes | yes | yes | yes | no | no | yes | — |
| SNDK | 0002023554 | 3572 | BusEq | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| SO | 0000092122 | 4911 | Utils | present | yes | yes | yes | yes | yes | yes | no | yes | — |
| T | 0000732717 | 4813 | Telcm | present | yes | yes | yes | yes | yes | yes | no | yes | — |
| TMO | 0000097745 | 3829 | BusEq | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| TMUS | 0001283699 | 4812 | Telcm | present | yes | yes | yes | yes | yes | yes | no | yes | — |
| TSLA | 0001318605 | 3711 | Durbl | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| TXN | 0000097476 | 3674 | BusEq | present | yes | yes | yes | yes | yes | yes | yes | yes | — |
| UBER | 0001543151 | 7389 | Other | present | yes | yes | yes | yes | yes | no | no | yes | — |
| UNH | 0000731766 | 6324 | Money | present | yes | yes | yes | yes | yes | yes | no | yes | — |
| UNP | 0000100885 | 4011 | Other | present | yes | yes | yes | yes | yes | no | no | yes | — |
| UPS | 0001090727 | 4210 | Other | present | yes | yes | yes | yes | yes | no | no | yes | — |
| USB | 0000036104 | 6021 | Money | present | yes | yes | yes | yes | yes | no | no | yes | — |
| V | 0001403161 | 7389 | Other | present | yes | yes | yes | yes | yes | no | no | yes | — |
| VZ | 0000732712 | 4813 | Telcm | present | yes | yes | yes | yes | yes | yes | no | yes | — |
| WFC | 0000072971 | 6021 | Money | present | yes | yes | yes | yes | yes | no | no | yes | — |
| WMT | 0000104169 | 5331 | Shops | present | yes | yes | yes | yes | yes | yes | no | yes | — |
| XOM | 0002115436 | 2911 | Enrgy | present | yes | yes | yes | yes | yes | no | no | yes | — |

## Notes

- Item coverage means the source contains at least one eligible USD or share-count tag for that CIK; it does not guarantee a non-null feature on every decision date.
- Features are null before their source filing becomes available and after the annual facts become more than 18 months stale.
- SIC is from the SEC submissions snapshot and FF12 categories are from the dated Ken French `Siccodes12.zip` file; unmapped or missing SIC values are `Other`.
