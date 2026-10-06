# Survivorship coverage audit

This audit compares the prior current-asset top-25 with a revised point-in-time selection over historical daily bars. The grouped-bar pool includes any ticker with a reported bar on a session, including securities no longer active today.

Study window: 2024-10-07 to 2026-07-01 (exclusive).
Daily-bar sources: grouped:massive_1Day_raw.

| Measure | Result |
|---|---:|
| Candidate symbols, prior | 6,537 |
| Candidate symbols, revised | 7,384 |
| Revised candidates with cached daily bars | 5,617 |
| Revised candidates missing cached daily bars | 1,767 |
| Inactive candidate symbols in revised pool | 1863 |
| Inactive candidates with cached daily bars | 268 |
| Sessions with a changed top 25 | 0/413 |
| Selected inactive ticker-sessions | 0 |

## Added symbols selected by the revised universe

No newly selected symbols had cached bars and displaced the prior top 25 in this window.

## Known delisted examples

| Symbol | In candidate pool | Daily bars cached | Selected | Delisted date |
|---|:---:|:---:|:---:|---|
| SIVB | no | no | no | — |
| FRC | no | no | no | — |
| BBBY | no | yes | no | — |

## Limits

Inactive-reference crawl coverage: 129,700 cached XNAS rows across 1,288 pages; ticker range ['A', 'EQFN']; complete=False

- Point-in-time grouped bars can include securities no longer active, but do not identify historical exchange membership or an inactive/delisting status by themselves.
- Only symbols with provider bars can be ranked; absent bars can still hide formerly liquid delisted stocks.
- Inactive is measured as of the current reference snapshot, not the security's status on each session.
- This audit corrects candidate coverage where data exists; it does not certify zero survivorship bias.
- The Massive inactive-common-stock crawl is incomplete: it contains only XNAS records through ticker EQFN, is alphabetically truncated, and has no XNYS/XASE inactive records.
- The candidate extension is therefore incomplete and non-random; results quantify only securities represented in this cached subset.
- Historical eligibility is based on point-in-time daily bars and the project liquidity rules, not certified historical exchange membership.
- The same Massive grouped daily bars are used for the current-candidate control and the expanded candidate universe to isolate candidate-pool changes.
