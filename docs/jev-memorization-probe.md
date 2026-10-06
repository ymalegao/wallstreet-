# JEV-9B memorization probe

## Frozen protocol (committed before inference)

This probe asks whether the pinned local JEV-9B System-1 decision head can identify the direction of completed monthly stock returns from ticker identity and month. It is a contamination diagnostic, not proof of the model's training cutoff and not a measure of trading skill.

The outcome months are January 2023 through September 2026. The label compares the final split/dividend-adjusted daily close in each month with the final adjusted close in the preceding month. The question set uses listed-stock candidates with complete cached month-end observations from December 2022 through September 2026. It balances `higher` and `lower` labels separately within every month, sampling the same number from each side without replacement and capping each class at 32. When one side is scarce, that month has a smaller sample; no duplicated labels are added. The universe and sampling seed are fixed in `configs/jev_memorization.yaml`.

Each named-ticker question is paired with an otherwise equivalent control that withholds the ticker identity. The month remains visible in both conditions, and options are consistently randomized within each matched pair. Chance is 50% because the evaluated questions are balanced within month. The local `jev` typed-choice path is used with the pinned revision `b63f651ce8ed64481d3f5e73ecdb05f740042f01`; no hosted model, fine-tuning, paid inference API, or order endpoint is used. Inference requests are cached locally.

Report accuracy by month with 95% Wilson score intervals, for both named and identity-masked prompts. For each month, test named accuracy against 50% with a one-sided exact binomial test; apply Holm-Bonferroni correction across all 45 months. The detected cutoff is the latest month with adjusted p < 0.05. The candidate clean window begins the next month. If no month is positive, set no cutoff and leave the clean window unestablished. A failure to reject chance is not evidence that the model is uncontaminated; a positive result is evidence of recoverable historical information, which can also arise through memorized sources or systematic priors.

The question set is deterministically built and hashed before inference. Its labels, prompts, month counts, ticker pool and hash are recorded in the run manifest. Inference must not start until the frozen question-set hash has been written to the manifest and protocol commit.

The frozen pre-inference set contains 1,778 questions over 63 tickers and 45 months; each question has a matched identity-masked control. Per-month sample sizes range from 10 to 62, with exactly equal counts for the two labels. Its SHA-256 is `9911d11ad7ebf91a8dfb626cc84ffac83cadede62eebb85e8d68f13942a04f68`.

## Results

Pending the frozen question-set build and local inference run.

## Limits

The local adjusted-bar cache supports the full requested period for a fixed set of 63 currently listed-stock candidates. This is a small, survivorship-affected probe universe, and several early months have few minority-class outcomes. No point-in-time EPS-consensus history is cached, so earnings-beat questions are excluded from this version. The probe can reveal a detectable memory signal; it cannot certify a clean month when accuracy is statistically indistinguishable from chance.
