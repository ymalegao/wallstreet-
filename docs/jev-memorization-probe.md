# JEV-9B memorization probe

## Frozen protocol (committed before inference)

This probe asks whether the pinned local JEV-9B System-1 decision head can identify the direction of completed monthly stock returns from ticker identity and month. It is a contamination diagnostic, not proof of the model's training cutoff and not a measure of trading skill.

The outcome months are January 2023 through September 2026. The label compares the final split/dividend-adjusted daily close in each month with the final adjusted close in the preceding month. The question set uses listed-stock candidates with complete cached month-end observations from December 2022 through September 2026. It balances `higher` and `lower` labels separately within every month, sampling the same number from each side without replacement and capping each class at 32. When one side is scarce, that month has a smaller sample; no duplicated labels are added. The universe and sampling seed are fixed in `configs/jev_memorization.yaml`.

Each named-ticker question is paired with an otherwise equivalent control that withholds the ticker identity. The month remains visible in both conditions, and options are consistently randomized within each matched pair. Chance is 50% because the evaluated questions are balanced within month. The local `jev` typed-choice path is used with the pinned revision `b63f651ce8ed64481d3f5e73ecdb05f740042f01`; no hosted model, fine-tuning, paid inference API, or order endpoint is used. Inference requests are cached locally.

Report accuracy by month with 95% Wilson score intervals, for both named and identity-masked prompts. For each month, test named accuracy against 50% with a one-sided exact binomial test; apply Holm-Bonferroni correction across all 45 months. The detected cutoff is the latest month with adjusted p < 0.05. The candidate clean window begins the next month. If no month is positive, set no cutoff and leave the clean window unestablished. A failure to reject chance is not evidence that the model is uncontaminated; a positive result is evidence of recoverable historical information, which can also arise through memorized sources or systematic priors.

The question set is deterministically built and hashed before inference. Its labels, prompts, month counts, ticker pool and hash are recorded in the run manifest. Inference must not start until the frozen question-set hash has been written to the manifest and protocol commit.

The frozen pre-inference set contains 1,778 questions over 63 tickers and 45 months; each question has a matched identity-masked control. Per-month sample sizes range from 10 to 62, with exactly equal counts for the two labels. Its SHA-256 is `9911d11ad7ebf91a8dfb626cc84ffac83cadede62eebb85e8d68f13942a04f68`.

## Results

Revision `b63f651ce8ed64481d3f5e73ecdb05f740042f01`; 1778 questions; set SHA-256 `9911d11ad7ebf91a8dfb626cc84ffac83cadede62eebb85e8d68f13942a04f68`.
Pooled named accuracy: **51.8%**; identity-masked control: **50.1%**; chance: 50%.

Detected cutoff month: **not detected**. Candidate clean-window start: **unestablished**.

A positive result indicates detectable above-chance recall on this probe. No positive result does not establish that the model lacks relevant training overlap.

| Month | n / condition | Named accuracy (95% CI) | Identity-masked accuracy (95% CI) | Holm-adjusted p | Flagged |
|---|---:|---:|---:|---:|:---:|
| 2023-01 | 10 | 0.500 [0.237, 0.763] | 0.500 [0.237, 0.763] | 1 | no |
| 2023-02 | 48 | 0.542 [0.403, 0.674] | 0.500 [0.364, 0.636] | 1 | no |
| 2023-03 | 30 | 0.533 [0.361, 0.698] | 0.500 [0.332, 0.668] | 1 | no |
| 2023-04 | 58 | 0.569 [0.441, 0.688] | 0.500 [0.375, 0.625] | 1 | no |
| 2023-05 | 42 | 0.548 [0.399, 0.688] | 0.500 [0.355, 0.645] | 1 | no |
| 2023-06 | 22 | 0.545 [0.347, 0.731] | 0.500 [0.307, 0.693] | 1 | no |
| 2023-07 | 28 | 0.536 [0.358, 0.705] | 0.500 [0.326, 0.674] | 1 | no |
| 2023-08 | 48 | 0.625 [0.484, 0.748] | 0.542 [0.403, 0.674] | 1 | no |
| 2023-09 | 20 | 0.450 [0.258, 0.658] | 0.500 [0.299, 0.701] | 1 | no |
| 2023-10 | 38 | 0.553 [0.397, 0.699] | 0.500 [0.348, 0.652] | 1 | no |
| 2023-11 | 10 | 0.500 [0.237, 0.763] | 0.500 [0.237, 0.763] | 1 | no |
| 2023-12 | 26 | 0.500 [0.321, 0.679] | 0.500 [0.321, 0.679] | 1 | no |
| 2024-01 | 44 | 0.614 [0.466, 0.743] | 0.500 [0.358, 0.642] | 1 | no |
| 2024-02 | 28 | 0.571 [0.391, 0.735] | 0.500 [0.326, 0.674] | 1 | no |
| 2024-03 | 48 | 0.479 [0.345, 0.617] | 0.500 [0.364, 0.636] | 1 | no |
| 2024-04 | 22 | 0.545 [0.347, 0.731] | 0.500 [0.307, 0.693] | 1 | no |
| 2024-05 | 26 | 0.423 [0.255, 0.611] | 0.500 [0.321, 0.679] | 1 | no |
| 2024-06 | 38 | 0.579 [0.422, 0.721] | 0.500 [0.348, 0.652] | 1 | no |
| 2024-07 | 56 | 0.518 [0.390, 0.643] | 0.500 [0.373, 0.627] | 1 | no |
| 2024-08 | 62 | 0.516 [0.394, 0.636] | 0.500 [0.379, 0.621] | 1 | no |
| 2024-09 | 52 | 0.481 [0.351, 0.613] | 0.500 [0.369, 0.631] | 1 | no |
| 2024-10 | 62 | 0.516 [0.394, 0.636] | 0.500 [0.379, 0.621] | 1 | no |
| 2024-11 | 26 | 0.423 [0.255, 0.611] | 0.500 [0.321, 0.679] | 1 | no |
| 2024-12 | 44 | 0.545 [0.401, 0.683] | 0.500 [0.358, 0.642] | 1 | no |
| 2025-01 | 38 | 0.553 [0.397, 0.699] | 0.500 [0.348, 0.652] | 1 | no |
| 2025-02 | 46 | 0.587 [0.443, 0.717] | 0.500 [0.361, 0.639] | 1 | no |
| 2025-03 | 14 | 0.500 [0.268, 0.732] | 0.500 [0.268, 0.732] | 1 | no |
| 2025-04 | 44 | 0.568 [0.422, 0.703] | 0.500 [0.358, 0.642] | 1 | no |
| 2025-05 | 26 | 0.577 [0.389, 0.745] | 0.500 [0.321, 0.679] | 1 | no |
| 2025-06 | 26 | 0.577 [0.389, 0.745] | 0.500 [0.321, 0.679] | 1 | no |
| 2025-07 | 54 | 0.519 [0.389, 0.646] | 0.500 [0.371, 0.629] | 1 | no |
| 2025-08 | 50 | 0.340 [0.224, 0.478] | 0.500 [0.366, 0.634] | 1 | no |
| 2025-09 | 36 | 0.500 [0.345, 0.655] | 0.500 [0.345, 0.655] | 1 | no |
| 2025-10 | 50 | 0.400 [0.276, 0.538] | 0.500 [0.366, 0.634] | 1 | no |
| 2025-11 | 44 | 0.455 [0.317, 0.599] | 0.500 [0.358, 0.642] | 1 | no |
| 2025-12 | 52 | 0.519 [0.387, 0.649] | 0.500 [0.369, 0.631] | 1 | no |
| 2026-01 | 56 | 0.482 [0.357, 0.610] | 0.500 [0.373, 0.627] | 1 | no |
| 2026-02 | 54 | 0.426 [0.303, 0.558] | 0.500 [0.371, 0.629] | 1 | no |
| 2026-03 | 22 | 0.591 [0.387, 0.767] | 0.500 [0.307, 0.693] | 1 | no |
| 2026-04 | 26 | 0.538 [0.355, 0.712] | 0.500 [0.321, 0.679] | 1 | no |
| 2026-05 | 42 | 0.595 [0.445, 0.730] | 0.500 [0.355, 0.645] | 1 | no |
| 2026-06 | 54 | 0.426 [0.303, 0.558] | 0.500 [0.371, 0.629] | 1 | no |
| 2026-07 | 52 | 0.635 [0.499, 0.752] | 0.500 [0.369, 0.631] | 1 | no |
| 2026-08 | 48 | 0.542 [0.403, 0.674] | 0.500 [0.364, 0.636] | 1 | no |
| 2026-09 | 56 | 0.446 [0.324, 0.576] | 0.500 [0.373, 0.627] | 1 | no |

A detected month flags above-chance historical recall under this probe. An undetected month does not certify absence of training overlap. The proposed clean start is a conservative test boundary, not proof.

## Limits
The local adjusted-bar cache supports the full requested period for a fixed set of 63 currently listed-stock candidates. This is a small, survivorship-affected probe universe, and several early months have few minority-class outcomes. No point-in-time EPS-consensus history is cached, so earnings-beat questions are excluded from this version. The probe can reveal a detectable memory signal; it cannot certify a clean month when accuracy is statistically indistinguishable from chance.
