# Zero-shot financial-text benchmark

Model: `jev-decision`. Revision: `b63f651ce8ed64481d3f5e73ecdb05f740042f01`.

| Dataset | N | Accuracy | Macro F1 | Majority accuracy | Errors |
|---|---:|---:|---:|---:|---:|
| phrasebank | 2264 | 0.882 | 0.877 | 0.614 | 0 |
| fiqa | 234 | 0.876 | 0.761 | 0.590 | 0 |

## Interpretation

Public datasets may have been in model training; these are capability checks, not clean generalization.
Sentiment accuracy is not a trading profitability measurement.
PhraseBank is CC-BY-NC-SA-3.0; research use only. FiQA mirror declares MIT.

FiQA uses its test split with fixed ±0.1 score thresholds. PhraseBank uses the all-agree set.
Subset selection uses seed 42. No threshold/prompt was tuned on these evaluation labels.
