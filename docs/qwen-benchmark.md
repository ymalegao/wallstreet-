# Zero-shot financial-text benchmark

Model: `qwen3.8-flash-next`. Revision: `local-unpinned`.

| Dataset | N | Accuracy | Macro F1 | Majority accuracy | Errors |
|---|---:|---:|---:|---:|---:|
| phrasebank | 2264 | 0.953 | 0.949 | 0.614 | 0 |
| fiqa | 234 | 0.846 | 0.716 | 0.590 | 0 |

## Interpretation

Public datasets may have been in model training; these are capability checks, not clean generalization.
Sentiment accuracy is not a trading profitability measurement.
PhraseBank is CC-BY-NC-SA-3.0; research use only. FiQA mirror declares MIT.

FiQA uses its test split with fixed ±0.1 score thresholds. PhraseBank uses the all-agree set.
Subset selection uses seed 42. No threshold/prompt was tuned on these evaluation labels.
