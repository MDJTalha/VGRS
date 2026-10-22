# Model evaluation utilities

Install the optional tooling with `python -m pip install -r ml/evaluation/requirements.txt` using an environment with access to the repository files.

## Speaker/group split

`python ml/evaluation/split_by_speaker.py --manifest data/manifest.csv --output data/manifest.with-split.csv --seed 42`

The CSV must contain `speaker_id` and the target `label`. The utility prefers `StratifiedGroupKFold` and falls back to `GroupShuffleSplit` if the available labels/groups do not support the stratified fold setup. It assigns the held-out test group first and then splits the remaining data into train/validation. Ratios are approximate because a speaker's recordings cannot be divided across partitions. If a `duplicate_group_id` column exists, linked duplicate recordings and their speakers are kept together. Populate that column after exact/near-duplicate checks; the script does not discover duplicate audio automatically.

## Prediction metrics

`python ml/evaluation/evaluate_predictions.py --predictions data/held-out-predictions.csv --output reports/metrics.json --true-col true_label --pred-col predicted_label --speaker-col speaker_id --bootstrap 1000 --subgroup-col accent_group`

The input requires one row per evaluated prediction, ground-truth/predicted class, and ideally speaker ID. Optional probability columns use the `prob_` prefix followed by the class label (for example `prob_child`, `prob_feminine`, `prob_masculine`). They must be finite values between 0 and 1 and sum to approximately one. Do not pass arbitrary logits as probabilities. Metrics include macro-F1, balanced accuracy, accuracy, per-class precision/recall/F1, confusion matrix, coverage under abstention, optional probability metrics, subgroup summaries, and optional speaker-cluster bootstrap intervals. Always report the denominators and evaluation split with the scores.

These utilities analyze supplied labels and predictions; they do not train a model, generate predictions or prove dataset quality. Maintain independent test speakers and perform data-rights, consent, duplicate, subgroup and label-quality reviews before claiming generalization.
