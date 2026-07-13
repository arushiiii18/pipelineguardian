# Evaluation Results

Scored against `synthetic_notebooks` (10 synthetic files, deterministic tools only — see eval.py docstring for why the LLM tool isn't scored here).

## Aggregate

- Precision: **1.0**
- Recall: **1.0**
- F1: **1.0**
- True positives: 6, False positives: 0, False negatives: 0

## Per-file breakdown

| File | Expected | Got | TP | FP | FN |
|---|---|---|---|---|---|
| 01_scaler_leak.py | {'scaler_fit_before_split': 1} | {'scaler_fit_before_split': 1} | 1 | 0 | 0 |
| 02_clean_scaler.py | {} | {} | 0 | 0 | 0 |
| 03_fillna_leak.py | {'fillna_uses_full_dataset_stat': 1} | {'fillna_uses_full_dataset_stat': 1} | 1 | 0 | 0 |
| 04_clean_fillna.py | {} | {} | 0 | 0 | 0 |
| 05_target_in_features.py | {'target_column_in_feature_list': 1} | {'target_column_in_feature_list': 1} | 1 | 0 | 0 |
| 06_clean_target.py | {} | {} | 0 | 0 | 0 |
| 07_missing_random_state.py | {'missing_random_state': 2} | {'missing_random_state': 2} | 2 | 0 | 0 |
| 08_missing_seed_torch.py | {'missing_seed_torch': 1} | {'missing_seed_torch': 1} | 1 | 0 | 0 |
| 09_clean_seeded_torch.py | {} | {} | 0 | 0 | 0 |
| 10_all_clean_baseline.py | {} | {} | 0 | 0 | 0 |
