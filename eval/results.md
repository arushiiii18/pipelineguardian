# Evaluation Results

Scored against `synthetic_notebooks` (20 synthetic files, deterministic tools only — see eval.py docstring for why the LLM tool isn't scored here).

## Aggregate

- Precision: **1.0**
- Recall: **1.0**
- F1: **1.0**
- True positives: 9, False positives: 0, False negatives: 0

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
| 11_timestamp_correct_split.py | {} | {} | 0 | 0 | 0 |
| 12_group_leak_random_split.py | {} | {} | 0 | 0 | 0 |
| 13_missing_seed_python_random.py | {'missing_seed_python_random': 1} | {'missing_seed_python_random': 1} | 1 | 0 | 0 |
| 14_clean_seeded_python_random.py | {} | {} | 0 | 0 | 0 |
| 15_overlap_no_split.py | {'no_split_before_fit_eval': 1} | {'no_split_before_fit_eval': 1} | 1 | 0 | 0 |
| 16_overlap_clean_split.py | {} | {} | 0 | 0 | 0 |
| 17_overlap_group_unaware.py | {} | {} | 0 | 0 | 0 |
| 18_overlap_group_aware.py | {} | {} | 0 | 0 | 0 |
| 19_multitest_repeated_peek.py | {'repeated_test_evaluation': 1} | {'repeated_test_evaluation': 1} | 1 | 0 | 0 |
| 20_multitest_cv_then_final_eval.py | {} | {} | 0 | 0 | 0 |
