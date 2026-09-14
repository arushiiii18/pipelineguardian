# Evaluation Results

Scored against `synthetic_notebooks` (40 synthetic files, deterministic tools only — see eval.py docstring for why the LLM tool isn't scored here).

## Aggregate

- Precision: **1.0**
- Recall: **1.0**
- F1: **1.0**
- True positives: 21, False positives: 0, False negatives: 0

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
| 21_manual_boolean_mask_split.py | {} | {} | 0 | 0 | 0 |
| 22_minmax_scaler_fit_transform.py | {'scaler_fit_before_split': 1} | {'scaler_fit_before_split': 1} | 1 | 0 | 0 |
| 23_robust_scaler_fit_then_transform.py | {'scaler_fit_before_split': 1} | {'scaler_fit_before_split': 1} | 1 | 0 | 0 |
| 24_label_encoder_before_split.py | {'scaler_fit_before_split': 1} | {'scaler_fit_before_split': 1} | 1 | 0 | 0 |
| 25_clean_pipeline_scaler.py | {} | {} | 0 | 0 | 0 |
| 26_imputer_fit_before_split.py | {'scaler_fit_before_split': 1} | {'scaler_fit_before_split': 1} | 1 | 0 | 0 |
| 27_clean_imputer_after_split.py | {} | {} | 0 | 0 | 0 |
| 28_fillna_median_full_dataset.py | {'fillna_uses_full_dataset_stat': 1} | {'fillna_uses_full_dataset_stat': 1} | 1 | 0 | 0 |
| 29_clean_fillna_train_only.py | {} | {} | 0 | 0 | 0 |
| 30_target_in_features_aliased.py | {} | {} | 0 | 0 | 0 |
| 31_missing_numpy_seed_rand.py | {'missing_seed_numpy': 1} | {'missing_seed_numpy': 1} | 1 | 0 | 0 |
| 32_clean_numpy_seed.py | {} | {} | 0 | 0 | 0 |
| 33_missing_random_state_kfold.py | {'missing_random_state': 1} | {'missing_random_state': 1} | 1 | 0 | 0 |
| 34_clean_random_state_kfold.py | {} | {} | 0 | 0 | 0 |
| 35_missing_random_state_kmeans.py | {'missing_random_state': 1} | {'missing_random_state': 1} | 1 | 0 | 0 |
| 36_overlap_no_split_eval_score.py | {'no_split_before_fit_eval': 1} | {'no_split_before_fit_eval': 1} | 1 | 0 | 0 |
| 37_overlap_clean_train_eval_split.py | {} | {} | 0 | 0 | 0 |
| 38_multitest_repeated_test_predict.py | {'repeated_test_evaluation': 1} | {'repeated_test_evaluation': 1} | 1 | 0 | 0 |
| 39_multitest_clean_validation_test.py | {'repeated_test_evaluation': 1} | {'repeated_test_evaluation': 1} | 1 | 0 | 0 |
| 40_reproducibility_missing_python_seed.py | {'missing_seed_python_random': 1} | {'missing_seed_python_random': 1} | 1 | 0 | 0 |
