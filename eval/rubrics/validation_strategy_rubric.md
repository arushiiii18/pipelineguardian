# Labeling Rubric: Validation Strategy Mismatch

## Purpose
This document defines objective labeling rules for evaluating whether a dataset's train/test split strategy is appropriate given its inherent schema signals.

---

## 1. Time-Series / Temporal Signals (`timestamp_columns`)

### Scenario A: Temporal Data + Random Split -> INCORRECT
- **Condition**: Dataset contains temporal or time-series ordering (e.g. `timestamp`, `date`, `datetime`, `transaction_date`).
- **Observed Split**: Standard random `train_test_split(..., shuffle=True)` or standard random `KFold`.
- **Ground Truth**: **INCORRECT** (`is_appropriate: false`). Random splitting across time causes temporal lookahead leakage, where future information leaks into the training set to predict the past.

### Scenario B: Temporal Data + Temporal Split -> ACCEPTABLE
- **Condition**: Dataset contains temporal columns.
- **Observed Split**: Chronological slice (e.g., `train = df[df['date'] < '2023-01-01']`) or `TimeSeriesSplit`.
- **Ground Truth**: **ACCEPTABLE** (`is_appropriate: true`).

### Scenario C: Timestamp Column Present but Irrelevant -> ACCEPTABLE
- **Condition**: Dataset has a metadata timestamp (e.g. `created_at` in a static tabular metadata table where target is static user demographic, or timestamp is dropped prior to split and non-sequential).
- **Observed Split**: Standard `train_test_split`.
- **Ground Truth**: **ACCEPTABLE** (`is_appropriate: true`). If the timestamp is purely nominal or non-autocorrelated metadata, random splitting is valid.

---

## 2. Grouped Entity Signals (`group_columns`)

### Scenario A: Grouped Entities + Random Split -> INCORRECT
- **Condition**: Dataset contains multiple rows per entity/group (e.g. `patient_id`, `user_id`, `customer_id`, `group_id`).
- **Observed Split**: Standard row-wise random `train_test_split` or random `KFold`.
- **Ground Truth**: **INCORRECT** (`is_appropriate: false`). Row-level random splitting allows records from the same entity/patient/user to appear in both training and test sets, causing overlap leakage.

### Scenario B: Grouped Entities + Group-Aware Split -> ACCEPTABLE
- **Condition**: Dataset contains group columns.
- **Observed Split**: `GroupKFold`, `GroupShuffleSplit`, or explicit entity-level splitting (e.g., `train_users, test_users = ...`).
- **Ground Truth**: **ACCEPTABLE** (`is_appropriate: true`).

---

## 3. Class Imbalance Signals (`imbalance`)

### Scenario A: Highly Imbalanced Target + Standard Unstratified Split -> INCORRECT
- **Condition**: Target column exhibits severe class imbalance (e.g. minority class < 5% or 10% of dataset).
- **Observed Split**: Standard unstratified `train_test_split` or standard unstratified `KFold`.
- **Ground Truth**: **INCORRECT** (`is_appropriate: false`). Unstratified splits risk creating test folds with zero or near-zero instances of the minority class.

### Scenario B: Imbalanced Target + Stratified Split -> ACCEPTABLE
- **Condition**: Imbalanced target.
- **Observed Split**: `train_test_split(..., stratify=y)` or `StratifiedKFold`.
- **Ground Truth**: **ACCEPTABLE** (`is_appropriate: true`).
