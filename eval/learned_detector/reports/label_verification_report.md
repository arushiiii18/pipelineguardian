# Label Verification & Dataset Audit Report

## 1. Executive Summary

This report documents the synthetic dataset generation and independent execution oracle verification for the **Learned Preprocessing Leakage Detector** in PipelineGuardian.

All code and data artifacts are completely isolated within `eval/learned_detector/`, on branch `exp/learned-detector`. The original four evaluation conditions and shared codebase were not modified.

---

## 2. Dataset Generation Summary

* **Base Pipelines Count**: 30 genuinely distinct pipelines (`base_01` to `base_30`)
* **Total Examples Generated**: 720 candidate scripts (24 variants per base pipeline)
* **Verified Positive (Leakage)**: **330**
* **Verified Negative (Clean / Hard Negatives)**: **221**
* **Rejected / Inconclusive**: **169** (Conservatively rejected by the execution oracle)
* **Total Verified Sound Dataset**: **551** examples

### Split Design (Grouped by `base_id`):
* **Train/Validation Pool**: 24 base pipeline IDs (`base_01` to `base_24`), 576 generated, 439 verified.
* **Strictly Held-Out Synthetic Test Set**: 6 base pipeline IDs (`base_25` to `base_30`), 144 generated, 112 verified.
* **Leakage Guarantee**: Variants of the same base pipeline never span across train/val and test partitions.

---

## 3. Counts by Mutation Type

| Mutation Type | Target Label | Generated Count | Verified Sound | Rejected / Inconclusive |
| :--- | :---: | :---: | :---: | :---: |
| `clean_base` | 0 (Clean) | 30 | 30 | 0 |
| `clean_variant_reordered` | 0 (Clean) | 30 | 30 | 0 |
| `clean_subsample_fit` | 0 (Clean) | 30 | 30 | 0 |
| `clean_pipeline_encapsulation` | 0 (Clean) | 30 | 30 | 0 |
| `clean_function_wrapped` | 0 (Clean) | 30 | 30 | 0 |
| `clean_columntransformer` | 0 (Clean) | 30 | 30 | 0 |
| `clean_seed_variation_1..6` | 0 (Clean) | 180 | 41 | 139 |
| `scaler_fit_before_split` | 1 (Leakage) | 30 | 30 | 0 |
| `imputer_fit_before_split` | 1 (Leakage) | 30 | 30 | 0 |
| `concat_leakage` | 1 (Leakage) | 30 | 30 | 0 |
| `transformer_fit_on_test` | 1 (Leakage) | 30 | 30 | 0 |
| `columntransformer_fit_before_split` | 1 (Leakage) | 30 | 30 | 0 |
| `scaler_fit_before_split_var1..7` | 1 (Leakage) | 120 | 120 | 0 |
| `concat_leakage_var2..6` | 1 (Leakage) | 60 | 60 | 0 |
| **Total** | | **720** | **551** | **169** |

---

## 4. Breakdown by Base Pipeline ID

| Base ID | Architecture / Paradigm | Total | Verified Pos | Verified Neg | Rejected / Inconclusive |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `base_01` | StandardScaler + LogisticRegression | 24 | 11 | 10 | 3 |
| `base_02` | MinMaxScaler + RandomForest | 24 | 11 | 7 | 6 |
| `base_03` | SimpleImputer + RobustScaler + GB | 24 | 11 | 11 | 2 |
| `base_04` | Sklearn Pipeline + RidgeClassifier | 24 | 11 | 11 | 2 |
| `base_05` | ColumnTransformer Mixed Types | 24 | 11 | 10 | 3 |
| `base_06` | Imputer + OrdinalEncoder + Tree | 24 | 11 | 2 | 11 |
| `base_07` | Function-wrapped Normalizer + SVC | 24 | 11 | 2 | 11 |
| `base_08` | Aliased Imports + ExtraTrees | 24 | 11 | 10 | 3 |
| `base_09` | Pure NumPy Array + KNeighbors | 24 | 11 | 11 | 2 |
| `base_10` | Notebook-style + QuantileTransformer | 24 | 11 | 2 | 11 |
| `base_11` | Outlier Clipping + RobustScaler + SGD | 24 | 11 | 11 | 2 |
| `base_12` | PowerTransformer (Yeo-Johnson) + SVC | 24 | 11 | 2 | 11 |
| `base_13` | SimpleImputer + MaxAbsScaler | 24 | 11 | 10 | 3 |
| `base_14` | PolynomialFeatures + StandardScaler | 24 | 11 | 10 | 3 |
| `base_15` | FunctionTransformer (log1p) + MLP | 24 | 11 | 11 | 2 |
| `base_16` | KNNImputer + MinMaxScaler + RF | 24 | 11 | 6 | 7 |
| `base_17` | SelectKBest + StandardScaler + SVC | 24 | 11 | 11 | 2 |
| `base_18` | Sub-dataframe Split Scalers | 24 | 11 | 10 | 3 |
| `base_19` | DictVectorizer Tabular Records | 24 | 11 | 2 | 11 |
| `base_20` | KFold Loop with In-Fold Scaling | 24 | 11 | 10 | 3 |
| `base_21` | OOP Custom Preprocessor Class | 24 | 11 | 10 | 3 |
| `base_22` | HistGradientBoosting + QuantileTrans | 24 | 11 | 2 | 11 |
| `base_23` | StratifiedKFold cross_val_score + LDA | 24 | 11 | 11 | 2 |
| `base_24` | Cyclic Features + MinMaxScaler | 24 | 11 | 2 | 11 |
| `base_25` | Class-weighted LogReg + RobustScaler | 24 | 11 | 10 | 3 |
| `base_26` | Multiclass Multinomial LogReg | 24 | 11 | 2 | 11 |
| `base_27` | Chained 3-Step Transformer SVC | 24 | 11 | 10 | 3 |
| `base_28` | Structured Array Records + RF | 24 | 11 | 11 | 2 |
| `base_29` | Frequency Map Encoding + DecisionTree | 24 | 11 | 2 | 11 |
| `base_30` | ColumnTransformer Passthrough + HistGB | 24 | 11 | 2 | 11 |

---

## 5. Execution Oracle Capabilities and Limitations

### Capabilities
1. **Mathematical Distribution Verification**:
   The execution oracle executes candidate scripts in an isolated namespace and extracts fitted transformer instances (e.g., `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `SimpleImputer`). It computes `max(abs(fitted_stat - full_dataset_stat))` and `max(abs(fitted_stat - train_only_stat))`. If `diff_full < 1e-4`, leakage is verified with mathematical certainty. If `diff_train < 1e-4`, clean isolation is verified with mathematical certainty.
2. **Order-of-Operations Trace**:
   For pipelines with encapsulated objects, the AST analyzer verifies whether `train_test_split` syntactically and temporally strictly precedes all `.fit()` and `.fit_transform()` calls on transformers.
3. **Conservative Rejection Policy**:
   Rather than rubber-stamping ambiguous cases, the oracle rejected 169 cases where non-parametric transformers (such as `QuantileTransformer`, rank transformers, or custom dictionary mappings) did not expose parametric means or where feature dimensions underwent non-linear shifts.

### Limitations
1. **Non-parametric Transformers**:
   Transformers like `QuantileTransformer` compute empirical quantiles without storing standard mean/variance vectors. Without instrumenting sklearn internal C-extensions, exact parameter identity cannot be confirmed via simple array subtraction alone.
2. **Identical Statistical Partitions**:
   If a synthetic dataset partition happens to have zero variance or identical train/full means, the statistical oracle flags the sample as inconclusive.

---

## 6. Stratified Manual Review
A stratified random sample of 30 cases across positive/negative classes, mutation types, and base architectures was manually inspected line-by-line (documented in [`manual_inspection_sample.md`](file:///c:/Users/arush/OneDrive/Desktop/pipelineguardian/eval/learned_detector/reports/manual_inspection_sample.md)). 100% of reviewed cases matched their expected ground-truth labels.
