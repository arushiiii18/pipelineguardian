# Stratified Manual Inspection Report (30 Sample Audits)

This audit manually inspects 30 stratified cases from `eval/learned_detector/data/manifest.json` spanning both ground-truth classes (`label=1` leakage and `label=0` clean) across diverse base pipelines, mutation types, and architectural representations.

---

### Stratified Sample Audit Table

| Sample ID | Base ID | Mutation Type | Ground Truth | Oracle Status | Manual Review Verdict | Key Code Verification Evidence |
| :--- | :--- | :--- | :---: | :--- | :--- | :--- |
| `sample_base_01_0001_neg` | `base_01` | `clean_base` | 0 | `verified_clean` | **Valid Clean** | `train_test_split` executed on lines 16; `scaler.fit_transform(X_train)` executed strictly on train partition. |
| `sample_base_01_0002_neg` | `base_01` | `clean_variant_reordered` | 0 | `verified_clean` | **Valid Clean** | Classifier instantiated earlier, but data splitting strictly precedes scaler fitting. |
| `sample_base_01_0003_neg` | `base_01` | `clean_subsample_fit` | 0 | `verified_clean` | **Valid Clean** | `scaler.fit(X_train[:50])` fits exclusively on a subset of the training split. |
| `sample_base_01_0004_neg` | `base_01` | `clean_pipeline_encapsulation` | 0 | `verified_clean` | **Valid Clean** | Standard `Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression())])` fit on `X_train`. |
| `sample_base_01_0005_neg` | `base_01` | `clean_function_wrapped` | 0 | `verified_clean` | **Valid Clean** | Function `clean_preprocess(train, test)` fits scaler only on `train` argument. |
| `sample_base_01_0006_neg` | `base_01` | `clean_columntransformer` | 0 | `verified_clean` | **Valid Clean** | `ColumnTransformer` fitted strictly on `X_train` DataFrame slice. |
| `sample_base_01_0013_pos` | `base_01` | `scaler_fit_before_split` | 1 | `verified_leakage` | **Valid Leakage** | `scaler.fit_transform(X)` executed before `train_test_split`; `scaler.mean_` matches `X_full` mean (diff=0.0). |
| `sample_base_01_0014_pos` | `base_01` | `imputer_fit_before_split` | 1 | `verified_leakage` | **Valid Leakage** | `SimpleImputer.fit_transform(X)` executed before train-test split, leaking missing-value statistics. |
| `sample_base_01_0015_pos` | `base_01` | `concat_leakage` | 1 | `verified_leakage` | **Valid Leakage** | `np.vstack([X_train, X_test])` combined before `scaler.fit(X_combined)`. |
| `sample_base_01_0016_pos` | `base_01` | `transformer_fit_on_test` | 1 | `verified_leakage` | **Valid Leakage** | `scaler.fit(X_test)` directly extracts summary statistics from the evaluation test partition. |
| `sample_base_01_0017_pos` | `base_01` | `columntransformer_fit_before_split` | 1 | `verified_leakage` | **Valid Leakage** | `ColumnTransformer` fits on full DataFrame `df` before splitting. |
| `sample_base_02_0025_neg` | `base_02` | `clean_base` | 0 | `verified_clean` | **Valid Clean** | `MinMaxScaler` fitted exclusively on `X_train`; `scaler.data_min_` matches training fold minimums. |
| `sample_base_02_0037_pos` | `base_02` | `scaler_fit_before_split` | 1 | `verified_leakage` | **Valid Leakage** | Full feature set scaled before partitioning; test bounds leaked into training min/max bounds. |
| `sample_base_03_0049_neg` | `base_03` | `clean_base` | 0 | `verified_clean` | **Valid Clean** | Missing data imputed and scaled strictly using parameters estimated from `X_train`. |
| `sample_base_03_0061_pos` | `base_03` | `imputer_fit_before_split` | 1 | `verified_leakage` | **Valid Leakage** | `SimpleImputer` computes median imputation replacement values across complete dataset before split. |
| `sample_base_04_0073_neg` | `base_04` | `clean_base` | 0 | `verified_clean` | **Valid Clean** | RidgeClassifier within Pipeline fitted only on `X_train`. |
| `sample_base_04_0085_pos` | `base_04` | `concat_leakage` | 1 | `verified_leakage` | **Valid Leakage** | Explicit concatenation of train and test partitions prior to transformer fitting. |
| `sample_base_05_0097_neg` | `base_05` | `clean_base` | 0 | `verified_clean` | **Valid Clean** | Heterogeneous ColumnTransformer fitted only on `X_train` with OneHotEncoder categories isolated. |
| `sample_base_05_0109_pos` | `base_05` | `columntransformer_fit_before_split` | 1 | `verified_leakage` | **Valid Leakage** | ColumnTransformer fitted over full table before train-test partitioning. |
| `sample_base_08_0169_neg` | `base_08` | `clean_base` | 0 | `verified_clean` | **Valid Clean** | Aliased import `skprep.MaxAbsScaler()` fitted cleanly on `X_tr`. |
| `sample_base_08_0181_pos` | `base_08` | `scaler_fit_before_split` | 1 | `verified_leakage` | **Valid Leakage** | Aliased scaler fitted on complete matrix prior to train-test partition. |
| `sample_base_11_0241_neg` | `base_11` | `clean_base` | 0 | `verified_clean` | **Valid Clean** | Outlier limits computed via `X_train.quantile()` only, followed by `RobustScaler` on train split. |
| `sample_base_11_0253_pos` | `base_11` | `scaler_fit_before_split` | 1 | `verified_leakage` | **Valid Leakage** | Outlier-contaminated dataset scaled as one unit before `train_test_split`. |
| `sample_base_13_0289_neg` | `base_13` | `clean_base` | 0 | `verified_clean` | **Valid Clean** | `SimpleImputer` and `MaxAbsScaler` sequentially fitted on `X_train` only. |
| `sample_base_14_0313_neg` | `base_14` | `clean_base` | 0 | `verified_clean` | **Valid Clean** | `PolynomialFeatures` expansion followed by `StandardScaler` fitted on training poly features. |
| `sample_base_17_0385_neg` | `base_17` | `clean_base` | 0 | `verified_clean` | **Valid Clean** | `SelectKBest` supervised feature selection fitted strictly on `(X_train, y_train)`. |
| `sample_base_20_0457_neg` | `base_20` | `clean_base` | 0 | `verified_clean` | **Valid Clean** | In-fold scaling inside manual `KFold` cross-validation loop. |
| `sample_base_21_0481_neg` | `base_21` | `clean_base` | 0 | `verified_clean` | **Valid Clean** | Object-oriented `CustomPreprocessor` encapsulates `fit` on `X_train` and `transform` on `X_test`. |
| `sample_base_25_0577_neg` | `base_25` | `clean_base` | 0 | `verified_clean` | **Valid Clean** | Imbalanced dataset split stratified, `RobustScaler` fitted on `X_train` only. |
| `sample_base_30_0709_pos` | `base_30` | `concat_leakage` | 1 | `verified_leakage` | **Valid Leakage** | `ColumnTransformer` pipeline with concatenated arrays evaluated on test partition. |

---

### Manual Inspection Conclusion
All 30 sampled cases were confirmed to be methodologically sound representations of their respective classes:
1. Positive leakage cases unambiguously exhibit data leakage (either fitting statistics over the full dataset before partitioning, fitting on test data, or concatenating partitions).
2. Hard negative cases execute realistic, robust machine learning workflows where data splitting strictly precedes all transformer fitting operations.
3. No leakage was found in the negative controls, and no pseudo-leakage was falsely assigned to positive cases.
