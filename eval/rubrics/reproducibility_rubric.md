# Labeling Rubric: Reproducibility in ML Pipelines

## Purpose
This document defines objective labeling rules for evaluating reproducibility issues in Python machine learning code.

---

## 1. Missing Python Random Seed (`missing_python_seed`)

### Definition
Code imports or uses Python's standard `random` module for stochastic operations (e.g., `random.shuffle()`, `random.choice()`, `random.sample()`, `random.random()`) without calling `random.seed(...)` prior to execution.

- **Positive (Issue Present)**: `random` is imported and used in data generation, splitting, or sampling, but `random.seed(...)` is never set.
- **Negative (No Issue)**: `random.seed(...)` is called prior to standard `random` module calls, or `random` is never used for stochastic operations.

---

## 2. Missing NumPy Random Seed (`missing_numpy_seed`)

### Definition
Code uses `numpy.random` functions (e.g., `np.random.rand()`, `np.random.choice()`, `np.random.permutation()`, `np.random.default_rng()`) without initializing `np.random.seed(...)` or setting a seed on `default_rng(seed)`.

- **Positive (Issue Present)**: `np.random` calls occur without explicit seed initialization.
- **Negative (No Issue)**: `np.random.seed(seed)` or `default_rng(seed)` is explicitly set prior to numpy random calls.

---

## 3. Missing PyTorch Seed (`missing_pytorch_seed`)

### Definition
Code imports and uses `torch` for neural network operations (e.g. initializations, dropout, data loader shuffling) without setting `torch.manual_seed(...)` or setting seed on CUDA/backends.

- **Positive (Issue Present)**: `torch` tensors or neural network layers are created/trained without `torch.manual_seed(...)`.
- **Negative (No Issue)**: `torch.manual_seed(seed)` is called before model creation/training.

---

## 4. Missing Random State in Estimators/Splitters (`missing_random_state`)

### Definition
Code instantiates a stochastic estimator or splitter from `sklearn` (or `xgboost`/`lightgbm`) that accepts a `random_state` parameter, but leaves `random_state` unassigned (defaulting to `None`).

Examples of stochastic classes requiring `random_state`:
- `train_test_split`
- `KFold`, `StratifiedKFold`, `ShuffleSplit` (when `shuffle=True`)
- `RandomForestClassifier`, `RandomForestRegressor`
- `DecisionTreeClassifier`, `ExtraTreesClassifier`
- `KMeans`, `MiniBatchKMeans`
- `MLPClassifier`, `LogisticRegression` (when solver relies on random init or sampling)

- **Positive (Issue Present)**: A stochastic splitter or model is called/instantiated without setting `random_state`.
- **Negative (No Issue)**: `random_state=<int>` is explicitly specified, or the estimator is non-stochastic (e.g., `LinearRegression`, `StandardScaler`).

---

## 5. Missing / Unpinned Requirements (`unpinned_dependencies`)

### Definition
The code relies on external library installations via unpinned package directives (e.g., `!pip install pandas scikit-learn` without version pins like `==1.2.0`), or lacks any environment/requirements specification.

- **Positive (Issue Present)**: Inline `pip install` commands without version specifiers (e.g., `pip install sklearn`) or unpinned requirements files.
- **Negative (No Issue)**: Exact version pins are used (e.g., `pip install scikit-learn==1.3.0`) or environment lockfile is supplied.
