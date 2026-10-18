# Yang External Evaluation Report (Comprehensive Audit)

**Experiment:** PipelineGuardian Learned Preprocessing Leakage Detector  
**Branch:** `exp/learned-detector`  
**Evaluated Checkpoint:** `checkpoints/baseline_rf.joblib` (SHA-256 frozen; not retrained or retuned)  
**Adapter Version:** `1.0.0` (frozen before Yang evaluation)  
**Ground Truth:** `eval/yang_corpus/ground_truth/ground-truth.csv` (`pre` column)  
**Rule Baseline:** `eval/yang_results_rule_only.json` (`pre_pred` column)  
**Evaluation Scripts:** `src/run_yang_evaluation.py` (v1 raw JSON) and `src/run_yang_evaluation_v2.py` (v2 adapter)  

---

## 1. Executive Summary

This report documents the rigorous, audit-proof evaluation of the frozen learned Random Forest detector on the external Yang et al. benchmark corpus.

### Key Milestones & Diagnoses:
1. **Initial Raw-JSON Input Evaluation (v1)**: Passing raw `.ipynb` JSON strings to the model resulted in zero AST features firing (P=0.000, R=0.000, F1=0.000). This established an **input-representation failure**, because the model was trained exclusively on `.py` Python script strings.
2. **Notebook Input Adapter (v2)**: A frozen input adapter (`NotebookAdapter` v1.0.0) extracts code cells in notebook order and strips IPython magics (`%`, `!`, `%%`). Round-trip synthetic tests verified equivalence on clean and leaky scripts before Yang evaluation.
3. **Corrected Yang Evaluation (v2)**: With normalized Python source extracted, the frozen Random Forest still predicts clean (`0`) for all 57 eligible Yang notebooks (P=0.000, R=0.000, F1=0.000). This confirms genuine **external domain shift** (complex multi-stage pipelines, custom transformations, external functions, non-standard naming conventions).
4. **Baselines Comparison on Identical 57 Notebooks**:
   - **Rule-Only Baseline**: Precision = **0.8333** (95% CI: [0.5520, 0.9530]), Recall = **0.5556** (95% CI: [0.3372, 0.7536]), **F1 = 0.6667**.
   - **Regex Heuristic Baseline**: Precision = **0.4516** (95% CI: [0.2917, 0.6224]), Recall = **0.7778** (95% CI: [0.5428, 0.9155]), **F1 = 0.5714**.
   - **Always-1 Baseline**: Precision = 0.3158, Recall = 1.0000, F1 = 0.4795.
   - **Always-0 Baseline**: Precision = 0.0000, Recall = 0.0000, F1 = 0.0000.
   - **Learned RF (Adapter v2)**: Precision = 0.0000, Recall = 0.0000, F1 = 0.0000 (Accuracy = 68.42%).

---

## 2. Ground-Truth & Eligible-Set Audit

All 100 notebooks in `eval/yang_corpus/notebooks/` were audited against `ground-truth.csv` and `task2_dev_ids.txt`:

| Category | Count | Criteria & Justification |
| :--- | :---: | :--- |
| **Total Notebooks in Corpus** | **100** | Full set of `.ipynb` files in local evaluation directory |
| **Excluded: Invalid JSON** | **1** | `2021-09-09-nb_334.ipynb` contains corrupted JSON bytes and cannot be loaded |
| **Excluded: Fixed Dev IDs** | **12** | Explicitly withheld development IDs from `eval/task2_dev_ids.txt` |
| **Excluded: Unlabeled in GT** | **30** | Present in `ground-truth.csv`, but `pre` column is blank (unlabeled for preprocessing leakage) |
| **Eligible Evaluation Subset** | **57** | Strictly held-out, validly formatted, unambiguously labeled test notebooks |
| — Ground-Truth Positives (`pre` = Y) | 18 | Confirmed preprocessing leakage cases (31.58%) |
| — Ground-Truth Negatives (`pre` = N) | 39 | Confirmed sound / clean preprocessing cases (68.42%) |

### Arithmetic Verification:
`57 eligible + 30 unlabeled + 12 dev excluded + 1 invalid JSON = 100 total corpus notebooks.`

---

## 3. Coverage & Abstention Accounting

Safe failure handling is enforced: notebooks that cannot be parsed or yield empty code cells trigger an explicit `ABSTAIN` outcome rather than silently defaulting to clean (`0`).

| Metric | Value | 95% Wilson Score Confidence Interval |
| :--- | :---: | :---: |
| Eligible Notebooks | 57 | — |
| Processed Notebooks | 57 | — |
| Abstained Notebooks | 0 | — |
| **Coverage Percentage** | **100.0%** | **[93.67%, 100.00%]** |
| Abstained Positives | 0 | — |

*Note on Conservative Accounting:* Because abstention count is 0 on the eligible set, headline processed metrics and conservative accounting (where abstained positives count as missed false negatives) are identical.

---

## 4. Benchmark Performance on Exactly the Same 57 Notebooks

Every method was evaluated on the **exact same 57 eligible notebook IDs** (`common_denominator_count = 57`).

| System / Baseline | TP | FP | FN | TN | Precision [95% Wilson CI] | Recall [95% Wilson CI] | F1 | Accuracy [95% Wilson CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rule-Only Baseline** | 10 | 2 | 8 | 37 | **0.8333** [0.5520, 0.9530] | **0.5556** [0.3372, 0.7536] | **0.6667** | **0.8246** [0.7061, 0.9029] |
| **Regex Heuristic** | 14 | 17 | 4 | 22 | 0.4516 [0.2917, 0.6224] | **0.7778** [0.5428, 0.9155] | **0.5714** | 0.6316 [0.5015, 0.7449] |
| **Always-Positive** | 18 | 39 | 0 | 0 | 0.3158 [0.2104, 0.4450] | 1.0000 [0.8241, 1.0000] | 0.4795 | 0.3158 [0.2104, 0.4450] |
| **Learned RF (Adapter v2)** | 0 | 0 | 18 | 39 | 0.0000 [—] | 0.0000 [0.0000, 0.1759] | 0.0000 | 0.6842 [0.5550, 0.7896] |
| **Learned RF (Raw JSON v1)**| 0 | 0 | 18 | 39 | 0.0000 [—] | 0.0000 [0.0000, 0.1759] | 0.0000 | 0.6842 [0.5550, 0.7896] |
| **Always-Negative** | 0 | 0 | 18 | 39 | 0.0000 [—] | 0.0000 [0.0000, 0.1759] | 0.0000 | 0.6842 [0.5550, 0.7896] |

---

## 5. Adapter Specification & Validation

The `NotebookAdapter` (`src/notebook_adapter.py`, version 1.0.0) extracts `.ipynb` code cells in chronological notebook execution order.

### Preprocessing Applied:
1. **Cell Filtering**: Extracts cells where `cell_type == "code"`. Ignores markdown, raw, and heading cells.
2. **Magics Normalization**: Strips IPython line magics (`%time`, `%matplotlib`, `%autoreload`) and cell magics (`%%time`, `%%writefile`), and comments out system shell calls (`!pip`, `!ls`) with `# [IPYTHON SHELL]`.
3. **Empty Cell Stripping**: Removes empty or whitespace-only code cells.
4. **Delimiter Joining**: Concatenates cell blocks with newlines.

### Documented Adapter Limitations:
- **Out-of-Order Execution**: Jupyter notebooks allow non-linear cell execution by interactive users. The adapter parses cells in sequential file order.
- **Dynamic Globals**: Variables defined in hidden cells, runtime environments, or interactive widgets are not statically analyzed.
- **Synthetic Round-Trip Equivalence**: 14 tests in `tests/test_notebook_adapter.py` confirm that synthetic `.py` scripts converted to `.ipynb` and extracted via `NotebookAdapter` produce identical AST features and model predictions to their original `.py` counterparts.

---

## 6. Diagnosis: Synthetic Generalization vs External Reality

### Why did the Learned RF achieve Macro-F1 = 1.000 on Synthetic Data but 0.000 on Yang?
1. **Structural AST Feature Invariance**: On synthetic data, the AST features (e.g. `fit_before_split_flag`, `split_to_fit_line_delta`) accurately reflected standard scikit-learn calls (`StandardScaler`, `MinMaxScaler`, `SimpleImputer`, `train_test_split`).
2. **Real-World Notebook Complexity**: In the Yang corpus:
   - Data preprocessing often uses manual loops, pandas expressions (`df[col] = (df[col] - df[col].mean()) / df[col].std()`), or custom utility functions.
   - Variable names rarely conform to standard synthetic identifiers (`X_train`, `X_test`, `df`).
   - Cross-validation splits use custom KFold loops or manual index slicing rather than straightforward `train_test_split(...)` single-call patterns.
   - When the AST visitor encounters these non-standard patterns, `split_line` or `fit_line` resolve to `None`, resulting in zeroed order flags.
3. **Conservative Decision Threshold**: The Random Forest was tuned with `min_samples_leaf=2` and class-balanced split logic. When ambiguous features are presented, the ensemble defaults to the negative majority class (68.4% of the real-world dataset).

---

## 7. Artifact Registry

| Component | Path | Description |
| :--- | :--- | :--- |
| **Adapter Module** | [`src/notebook_adapter.py`](file:///c:/Users/arush/OneDrive/Desktop/pipelineguardian/eval/learned_detector/src/notebook_adapter.py) | Frozen v1.0.0 `.ipynb` code cell extractor |
| **Adapter Tests** | [`tests/test_notebook_adapter.py`](file:///c:/Users/arush/OneDrive/Desktop/pipelineguardian/eval/learned_detector/tests/test_notebook_adapter.py) | 14 round-trip, magic, and edge case tests |
| **V1 Evaluation Script** | [`src/run_yang_evaluation.py`](file:///c:/Users/arush/OneDrive/Desktop/pipelineguardian/eval/learned_detector/src/run_yang_evaluation.py) | Initial raw-input evaluation script |
| **V1 Evaluation Result** | [`results/yang_scored_evaluation.json`](file:///c:/Users/arush/OneDrive/Desktop/pipelineguardian/eval/learned_detector/results/yang_scored_evaluation.json) | Raw-input representation failure artifact |
| **V2 Evaluation Script** | [`src/run_yang_evaluation_v2.py`](file:///c:/Users/arush/OneDrive/Desktop/pipelineguardian/eval/learned_detector/src/run_yang_evaluation_v2.py) | Corrected scored evaluation with baselines & Wilson CIs |
| **V2 Evaluation Result** | [`results/yang_scored_evaluation_v2.json`](file:///c:/Users/arush/OneDrive/Desktop/pipelineguardian/eval/learned_detector/results/yang_scored_evaluation_v2.json) | Complete benchmark results, manifest, and Wilson intervals |
| **V2 Test Suite** | [`tests/test_yang_evaluation_v2.py`](file:///c:/Users/arush/OneDrive/Desktop/pipelineguardian/eval/learned_detector/tests/test_yang_evaluation_v2.py) | 12 tests for Wilson intervals, baselines, and manifest math |
| **Synthetic Robustness Script** | [`src/run_synthetic_robustness.py`](file:///c:/Users/arush/OneDrive/Desktop/pipelineguardian/eval/learned_detector/src/run_synthetic_robustness.py) | Phase 5 ablation, contrast pairs, and cluster bootstrap |
| **Synthetic Robustness Report** | [`reports/synthetic_robustness_report.md`](file:///c:/Users/arush/OneDrive/Desktop/pipelineguardian/eval/learned_detector/reports/synthetic_robustness_report.md) | Comprehensive Phase 5 findings report |
| **Model Weights** | [`checkpoints/baseline_rf.joblib`](file:///c:/Users/arush/OneDrive/Desktop/pipelineguardian/eval/learned_detector/checkpoints/baseline_rf.joblib) | Frozen RF checkpoint (SHA-256 verified) |
