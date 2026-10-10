# Yang External Evaluation Report

**Experiment:** Learned Preprocessing Leakage Detector  
**Branch:** `exp/learned-detector`  
**Evaluated checkpoint:** `checkpoints/baseline_rf.joblib` (frozen; not retrained)  
**Ground truth:** `eval/yang_corpus/ground_truth/ground-truth.csv` (`pre` column)  
**Rule baseline:** `eval/yang_results_rule_only.json` (`pre_pred` column)

---

## Eligible Notebook Set

| Category | Count | Notes |
|---|---|---|
| Total notebooks in corpus | 100 | |
| Excluded — invalid JSON | 1 | `2021-09-09-nb_334.ipynb` |
| Excluded — fixed dev IDs | 12 | From `eval/task2_dev_ids.txt` |
| No ground-truth entry | 30 | Local notebooks not in `ground-truth.csv` |
| **Eligible (included)** | **57** | Matched across corpus + ground truth + exclusions |
| Positive (leakage=1) | 18 | `pre` column starts with `Y` |
| Negative (leakage=0) | 39 | `pre` column starts with `N` |

Ground-truth label column: **`pre`** (preprocessing leakage). Labels are binary: `Y*` → 1, `N*` → 0.

---

## Results on 57 Eligible Notebooks

| Metric | Learned RF | Rule-Only Baseline |
|---|---|---|
| Precision | **0.000** | **0.833** |
| Recall | **0.000** | **0.556** |
| F1 | **0.000** | **0.667** |
| Support | 57 | 57 |
| TP | 0 | 10 |
| FP | 0 | 2 |
| FN | 18 | 8 |
| TN | 39 | 37 |

Both systems were evaluated on **identical eligible notebooks** (common denominator = 57).

---

## Root Cause: Domain Shift

The Random Forest was trained on synthetic Python `.py` scripts using structural AST features (split-line indices, `.fit()` call counts before/after the split, etc.). Yang notebooks are raw `.ipynb` JSON blobs.

When a `.ipynb` file is read as a plain string, none of the AST-based structural features fire — they all return 0. The TF-IDF vocabulary, fitted on synthetic scripts, also does not match the token distribution in ipynb JSON. As a result, the model predicts label=0 for every notebook.

**This is a training/evaluation distribution mismatch, not a labeling error.** The model's synthetic test results (Macro-F1 = 1.0) remain valid for the synthetic test distribution.

To correctly evaluate on Yang notebooks, one of the following changes is needed before evaluation (all requiring retraining from scratch on new data):
1. Extract cell source text from ipynb JSON before featurizing.
2. Retrain on ipynb-formatted examples, or convert synthetic examples to ipynb.
3. Use a representation that is agnostic to file format (e.g., token-level patterns over extracted source only).

**No retraining was performed here.** This report documents the current model's behavior on real-world Yang notebooks as-is.

---

## Artifact Paths

| Artifact | Path |
|---|---|
| Scored evaluation | `eval/learned_detector/results/yang_scored_evaluation.json` |
| Evaluation script | `eval/learned_detector/src/run_yang_evaluation.py` |
| Test suite | `eval/learned_detector/tests/test_yang_evaluation.py` |
| RF Checkpoint | `eval/learned_detector/checkpoints/baseline_rf.joblib` |
| Ground truth | `eval/yang_corpus/ground_truth/ground-truth.csv` |
| Rule baseline | `eval/yang_results_rule_only.json` |

---

## Remaining Blockers

None for this evaluation. The comparison between the learned model and the rule-only system on the same 57 eligible notebooks is complete. Future work (retraining on ipynb-formatted data) is independent and optional.
