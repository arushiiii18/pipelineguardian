# Learned Leakage Detector (Experimental Milestone)

## 1. Overview
This directory contains the completed research experiment for the **Learned Preprocessing Leakage Detector** in PipelineGuardian.

### Objective
Binary classification of **preprocessing data leakage** in ML pipelines:
- `1`: Preprocessing leakage present (e.g. preprocessor fitted over complete dataset or test partition).
- `0`: No preprocessing leakage present (clean / sound ML pipeline).

### Isolation Guarantee
This experiment does **NOT** modify, replace, or invalidate PipelineGuardian's core 4 evaluation conditions (`rule_only`, `llm_only`, `llm_with_context`, `hybrid`) on `main`. All experimental code, manifests, model weights, and benchmark outputs remain strictly inside `eval/learned_detector/` on branch `exp/learned-detector`.

---

## 2. Models Evaluated
1. **TF-IDF + Logistic Regression** (Baseline A): 1000 n-gram features, C=1.0.
2. **Structural AST + TF-IDF Random Forest** (Baseline B): 17 structural AST features + 1000 n-grams, 50 estimators, max_depth=5.
3. **AST Data-Flow Plain-PyTorch GIN** (Neural Graph Model): Executed on NVIDIA GeForce RTX 2050 GPU (CUDA), 2 GIN layers, hidden_dim=64.
4. **Simple Regex Heuristic**: Sequence pattern check for `.fit(...)` calls preceding `train_test_split(...)` calls.
5. **Rule-Only Detector**: PipelineGuardian's deterministic rule engine evaluated on the identical notebook set.

---

## 3. Dataset & Robustness Summary
- **30 Base Pipelines** (`base_01` to `base_30`): 720 candidate variants generated, 551 verified sound by execution oracle, 169 rejected/inconclusive.
- **Grouped Partitioning**: 24 base pipelines (448 samples) in train/val pool; 6 base pipelines (103 samples) strictly held-out.
- **Feature Ablation**: When ordering/shortcut indicators are removed, Random Forest maintains **Held-out Macro-F1 = 0.9894** (CV Macro-F1 = 0.9767), showing genuine structural pattern recognition beyond line delta shortcuts.
- **Contrast Pair Sensitivity**: On 72 minimal contrast pairs, Random Forest achieves **94.4% pairwise flip accuracy** (100.0% for Logistic Regression).
- **Cluster Bootstrap 95% CIs** (resampled by base pipeline cluster, B=1000):
  - Random Forest Held-out Macro-F1: **0.9894** (95% CI: [0.9712, 1.0000])
  - Logistic Regression Held-out Macro-F1: **1.0000** (95% CI: [1.0000, 1.0000])
- **GNN Multi-Seed GPU Stability**: 5 seeds on CUDA: Macro-F1 = **0.8155 ± 0.0759**.

---

## 4. Yang External Evaluation Summary (57 Eligible Notebooks)

All methods evaluated on the exact same 57 eligible notebooks (`ground-truth.csv`, `pre` label column):

| System / Baseline | Precision [95% Wilson CI] | Recall [95% Wilson CI] | F1 | Accuracy [95% Wilson CI] | TP | FP | FN | TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rule-Only Baseline** | **0.8333** [0.5520, 0.9530] | **0.5556** [0.3372, 0.7536] | **0.6667** | **0.8246** [0.7061, 0.9029] | 10 | 2 | 8 | 37 |
| **Regex Heuristic** | 0.4516 [0.2917, 0.6224] | **0.7778** [0.5428, 0.9155] | **0.5714** | 0.6316 [0.5015, 0.7449] | 14 | 17 | 4 | 22 |
| **Always-Positive** | 0.3158 [0.2104, 0.4450] | 1.0000 [0.8241, 1.0000] | 0.4795 | 0.3158 [0.2104, 0.4450] | 18 | 39 | 0 | 0 |
| **Learned RF (Adapter v2)** | 0.0000 [—] | 0.0000 [0.0000, 0.1759] | 0.0000 | 0.6842 [0.5550, 0.7896] | 0 | 0 | 18 | 39 |
| **Learned RF (Raw JSON v1)**| 0.0000 [—] | 0.0000 [0.0000, 0.1759] | 0.0000 | 0.6842 [0.5550, 0.7896] | 0 | 0 | 18 | 39 |
| **Always-Negative** | 0.0000 [—] | 0.0000 [0.0000, 0.1759] | 0.0000 | 0.6842 [0.5550, 0.7896] | 0 | 0 | 18 | 39 |

### Domain Shift Diagnosis:
- **v1 Run Diagnosis**: Input-representation failure (passing raw `.ipynb` JSON strings prevented AST parsing).
- **v2 Run Diagnosis**: Validated input adapter normalized `.ipynb` source, but the frozen Random Forest (trained on synthetic scikit-learn patterns) defaulted to the negative majority class (accuracy 68.4%) on complex, non-standard real-world data science notebooks.

---

## 5. Exact Reproduction Commands

All commands execute in the dedicated Python 3.12 environment:

### Run Complete Test Suite (61 tests):
```powershell
.\eval\learned_detector\venv\Scripts\pytest.exe eval\learned_detector\tests\
```

### Run Notebook Adapter Unit & Round-Trip Tests:
```powershell
.\eval\learned_detector\venv\Scripts\pytest.exe eval\learned_detector\tests\test_notebook_adapter.py
```

### Run Corrected Yang External Evaluation (v2 with Wilson CIs):
```powershell
.\eval\learned_detector\venv\Scripts\python.exe eval\learned_detector\src\run_yang_evaluation_v2.py
```

### Run Phase 5 Synthetic Robustness, Ablation, and Multi-Seed GNN Checks:
```powershell
.\eval\learned_detector\venv\Scripts\python.exe eval\learned_detector\src\run_synthetic_robustness.py
```

### Train / Re-tune Baselines and GNN on GPU:
```powershell
.\eval\learned_detector\venv\Scripts\python.exe eval\learned_detector\src\baseline_logreg.py
.\eval\learned_detector\venv\Scripts\python.exe eval\learned_detector\src\baseline_rf.py
.\eval\learned_detector\venv\Scripts\python.exe eval\learned_detector\src\gnn_model.py
```

---

## 6. Key Artifacts & Reports Registry

| Artifact | Location |
| :--- | :--- |
| **Yang External Evaluation Report** | [`reports/yang_external_evaluation_report.md`](file:///c:/Users/arush/OneDrive/Desktop/pipelineguardian/eval/learned_detector/reports/yang_external_evaluation_report.md) |
| **Synthetic Robustness Report** | [`reports/synthetic_robustness_report.md`](file:///c:/Users/arush/OneDrive/Desktop/pipelineguardian/eval/learned_detector/reports/synthetic_robustness_report.md) |
| **Yang v2 Scored Evaluation Artifact** | [`results/yang_scored_evaluation_v2.json`](file:///c:/Users/arush/OneDrive/Desktop/pipelineguardian/eval/learned_detector/results/yang_scored_evaluation_v2.json) |
| **Yang v1 Scored Evaluation Artifact** | [`results/yang_scored_evaluation.json`](file:///c:/Users/arush/OneDrive/Desktop/pipelineguardian/eval/learned_detector/results/yang_scored_evaluation.json) |
| **Synthetic Robustness Artifact** | [`results/synthetic_robustness_results.json`](file:///c:/Users/arush/OneDrive/Desktop/pipelineguardian/eval/learned_detector/results/synthetic_robustness_results.json) |
| **Notebook Adapter Module** | [`src/notebook_adapter.py`](file:///c:/Users/arush/OneDrive/Desktop/pipelineguardian/eval/learned_detector/src/notebook_adapter.py) |
| **Comparative Model Report** | [`reports/comparative_evaluation_report.md`](file:///c:/Users/arush/OneDrive/Desktop/pipelineguardian/eval/learned_detector/reports/comparative_evaluation_report.md) |
| **Dataset Manifest** | [`data/manifest.json`](file:///c:/Users/arush/OneDrive/Desktop/pipelineguardian/eval/learned_detector/data/manifest.json) |
| **Model Checkpoints** | `checkpoints/baseline_rf.joblib`, `checkpoints/baseline_logreg.joblib`, `checkpoints/gnn_gin.pt` |
