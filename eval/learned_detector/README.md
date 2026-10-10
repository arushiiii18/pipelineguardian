# Learned Leakage Detector (Experimental Branch)

## 1. Overview
This directory contains the isolated 10-day research experiment for the **Learned Preprocessing Leakage Detector** in PipelineGuardian.

### Objective
Binary classification of **preprocessing leakage** in ML pipelines:
- `1`: Preprocessing leakage present (e.g. preprocessor fitted over complete dataset or test set).
- `0`: No preprocessing leakage present (clean / sound ML pipeline).

### Isolation Guarantee
This experiment does **NOT** modify, replace, or invalidate PipelineGuardian's core 4 evaluation conditions (`rule_only`, `llm_only`, `llm_with_context`, `hybrid`). All experimental code, manifests, model weights, and benchmark outputs remain strictly inside `eval/learned_detector/`.

---

## 2. Model Ladder
1. **TF-IDF + Logistic Regression** (Baseline A)
2. **Random Forest** (Baseline B)
3. **AST Data-Flow Plain-PyTorch GIN** (Experimental Neural Model — pending Day 6 gate)

---

## 3. Dataset Architecture & Split Strategy
- **30 Clean Base Pipelines** (`base_01` to `base_30`) representing distinct realistic ML workflows (continuous scalers, imputers, ColumnTransformers, Pipelines, custom functions, aliased imports, cyclic features, etc.).
- **720 Generated Candidate Scripts** (24 mutations per base pipeline).
- **551 Independently Verified Examples** (330 positive leakage, 221 verified clean).
- **169 Rejected Examples** (conservatively rejected by the execution oracle where exact continuous statistics were unmeasurable or shifted).
- **Grouped Split Design**:
  - **Train/Validation Pool**: 24 base pipeline IDs (`base_01` to `base_24`), 439 verified cases (for GroupKFold cross-validation).
  - **Held-Out Synthetic Test Set**: 6 base pipeline IDs (`base_25` to `base_30`), 112 verified cases (zero base pipeline leakage into training).

---

## 4. Reproducibility Commands

All commands use the dedicated Python 3.12 virtual environment:

### Run Full Test Suite (GPU smoke test + Generator + Verifier):
```powershell
.\eval\learned_detector\venv\Scripts\pytest.exe eval\learned_detector\tests\
```

### Regenerate Synthetic Dataset & Run Execution Oracle:
```powershell
.\eval\learned_detector\venv\Scripts\python.exe eval\learned_detector\src\generator.py
```

### Re-run GPU Smoke Test:
```powershell
.\eval\learned_detector\venv\Scripts\python.exe eval\learned_detector\tests\test_gpu_smoke.py
```

### Train and Evaluate Baselines and GNN:
```powershell
.\eval\learned_detector\venv\Scripts\python.exe eval\learned_detector\src\baseline_logreg.py
.\eval\learned_detector\venv\Scripts\python.exe eval\learned_detector\src\baseline_rf.py
.\eval\learned_detector\venv\Scripts\python.exe eval\learned_detector\src\gnn_model.py
```

### Run Yang External Evaluation Contract (Currently Blocked):
```powershell
.\eval\learned_detector\venv\Scripts\python.exe eval\learned_detector\src\evaluate_yang_contract.py > eval\learned_detector\results\yang_evaluation_results.json
```
*(Note: Current output only contains unscored predictions on local notebooks. Genuine evaluation requires teammate ground-truth and baseline predictions.)*

---

## 5. Artifacts and Reports
- **Manifest Schema**: [`manifest_schema.json`](file:///c:/Users/arush/OneDrive/Desktop/pipelineguardian/eval/learned_detector/manifest_schema.json)
- **Dataset Manifest**: [`data/manifest.json`](file:///c:/Users/arush/OneDrive/Desktop/pipelineguardian/eval/learned_detector/data/manifest.json)
- **Generated Code Files**: `data/examples/`
- **GPU Smoke Test Report**: [`reports/gpu_verification.json`](file:///c:/Users/arush/OneDrive/Desktop/pipelineguardian/eval/learned_detector/reports/gpu_verification.json)
- **Label Verification Report**: [`reports/label_verification_report.md`](file:///c:/Users/arush/OneDrive/Desktop/pipelineguardian/eval/learned_detector/reports/label_verification_report.md)
- **Comparative Evaluation Report**: [`reports/comparative_evaluation_report.md`](file:///c:/Users/arush/OneDrive/Desktop/pipelineguardian/eval/learned_detector/reports/comparative_evaluation_report.md)
- **Yang Evaluation Results (Unscored Predictions)**: [`results/yang_evaluation_results.json`](file:///c:/Users/arush/OneDrive/Desktop/pipelineguardian/eval/learned_detector/results/yang_evaluation_results.json)
- **Manual Sample Audit (30 Cases)**: [`reports/manual_inspection_sample.md`](file:///c:/Users/arush/OneDrive/Desktop/pipelineguardian/eval/learned_detector/reports/manual_inspection_sample.md)
