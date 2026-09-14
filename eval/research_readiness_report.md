# PipelineGuardian Research-Readiness Report

**Date**: September 14, 2026  
**Repository State**: Pre-Experiment Evaluation Infrastructure Complete  
**Four Architectures**: `RULE_ONLY`, `LLM_ONLY`, `LLM_WITH_CONTEXT`, `HYBRID` (Fully Preserved & Unaltered)

---

## Executive Summary
This report documents the audit, fixes, and ground-truth infrastructure created for PipelineGuardian to transition the codebase from initial engineering implementation to a rigorous, defensible academic experiment.

---

## 1. What Was Fixed in Evaluation Infrastructure
1. **McNemar's Significance Testing**:
   - Fixed runtime crash in `eval/yang_eval.py` caused by deprecated/removed `scipy.stats.binom_test`. Replaced with `scipy.stats.binomtest` with exact binomial fallbacks.
   - Handled zero discordant pairs ($n_{01} + n_{10} = 0$) gracefully without division-by-zero errors.
   - Added `compare_architectures()` helper to compute paired per-notebook McNemar significance tests across all 4 architectures on the exact same notebook set.
2. **Class Imbalance & Metrics**:
   - Added per-condition Macro-F1 computation across evaluated categories (`PREPROCESSING`, `OVERLAP`, `MULTI_TEST`).
   - Removed raw accuracy as a primary metric to prevent misleading claims on imbalanced datasets.
3. **Issue Schema Rigor**:
   - Updated `pipelineguardian/models.py` to make `Issue.category` (`Category`) and `Issue.source` (`IssueSource`) mandatory non-optional fields.
   - Updated all tools (`validation_strategy_reviewer.py`, `leakage_detector.py`, `overlap_detector.py`, `multitest_detector.py`, `reproducibility_checker.py`, `llm_only_auditor.py`, `llm_context_auditor.py`, `hybrid_router.py`) to emit strongly typed findings.
   - Updated full test suite (135/135 tests passing).

---

## 2. What Ground-Truth Infrastructure Now Exists
1. **Written Labeling Rubrics** (`eval/rubrics/`):
   - [`reproducibility_rubric.md`](rubrics/reproducibility_rubric.md): Explicit objective rules for Python seed, NumPy seed, PyTorch seed, `random_state`, and pinned dependencies.
   - [`validation_strategy_rubric.md`](rubrics/validation_strategy_rubric.md): Explicit objective rules for time-series forecasting splits, grouped entity splits, class imbalance stratification, and negative control scenarios.
   - [`agreement_protocol.md`](rubrics/agreement_protocol.md): Dual-annotator agreement protocol (Cohen's Kappa & % agreement) and blind self-relabeling fallback protocol.
2. **Curated Curated Benchmarks**:
   - [`eval/reproducibility_benchmark/ground_truth.json`](reproducibility_benchmark/ground_truth.json): 20 test cases covering positive and negative controls across all 5 reproducibility sub-checks.
   - [`eval/validation_benchmark/ground_truth.json`](validation_benchmark/ground_truth.json): 11 unambiguous test cases evaluating split strategy appropriateness against schema signals.
3. **Expanded Synthetic Fixture Suite**:
   - Expanded from 20 to 40 diverse fixtures (`eval/synthetic_notebooks/`), incorporating stylistic variants (`fit_transform`, `MinMaxScaler`, `RobustScaler`, `LabelEncoder`, `Pipeline`, `SimpleImputer`, `KMeans`, `KFold`, fillna variations).
   - Documented explicit AST parsing limitations (e.g. manual boolean mask splitting `df[mask]`, aliased feature lists) without adding ad-hoc heuristics.
4. **Real-World Benchmark Framework**:
   - [`eval/real_world_reproducibility/`](real_world_reproducibility/): Framework structure and README.
   - [`eval/ingest_real_world.py`](ingest_real_world.py): Ingestion script for `.ipynb` / `.py` candidate notebooks into benchmark format.
   - [`eval/compute_agreement.py`](compute_agreement.py): Utility for calculating inter-rater agreement and Cohen's Kappa.

---

## 3. What Can Already Be Experimentally Evaluated
- **Deterministic Rule Engine on Synthetic Benchmark (40 Fixtures)**: Precision=1.0, Recall=1.0, F1=1.0 on specified AST check scope.
- **Yang Corpus Categories (`PREPROCESSING`, `OVERLAP`, `MULTI_TEST`)**: Runnable across all 4 architectures using `python -m eval.yang_eval --condition <cond>`.
- **Curated Reproducibility & Validation Benchmarks**: Reproducibility and split strategy checks against frozen curated ground truth.
- **McNemar Significance Testing**: Paired notebook comparisons between `RULE_ONLY`, `LLM_ONLY`, `LLM_WITH_CONTEXT`, and `HYBRID`.

---

## 4. What Still Requires Manual Corpus Collection & Labeling
- **Real-World Public Notebook Corpus (40–60 Notebooks)**: Needs collection from Kaggle/GitHub using `ingest_real_world.py`.
- **Ground-Truth Annotation**: The collected real-world public notebooks must be manually annotated against `reproducibility_rubric.md` and frozen before running detectors.
- **Inter-Rater Agreement Execution**: Running dual annotator pass (or blind self-relabeling) via `compute_agreement.py` on a random 20% sample once corpus is populated.

---

## 5. What Claims PipelineGuardian Can Legitimately Make Right Now
- **System Architecture**: The 4-architecture research design (`RULE_ONLY`, `LLM_ONLY`, `LLM_WITH_CONTEXT`, `HYBRID`) is fully implemented, typed, and unit-tested (135/135 tests passing).
- **Rule Engine Generalization**: Rule engine successfully generalizes across diverse syntactic variations of pre-defined AST patterns (40 synthetic fixtures).
- **Context Isolation & Routing**: Structural AST facts are isolated from rule detectors, and hybrid routing correctly merges deterministic and LLM findings.
- **Reproducible Evaluation Protocol**: Significance testing (McNemar), ground-truth binarization, and category mapping are mathematically fixed and verifiable.

---

## 6. What Claims It CANNOT Make Yet
- **Real-World Generalization**: Cannot claim verified accuracy/F1 on real-world Kaggle/GitHub reproducibility until the real-world public notebook corpus is collected, labeled, and evaluated.
- **Statistical Superiority of Hybrid vs LLM-Only on Yang Corpus**: Cannot claim LLM or Hybrid superiority until the LLM API runs (3 runs per notebook) are executed on the Yang corpus and tested via `compare_architectures()`.
- **Unfalsifiable Quality Claims**: PipelineGuardian intentionally refrains from emitting single "quality scores", keeping findings transparently itemized by category.
