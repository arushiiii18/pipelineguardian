# Yang-Corpus Evaluation Results

Condition: **llm_only**  | Notebooks scored: 100  | Skipped: 0  | **Macro-F1: 0.0513**

> [!NOTE]
> Categories REPRODUCIBILITY and VALIDATION_STRATEGY have no Yang ground-truth labels > and are evaluated via dedicated separate benchmarks.

## Per-Category Metrics

| Category | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| Preprocessing | 1.0 | 0.05 | 0.0952 | 1 | 0 | 19 |
| Overlap | 1.0 | 0.0 | 0.0 | 0 | 0 | 8 |
| Multi-test | 0.5 | 0.0312 | 0.0588 | 1 | 1 | 31 |

## LLM Variance (mean ± std F1 over runs)

- Preprocessing: 0.0952 ± 0.0
- Overlap: 0.0 ± 0.0
- Multi-test: 0.0588 ± 0.0
