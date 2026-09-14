# Real-World Reproducibility Benchmark Framework

## Overview
This benchmark evaluates PipelineGuardian's reproducibility detectors (`missing_seed_python_random`, `missing_seed_numpy`, `missing_seed_torch`, `missing_random_state`, `unpinned_dependencies`) on real-world public Jupyter notebooks collected from Kaggle and GitHub.

> [!IMPORTANT]
> **Corpus Population Status**: This directory defines the ingestion infrastructure, ground-truth schema, and annotation protocols. To maintain research validity, real-world public notebooks must be collected and independently labeled **prior** to running detector evaluations. Labels must remain frozen once established.

---

## Selection Criteria for Notebook Collection
To avoid selection bias, candidate notebooks should satisfy the following criteria:
1. **Source Diversity**: Collected from diverse Kaggle competitions (e.g. tabular, vision, NLP) and public GitHub repositories.
2. **Untouched History**: Notebooks must NOT have been examined during detector rule development.
3. **Realistic Workflows**: Must contain complete end-to-end ML workflows (data loading, preprocessing, model training, evaluation).
4. **Target Size**: 40–60 real public notebooks.

---

## Corpus Structure
```
eval/real_world_reproducibility/
├── README.md                  # Specification & protocol (this file)
├── ground_truth.json          # Frozen ground-truth labels
└── notebooks/                 # Ingested raw source code / notebooks
```

---

## Ground-Truth Labeling Protocol
1. Each ingested notebook is assigned a unique ID (e.g. `rw_nb_001`).
2. An independent human annotator reviews the notebook code against [`eval/rubrics/reproducibility_rubric.md`](../rubrics/reproducibility_rubric.md).
3. The annotator records labels for each sub-check:
   - `missing_python_seed`: boolean
   - `missing_numpy_seed`: boolean
   - `missing_pytorch_seed`: boolean
   - `missing_random_state`: boolean
   - `unpinned_dependencies`: boolean
4. Once labeled, `ground_truth.json` is frozen.
5. Detectors are executed against the frozen corpus to evaluate real-world Precision, Recall, and F1.

---

## Ingestion Utility
Use `eval/ingest_real_world.py` to ingest new candidate notebooks:
```bash
python -m eval.ingest_real_world --input_dir /path/to/raw_notebooks
```
