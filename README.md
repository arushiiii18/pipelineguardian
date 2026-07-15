# PipelineGuardian

Most ML bugs aren't syntax errors.

They're things like fitting a scaler before a train-test split, using a random split on timestamped data, leaking grouped entities across folds, or forgetting to seed experiments that are supposed to be reproducible.

Traditional linters don't understand ML workflows, and notebooks rarely warn you when you've accidentally introduced one of these problems.

PipelineGuardian is an ML workflow auditor for `.ipynb` notebooks and `.py` scripts that combines deterministic static analysis with one narrowly scoped LLM judgment step for cases where context genuinely matters.

The goal wasn't to build another "AI code reviewer." It was to explore where deterministic tooling stops being enough and where reasoning actually becomes necessary.

---

## What it checks

### Leakage Detector *(deterministic, AST-based)*

Detects common leakage patterns including:

- `StandardScaler`, encoders, or transformers fitted before `train_test_split`
- Dataset-wide statistics computed before splitting and reused later
- Target columns accidentally included in explicit feature lists

---

### Reproducibility Checker *(deterministic)*

Flags issues such as:

- Missing `random_state`
- Unseeded NumPy or PyTorch randomness
- Missing or unpinned dependencies

---

### Schema Inspector *(deterministic)*

Extracts contextual signals from datasets, including:

- Timestamp columns
- Group or entity identifiers
- Target imbalance

The inspector itself does not raise findings — it only provides context for downstream reasoning.

---

### Validation Strategy Reviewer *(conditional LLM reasoning)*

Runs only when schema signals introduce ambiguity that static rules alone cannot resolve.

Examples:

- A random split on timestamped sales data is probably incorrect.
- A churn dataset that merely contains a timestamp column may still be perfectly fine with a random split.

The difference requires judgment rather than pattern matching.

PipelineGuardian uses `llama-3.3-70b-versatile` through Groq and LangChain for this single task only.

---

## Why use an LLM at all?

Most of PipelineGuardian deliberately avoids one.

Leakage detection and reproducibility checks are deterministic problems and are handled through static analysis.

The LLM only participates when contextual reasoning becomes unavoidable.

Importantly, the decision to invoke the model is itself deterministic:

```python
if schema_contains_contextual_signal:
    run_validation_review()
```

The schema signals are objective.

The judgment about whether the existing validation strategy already accounts for those signals is the only part delegated to the model.

---

## Architecture

```text
Notebook / Script
        │
        ▼
 AST Parser
        │
        ▼
 Leakage Detector
        │
        ▼
 Reproducibility Checker
        │
        ▼
 Schema Inspector
        │
        ├── No contextual signal → finish
        │
        ▼
 Validation Strategy Reviewer (LLM)
        │
        ▼
 Typed Findings
```

---

## Output

Every finding is returned as a typed object containing:

- `check_name`
- `severity`
- `confidence_source`
- `evidence`
- `suggested_fix`
- `line_number` or `cell_number`

Confidence is intentionally simple:

- `deterministic` → exact rule match
- `inferred` → contextual LLM judgment

---

## Evaluation

Deterministic checks were validated against synthetic fixtures covering every supported failure mode.

```bash
python -m eval.eval
```

Validation strategy review is evaluated separately through representative scenarios rather than aggregate metrics:

```bash
python -m eval.validation_strategy_demo
```

Example scenarios include:

- Timestamp column + random split → flagged
- Timestamp column + temporal split → accepted
- Timestamp column present but irrelevant → accepted
- Grouped entities + random split → flagged

The goal was not to maximize benchmark numbers but to ensure every supported check behaves as intended.

---

## Usage

### CLI

```bash
python -m pipelineguardian.cli audit notebook.ipynb

python -m pipelineguardian.cli audit script.py \
    --data data.csv \
    --target churn \
    --json
```

CLI exits with:

- `0` → no high-severity findings
- `1` → one or more high-severity findings

making it usable as a CI gate.

---

### API

```bash
uvicorn pipelineguardian.api:app --reload
```

---

### Frontend

The frontend is intentionally minimal and communicates directly with the FastAPI backend.

---

## Stack

- Python AST
- FastAPI
- Pydantic
- Pandas
- nbformat
- Pytest
- LangChain
- Groq
- `llama-3.3-70b-versatile`

---

## Limitations

PipelineGuardian intentionally keeps its scope narrow.

Current limitations include:

- Focuses on a small set of well-defined leakage patterns
- Assumes sklearn-style workflows
- Does not execute code
- Does not inspect imported helper functions
- Does not track experiment history across notebooks

Every supported check is designed to be explainable, evidence-backed, and defensible.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate

# Windows
venv\Scripts\Activate.ps1

pip install -r requirements.txt

cp .env.example .env
# Add your GROQ_API_KEY

pytest tests/ -v
```

---

## Live Demo

Frontend:

https://pipelineguardian-eight.vercel.app/

Backend:

https://pipelineguardian-api.onrender.com/

> The backend runs on Render's free tier and may take 30-50 seconds to wake up after inactivity.

Try it with the examples inside `eval/synthetic_notebooks/`:

- `01_scaler_leak.py`
- `07_missing_random_state.py`
- `12_group_leak_random_split.py`
- `demo_before.ipynb`
- `demo_after.ipynb`

---

## Future Work

- Additional leakage patterns
- Richer validation strategy checks
- Experiment tracking integration
- CI/CD and pull request integration
- Repository-wide auditing

The scope was intentionally kept narrow enough that every result produced by the system can be explained and defended.
