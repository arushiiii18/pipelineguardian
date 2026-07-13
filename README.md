# PipelineGuardian

Most ML bugs aren't syntax errors. They're things like fitting a scaler before the train-test split, using a random split on timestamped data, or forgetting to seed experiments that are supposed to be reproducible. Traditional linters don't know anything about ML workflows, and most notebook environments won't warn you when you've introduced one of these.

This started as a simple leakage detector and gradually became an experiment in figuring out where deterministic tooling stops being enough and where LLM reasoning actually starts to help.

PipelineGuardian audits `.ipynb` notebooks and `.py` scripts using mostly deterministic checks, plus one narrow, targeted LLM call for the part that genuinely needs judgment rather than pattern-matching.

---

## What it checks

**Leakage Detector** *(deterministic, AST-based)* — scaler/encoder fit before `train_test_split`, dataset-wide statistics computed pre-split and reused, target column left in an explicit feature list.

**Reproducibility Checker** *(deterministic)* — missing `random_state`, unseeded NumPy/PyTorch randomness, missing or unpinned dependencies.

**Schema Inspector** *(deterministic)* — flags timestamp columns, group/ID columns, and target imbalance. Doesn't raise issues itself, just hands context to the next step.

**Validation Strategy Reviewer** *(LLM, Groq / `llama-3.3-70b-versatile` via LangChain)* — only runs when Schema Inspector finds something worth reasoning about. Judges whether the split strategy already in the code accounts for that signal, and explains why.

## Why an LLM at all?

Most of this project deliberately avoids one. Leakage and reproducibility checks are deterministic problems, handled with static analysis. The LLM only enters when there's genuine context to weigh:

> A random split on timestamped sales data is probably wrong — but a random split on a churn dataset that happens to have a timestamp column might be fine.

That distinction needs judgment, not a rule. This was the most interesting part of building it: **where deterministic analysis stops being enough and where LLM reasoning actually becomes useful.**

The decision to *call* the LLM is still a plain `if` statement based on Schema Inspector's output — the signals are objective. What the model actually does is the one real judgment call in the pipeline.

## Non-goals

No composite quality score — every issue stands on its own evidence. No claim to catch all leakage — a few specific patterns, listed above. No code style/PEP8 review. No cross-notebook experiment history.

## Output

Every finding is a typed object (`check_name`, `severity`, `confidence`, `evidence`, `suggested_fix`, plus line/cell location). `confidence` is either `deterministic` (exact match) or `inferred` (LLM judgment — currently only the validation-strategy check uses this).

## Evaluation

```bash
python -m eval.eval                        # deterministic checks: precision 1.00, recall 1.00, F1 1.00
python -m eval.validation_strategy_demo     # LLM tool, evaluated qualitatively
python -m eval.demo_before_after            # before: 5 issues → after: 0 issues
```

The perfect deterministic score is on 12 synthetic fixtures built to test these specific checks — it shows the pipeline works correctly, not that it generalizes to arbitrary real-world notebooks. The LLM tool is scored separately and qualitatively (reasoning printed, not reduced to a number), including a true-negative case — timestamp column present, split already correct — to check it isn't just flagging any date column on sight.

## Using it

```bash
# CLI — works standalone, no backend needed
python -m pipelineguardian.cli audit notebook.ipynb
python -m pipelineguardian.cli audit script.py --data data.csv --target churn --json

# API
uvicorn pipelineguardian.api:app --reload

# Frontend — plain HTML/JS, talks to the API directly
```

CLI exits `1` on any high-severity issue, `0` otherwise — usable as a CI gate.

## Stack

LangChain, Groq (`llama-3.3-70b-versatile`), Python AST, Pandas, Pydantic, FastAPI, Pytest, nbformat.

**Why Groq?** Free tier meant no billing setup during development. The judgment task is narrow enough that the open-weight model holds up well — checked against real cases, not assumed. Swapping to OpenAI is a one-line change given LangChain's shared interface.

## Setup

```bash
python -m venv venv && source venv/bin/activate   # venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY — free at console.groq.com
python -m pytest tests/ -v
```

## Live demo

Frontend: https://pipelineguardian-eight.vercel.app
Backend: https://pipelineguardian-api.onrender.com *(free tier — first request after idle takes ~30-50s to wake up)*

Try it with anything in `eval/synthetic_notebooks/` — `01_scaler_leak.py`, `07_missing_random_state.py`, `12_group_leak_random_split.py` for individual checks, or `demo_before.ipynb` / `demo_after.ipynb` for the full picture.

## What's next

Additional leakage patterns, experiment-tracking integration, richer validation-strategy checks, CI/CD integration for repo audits. Kept the scope narrow enough for now that every check here can be explained and defended.