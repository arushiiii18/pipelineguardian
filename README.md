# PipelineGuardian

An LLM-orchestrated auditor for ML notebooks and scripts. It catches engineering
correctness bugs that standard linters don't — data leakage, reproducibility
gaps, and validation-strategy mismatches — using an LLM as a **tool-calling
orchestrator**, not a narrator that describes code in prose.

## What it actually does

Given a `.ipynb` or `.py` file, PipelineGuardian runs:

1. **`leakage_detector`** (AST, deterministic) — scaler/encoder `.fit()` before
   `train_test_split`, `fillna()` using a full-dataset statistic computed
   pre-split, and a target column left in an explicit feature list.
2. **`reproducibility_checker`** (AST + regex, deterministic) — missing
   `random_state`, unseeded `torch`/`numpy` randomness, missing or unpinned
   `requirements.txt`.
3. **`schema_inspector`** (pandas, deterministic) — detects timestamp columns,
   group/ID columns, and class imbalance in the target. This tool doesn't
   report issues; it produces **signals** that the next step reads.
4. **`validation_strategy_reviewer`** (LLM, Groq/`llama-3.3-70b-versatile`) —
   **only invoked if `schema_inspector` found a timestamp column, group
   column, or meaningful class imbalance.** It judges whether the split
   strategy already used in the code accounts for that signal, and explains
   why if not.

Steps 1–2 always run. Step 4 is conditional on step 3's output — that
conditional branch is the actual reason this project uses an agent
orchestration layer instead of being a plain linter script. See
[Why LangChain](#why-langchain-and-not-just-a-script) below for the honest
version of that story.

## Non-goals

Stated up front, not left for someone to dig up:

- **No composite "quality score."** Every issue is reported individually with
  its own severity, confidence tier, and evidence. A single number would
  hide exactly the tradeoffs this tool exists to surface.
- **Not exhaustive leakage detection.** `leakage_detector` catches three
  specific, common patterns (see above) — not "all leakage." Real leakage
  can be far subtler than AST patterns can catch.
- **No experiment-history reconstruction** across notebook versions.
- **No general code style / PEP8 review.** Solved territory, not the point.

## Output format

Every issue is a structured object (Pydantic model, see `pipelineguardian/models.py`):

```python
class Issue(BaseModel):
    check_name: str
    severity: Literal["high", "medium", "low"]
    confidence: Literal["deterministic", "inferred"]
    message: str
    evidence: str
    line: int | None
    cell: int | None          # which notebook cell, if source was .ipynb
    suggested_fix: str
```

`confidence` matters: `deterministic` means an exact AST/regex match with no
ambiguity. `inferred` means an LLM judgment call — currently only
`validation_strategy_mismatch` uses this tier. The distinction is surfaced
to the user, not hidden.

## Evaluation

Real, runnable, in the repo — not a claimed number.

```bash
python -m eval.eval
```

Scored against 12 synthetic fixtures (6 with a deliberately injected issue,
6 clean/true-negative counterparts) covering all deterministic checks. As of
this build: **precision 1.0, recall 1.0, F1 1.0** — full breakdown written to
`eval/results.md` on every run.

**Read that number honestly, not as a headline.** These are the same
fixtures the detectors were built and tuned against, not held-out data. A
perfect score here proves the code runs and the logic is internally
consistent — it does not prove the checks generalize to real-world
notebooks with messier code. That's a limitation, not a hidden one.

The LLM tool (`validation_strategy_reviewer`) is **deliberately not included**
in this precision/recall number. A P/R score computed against a handful of
hand-labeled LLM judgments would be exactly the kind of unfalsifiable-sounding
metric this project's non-goals rule out for the composite score. Instead,
it's evaluated qualitatively and reproducibly via:

```bash
python -m eval.validation_strategy_demo
```

This runs three real cases against the live model — a true positive on a
timestamp signal, a **true negative** (timestamp present, but the code
already splits correctly — proving the tool discriminates rather than
reflexively flagging any date column), and a true positive on a group/ID
signal. All three currently match expectations; the model's actual reasoning
is printed, not just a pass/fail.

## Demo: before / after

```bash
python -m eval.demo_before_after
```

Runs the full agent on `eval/synthetic_notebooks/demo_before.ipynb` (4
deliberately injected issues: pre-split scaler fit, pre-split fillna leak,
2x missing `random_state`, plus a timestamp column with no time-aware split)
and `demo_after.ipynb` (same task, all fixed). Current result: **5 issues →
0 issues.**

## Why LangChain, and not just a script?

The honest answer, not the impressive-sounding one: **the conditional branch
is implemented as a plain Python `if` statement** (see `agent.py`), not as
an LLM autonomously deciding what to do. Dressing up an if-statement as
"agentic reasoning" would be exactly the kind of overclaim this project
tries to avoid — `schema_inspector`'s signals are objective (a timestamp
column either exists or it doesn't), so there's no ambiguity for an LLM to
resolve there.

What LangChain actually buys this project:
- **Structured output** (`.with_structured_output()`) turns the LLM's
  judgment into a typed, parseable `ValidationJudgment` object instead of
  prose that needs regex-scraping.
- **Conditional invocation** means the one non-deterministic, non-free tool
  in the system only runs when there's a genuine judgment call to make —
  roughly a third of the synthetic fixtures trigger it. Without that
  branch, this is a linter with an LLM caption generator bolted on, and
  that distinction is the actual interview answer.

## Why Groq instead of OpenAI?

A documented cost/availability tradeoff, not an afterthought: Groq's free
tier (`llama-3.3-70b-versatile`) removes a hard blocker during development —
no billing setup required. The tradeoff is an open-weight model instead of
a frontier one; for this project's narrow judgment (does the split account
for this specific signal, yes or no, with reasoning) that's held up
correctly across every case in `eval/validation_strategy_demo.py` so far.
Swapping to `ChatOpenAI` is a one-line change in
`validation_strategy_reviewer.build_default_llm()`, since both expose the
same LangChain chat-model interface.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
cp .env.example .env          # then edit .env with your real GROQ_API_KEY
python -m pytest tests/ -v
```

Get a free Groq API key at https://console.groq.com — no card required.

## Interview defense — quick reference

- **"Does this catch all leakage?"** No, by design — three specific AST
  patterns, stated as a non-goal, not a hidden gap.
- **"What's the score based on?"** No composite score. Each issue stands on
  its own evidence and confidence tier — see Non-goals.
- **"How did you evaluate it?"** 12 synthetic fixtures, real precision/recall
  from a script that actually runs, explicitly including true negatives.
  The LLM tool is evaluated separately and qualitatively, on purpose.
- **"Why LangChain and not a script?"** Conditional tool invocation — see
  above. The branch itself is deterministic Python; the LLM's job is scoped
  to the one genuine judgment call in the pipeline.
- **"Why Groq, not GPT-4?"** Documented cost tradeoff, one-line swap back.

## Status

Complete and deployed.

- Backend: https://pipelineguardian-api.onrender.com (Render free tier — cold
  starts after 15min idle, ~30-50s on first request)
- Frontend: https://pipelineguardian-eight.vercel.app
- Try it: upload any file from `eval/synthetic_notebooks/` — `01_scaler_leak.py`
  through `12_group_leak_random_split.py` for individual checks, or
  `demo_before.ipynb` / `demo_after.ipynb` for the full before/after comparison.