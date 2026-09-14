# PipelineGuardian

Most ML bugs aren't syntax errors.

They're things like fitting a scaler before a train-test split, leaking grouped entities across folds, repeatedly evaluating on the same test set, or running experiments without controlling randomness.

Your code can run perfectly and still produce a completely unreliable experiment.

Traditional linters are good at finding code-level problems. They aren't built to understand whether an ML workflow itself makes sense.

PipelineGuardian is an ML experimentation auditor for `.ipynb` notebooks and `.py` scripts. It combines deterministic static analysis with LLM-based reasoning to catch problems that can make ML experiments misleading, unreproducible, or invalid.

The idea isn't to build another "AI code reviewer."

It's to figure out where rules are enough, where contextual reasoning actually helps, and whether combining both is better than either approach alone.

## What it checks

### Preprocessing Leakage

Deterministic AST-based analysis detects patterns including:

- `StandardScaler`, encoders, or transformers fitted before `train_test_split`
- Dataset-wide statistics computed before splitting and reused later
- Target columns accidentally included in explicit feature lists

### Overlap Leakage

Checks for evaluation setups where information from the same entity or observation can cross the intended train/test boundary.

Examples include:

- Splitting data without accounting for grouped entities
- Evaluation setups where the split happens after information has already been fitted or established across the full dataset

### Multi-Test Leakage

Detects repeated use of the same held-out test set across multiple model candidates.

For example:

```python
model_a.score(X_test, y_test)
model_b.score(X_test, y_test)

The problem isn't that both models were evaluated.

The problem is using the same held-out test set to choose between them, which means it isn't really held out anymore.

### Reproducibility

Deterministic checks for common sources of non-reproducible experiments:

- Missing `random_state`
- Unseeded NumPy randomness
- Unseeded PyTorch randomness
- Unseeded Python `random`
- Missing or unpinned dependencies

### Schema Inspection

The auditor can extract dataset-level signals that may matter when judging an ML workflow:

- Timestamp columns
- Group or entity identifiers
- Target imbalance
- Other structural properties of the dataset

The schema inspector doesn't decide that something is wrong by itself.

It provides context for the architectures that use it.

### Validation Strategy

Some ML workflow decisions can't be determined reliably from syntax alone.

For example:

```text
Timestamp column + random split
```

could be a problem in a time-dependent prediction task.

But a timestamp existing in the dataset doesn't automatically make a random split wrong. It depends on what the timestamp represents and how the experiment is supposed to work.

That's where contextual reasoning becomes useful.

## Four Architectures

PipelineGuardian is also a research system for comparing four different auditing approaches:

```text
                    Notebook / Script
                           │
                      AST / Parser
                           │
             ┌─────────────┴─────────────┐
             │                           │
      Deterministic Rules          Structured Context
             │                           │
             └─────────────┬─────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
      rule_only        llm_only      llm_with_context
          │                │                │
          └────────────────┴────────────────┘
                           │
                         hybrid
```

### `rule_only`

Uses deterministic static analysis.

No LLM reasoning.

### `llm_only`

Gives the LLM the source code and available dataset schema and asks it to identify relevant issues.

### `llm_with_context`

Adds a structured `NotebookContext` containing signals extracted from the notebook.

The purpose is to test whether giving the model explicit, structured context improves its ability to reason about the workflow.

### `hybrid`

Combines deterministic rules with selective LLM escalation.

The routing itself is deterministic rather than letting the LLM decide when to call another LLM.

The four architectures are kept separate so their behavior can actually be compared rather than hiding everything behind one pipeline.

## Why use an LLM?

Most of PipelineGuardian deliberately doesn't.

If a problem has a clear syntactic signature, a deterministic rule is easier to test, reproduce, and explain.

The LLM is useful for cases where the same code pattern can be valid or invalid depending on the surrounding context.

The research question is therefore less:

> "Can an LLM find ML bugs?"

and more:

> **"What does an LLM add beyond deterministic analysis, and does structured context make that reasoning better?"**

The model currently used for the LLM conditions is `llama-3.3-70b-versatile` through Groq and LangChain.

## Output

Every finding is returned as a typed `Issue` containing:

```text
check_name
category
source
severity
confidence
message
evidence
line / cell
suggested_fix
```

This makes findings traceable to both the type of problem and the architecture that produced it.

Confidence is intentionally simple:

```text
deterministic → produced by a deterministic rule
inferred     → produced through contextual reasoning
```

There is no attempt to manufacture a single "ML quality score" from unrelated findings.

## Evaluation

PipelineGuardian uses two complementary evaluation setups.

### Synthetic Corpus

Synthetic notebooks provide controlled positive and negative examples for the supported failure modes.

The current corpus contains **20 fixtures** covering:

- preprocessing leakage
- clean preprocessing
- reproducibility failures
- correctly seeded workflows
- overlap scenarios
- group-aware and group-unaware splitting
- repeated test evaluation
- legitimate CV followed by a single final test evaluation

These fixtures are primarily used to verify that the implemented detectors behave as intended.

Run the deterministic evaluation with:

```bash
python -m eval.eval
```

Current result:

```text
Precision: 1.0
Recall:    1.0
F1:        1.0
TP=9 FP=0 FN=0
```

These numbers are for the controlled synthetic fixtures. They are not being presented as evidence that PipelineGuardian generalizes perfectly to real-world notebooks.

### Yang Corpus

The research evaluation also includes the **100-notebook Yang corpus** and its available ground-truth labels.

This provides an external benchmark for the categories covered by its annotations and allows the four architectures to be compared on the same underlying notebooks.

The evaluation harness supports:

- Per-category precision
- Recall
- F1
- Majority voting across repeated LLM runs
- Paired McNemar testing
- Experiment manifests
- Append-only LLM call logs

Categories for which the Yang corpus does not provide ground truth are evaluated separately rather than pretending that the benchmark covers them.

## Usage

### CLI

```bash
python -m pipelineguardian.cli audit notebook.ipynb

python -m pipelineguardian.cli audit script.py \
    --data data.csv \
    --target churn \
    --json
```

The CLI exits with:

```text
0 → no high-severity findings
1 → one or more high-severity findings
```

which makes it usable as a CI gate.

### API

```bash
uvicorn pipelineguardian.api:app --reload
```

### Tests

```bash
pytest tests/ -v
```

Current test suite:

```text
112 passed
```

## Stack

- Python
- Python AST
- FastAPI
- Pydantic
- Pandas
- nbformat
- Pytest
- LangChain
- Groq
- Llama 3.3 70B

## Limitations

PipelineGuardian is intentionally scoped around a defined set of ML workflow problems.

Current limitations include:

- A finite set of supported leakage patterns
- Primarily sklearn-style workflows
- Static analysis rather than code execution
- No analysis of imported helper functions outside the inspected file
- No experiment-history tracking across notebooks
- External benchmark coverage depends on available ground truth

These aren't claims of comprehensive ML auditing.

The goal is to make every supported finding explainable, evidence-backed, and testable.

## Setup

```bash
python -m venv venv
```

Activate it:

```bash
# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env`:

```bash
cp .env.example .env
```

Add your:

```text
GROQ_API_KEY
```

Then run:

```bash
pytest tests/ -v
```

## Live Demo

Frontend:

[https://pipelineguardian-eight.vercel.app/](https://pipelineguardian-eight.vercel.app/)

Backend:

[https://pipelineguardian-api.onrender.com/](https://pipelineguardian-api.onrender.com/)

The backend runs on Render's free tier and may take 30–50 seconds to wake up after inactivity.

Try the examples inside:

```text
eval/synthetic_notebooks/
```

including:

```text
01_scaler_leak.py
07_missing_random_state.py
12_group_leak_random_split.py
19_multitest_repeated_peek.py
20_multitest_cv_then_final_eval.py
```

## Future Work

- Larger and more diverse evaluation corpora
- Additional leakage patterns
- Expanded reproducibility checks
- Richer validation-strategy reasoning
- CI/CD integration
- Repository-wide auditing
- Experiment tracking integration

PipelineGuardian started as a static ML workflow auditor.

The research implementation is about taking that one step further:

**understanding where deterministic analysis is enough, where contextual reasoning is actually useful, and what happens when the two are combined.**

```
```
