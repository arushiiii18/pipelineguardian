"""
validation_strategy_reviewer.py — the one LLM-reasoning tool in the system.

Only invoked when schema_inspector's signals say it's worth it (see agent.py
for that branch — it's a plain Python if-statement, not this module's job).
What the LLM does here is the part that genuinely can't be reduced to a
deterministic rule: judging whether the split strategy already used in the
code is appropriate given the signal, not just flagging "signal present."
"""

from __future__ import annotations
from typing import Protocol, Optional
from pydantic import BaseModel, Field
from pipelineguardian.models import Issue, Severity, Confidence, Category, IssueSource
from pipelineguardian.tools.schema_inspector import SchemaSignals


class ValidationJudgment(BaseModel):
    is_appropriate: bool = Field(
        ..., description="True if the split strategy used in the code is appropriate given the signals"
    )
    reasoning: str = Field(
        ..., description="1-2 sentences of concrete reasoning tied to the specific signal and code"
    )
    suggested_fix: str = Field(
        default="", description="Concrete fix if not appropriate; empty string if appropriate"
    )


class StructuredLLM(Protocol):
    """Anything shaped like LangChain's `.invoke(prompt) -> ValidationJudgment`."""
    def invoke(self, prompt: str) -> ValidationJudgment: ...


def _build_prompt(pset, signals: SchemaSignals) -> str:
    signal_lines = []
    if signals.timestamp_columns:
        signal_lines.append(f"Timestamp-like column(s): {signals.timestamp_columns}")
    if signals.group_columns:
        signal_lines.append(f"Group/ID column(s) with repeated values across rows: {signals.group_columns}")
    if signals.imbalance:
        signal_lines.append(
            f"Class imbalance in target '{signals.imbalance['column']}': "
            f"minority class is {signals.imbalance['minority_ratio']*100:.1f}% of rows"
        )
    signals_block = "\n".join(f"- {s}" for s in signal_lines)

    return f"""You are reviewing a machine learning pipeline's train/test split strategy.

Schema signals detected in the dataset:
{signals_block}

Source code:
```python
{pset.source}
```

Judge whether the split strategy used in this code is appropriate given the
signals above. Rules of thumb, not absolutes:
- A timestamp column usually needs a time-based split, not a random one.
- A group/ID column with repeated rows usually needs a group-aware split
  (e.g. GroupKFold), so the same group doesn't leak across train and test.
- Class imbalance usually needs a stratified split.

If the code already uses stratify=, GroupKFold, TimeSeriesSplit, or an
explicit chronological slice that correctly addresses the relevant signal,
the strategy IS appropriate. Do not flag it just because a signal exists —
flag it only if the code's actual split doesn't account for that signal.

Stay scoped to the split strategy only. Do not mention random_state,
missing seeds, scaler ordering, or any other concern in your reasoning —
those are covered by separate checks and are out of scope here even if
you notice them in the code.
"""


def build_default_llm() -> StructuredLLM:
    """
    Constructs the real LangChain-backed LLM. Imported and instantiated only
    here, at call time — NOT at module import time — so importing this file
    (and everything that depends on it, including the test suite) never
    requires GROQ_API_KEY to be set.

    Uses Groq's free tier (llama-3.3-70b-versatile) by default — no billing
    required, just an API key from console.groq.com set as GROQ_API_KEY.
    This IS the "local/cheap-model swap as a cost tradeoff" decision from
    the interview prep: Groq over OpenAI here because the free tier removes
    a hard blocker during development, at the cost of using an open-weight
    model instead of a frontier one. Swapping back to ChatOpenAI later is a
    one-line change, since both expose the same LangChain chat-model interface.
    """
    from dotenv import load_dotenv
    load_dotenv()  # reads GROQ_API_KEY from a .env file in the project root, if present

    from langchain_groq import ChatGroq
    from pipelineguardian.llm_config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
    llm = ChatGroq(model=LLM_MODEL, temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS)
    return llm.with_structured_output(ValidationJudgment)


def review(pset, signals: SchemaSignals, llm: Optional[StructuredLLM] = None) -> list[Issue]:
    if llm is None:
        llm = build_default_llm()

    prompt = _build_prompt(pset, signals)
    judgment: ValidationJudgment = llm.invoke(prompt)

    if judgment.is_appropriate:
        return []

    return [Issue(
        check_name="validation_strategy_mismatch",
        category=Category.VALIDATION_STRATEGY,
        source=IssueSource.LLM_CONTEXT,
        severity=Severity.HIGH,
        confidence=Confidence.INFERRED,
        message=judgment.reasoning,
        evidence=(
            f"schema signals — timestamp: {signals.timestamp_columns}, "
            f"group: {signals.group_columns}, imbalance: {signals.imbalance}"
        ),
        suggested_fix=judgment.suggested_fix or "Review the split strategy against the detected schema signal.",
    )]
