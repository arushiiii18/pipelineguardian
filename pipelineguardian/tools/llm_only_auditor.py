"""
llm_only_auditor.py — LLM-only condition (§3.2).

Sees: raw flattened source + raw schema (columns, dtypes, sample_rows).
Never sees: NotebookContext structural facts, rule-engine output, Issue objects.

The prompt wording is LOCKED per spec §3.2. Do not modify it without
bumping PROMPT_VERSION in llm_config.py and re-tagging experiment-frozen.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from pydantic import BaseModel

from pipelineguardian.models import (
    Issue, Category, Severity, Confidence, IssueSource, AuditReport,
)
from pipelineguardian.tools.notebook_parser import ParsedSource
import pandas as pd


# ---------------------------------------------------------------------------
# Output schema shared with llm_context_auditor
# ---------------------------------------------------------------------------

class LLMFinding(BaseModel):
    category: Category
    severity: Severity
    message: str
    evidence: str
    line: Optional[int] = None
    suggested_fix: str


class LLMAuditResult(BaseModel):
    findings: list[LLMFinding]


# ---------------------------------------------------------------------------
# LLM protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class StructuredLLM(Protocol):
    def invoke(self, prompt: str) -> LLMAuditResult: ...


# ---------------------------------------------------------------------------
# Prompt builder — LOCKED wording per §3.2
# ---------------------------------------------------------------------------

def _build_prompt(
    pset: ParsedSource,
    df: Optional[pd.DataFrame] = None,
) -> str:
    if df is not None:
        columns_str = str(list(df.columns))
        dtypes_str = str({col: str(df[col].dtype) for col in df.columns})
        sample_rows_str = str(df.head(3).to_dict(orient="records"))
    else:
        columns_str = "Not provided"
        dtypes_str = "Not provided"
        sample_rows_str = "Not provided"

    return f"""You are auditing a machine learning notebook for methodological problems that would \
silently corrupt results or make them irreproducible.

You are looking specifically for issues in these categories:
1. Preprocessing leakage — a scaler, encoder, or imputer fit (or a statistic like \
mean/median) computed on data before/across the train/test split, then applied to \
test data.
2. Overlap leakage — rows, groups, or entities present in both the training and \
evaluation/test data.
3. Multi-test leakage — the same held-out data evaluated or selected among multiple \
times to pick a best-performing model, without a genuinely independent final test set.
4. Reproducibility issues — missing random seeds for train/test splitting, model \
initialization, or third-party libraries (numpy, torch, random), or unpinned \
dependencies.

Dataset schema (if provided):
Columns: {columns_str}
Dtypes: {dtypes_str}
Sample rows: {sample_rows_str}

Source code:
```python
{pset.source}
```

For each issue found, report: category (one of preprocessing / overlap / multi_test / \
reproducibility), severity (high / medium / low), a one-line message, the evidence (a \
code excerpt or line reference), and a concrete suggested fix. Only report issues you \
can point to concrete evidence for in the code above. If you find nothing in a \
category, do not report anything for that category."""


# ---------------------------------------------------------------------------
# LLM client builder
# ---------------------------------------------------------------------------

def build_default_llm() -> StructuredLLM:
    """Build the real LangChain-backed LLM. Only instantiated at call time —
    never at module import — so tests can import this module without GROQ_API_KEY."""
    from dotenv import load_dotenv
    load_dotenv()
    from langchain_groq import ChatGroq
    from pipelineguardian.llm_config import LLM_MODEL, LLM_TEMPERATURE
    llm = ChatGroq(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    return llm.with_structured_output(LLMAuditResult)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    pset: ParsedSource,
    df: Optional[pd.DataFrame] = None,
    llm: Optional[StructuredLLM] = None,
) -> list[Issue]:
    """Run the llm_only auditor. Returns Issues tagged source=LLM_ONLY."""
    if llm is None:
        llm = build_default_llm()

    prompt = _build_prompt(pset, df)
    result: LLMAuditResult = llm.invoke(prompt)

    issues = []
    for finding in result.findings:
        issues.append(Issue(
            check_name=f"llm_only_{finding.category.value if hasattr(finding.category, 'value') else finding.category}",
            category=finding.category,
            source=IssueSource.LLM_ONLY,
            severity=finding.severity,
            confidence=Confidence.INFERRED,
            message=finding.message,
            evidence=finding.evidence,
            line=finding.line,
            cell=None,
            suggested_fix=finding.suggested_fix,
        ))
    return issues
