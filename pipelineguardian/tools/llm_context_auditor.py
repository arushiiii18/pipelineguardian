"""
llm_context_auditor.py — LLM-with-context condition (§3.3).

Sees: raw source + raw schema + NotebookContext structural facts.
NEVER sees: rule-engine verdicts, Issue objects from any rule detector.

FAIRNESS CHECK (§3.3): the category description wording in this prompt is
IDENTICAL to llm_only_auditor.py. The only difference is the inserted
context block before "Source code:". If any other wording differs, that
is a bug — fix it before running any evaluation.

The context block wording is LOCKED per spec §3.3. Do not modify without
bumping PROMPT_VERSION in llm_config.py and re-tagging experiment-frozen.

The `categories_only` parameter (used by hybrid_router for OVERLAP/MULTI_TEST
escalation) restricts the numbered category list to only those categories,
using the exact same per-category wording — it does not change any other
part of the prompt.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from pydantic import BaseModel

from pipelineguardian.models import (
    Issue, Category, Severity, Confidence, IssueSource,
)
from pipelineguardian.tools.notebook_parser import ParsedSource
from pipelineguardian.tools.context_builder import NotebookContext
import pandas as pd


# Re-export shared output schema from llm_only_auditor for DRY consistency
from pipelineguardian.tools.llm_only_auditor import LLMFinding, LLMAuditResult


@runtime_checkable
class StructuredLLM(Protocol):
    def invoke(self, prompt: str) -> LLMAuditResult: ...


# ---------------------------------------------------------------------------
# Category descriptions — LOCKED, identical wording to llm_only_auditor
# (enforced by the §3.3 fairness check; any deviation is a bug)
# ---------------------------------------------------------------------------

_ALL_CATEGORY_DESCRIPTIONS = {
    Category.PREPROCESSING: (
        "Preprocessing leakage — a scaler, encoder, or imputer fit (or a statistic like "
        "mean/median) computed on data before/across the train/test split, then applied to "
        "test data."
    ),
    Category.OVERLAP: (
        "Overlap leakage — rows, groups, or entities present in both the training and "
        "evaluation/test data."
    ),
    Category.MULTI_TEST: (
        "Multi-test leakage — the same held-out data evaluated or selected among multiple "
        "times to pick a best-performing model, without a genuinely independent final test set."
    ),
    Category.REPRODUCIBILITY: (
        "Reproducibility issues — missing random seeds for train/test splitting, model "
        "initialization, or third-party libraries (numpy, torch, random), or unpinned "
        "dependencies."
    ),
}

# Ordered list for stable prompt numbering
_CATEGORY_ORDER = [
    Category.PREPROCESSING,
    Category.OVERLAP,
    Category.MULTI_TEST,
    Category.REPRODUCIBILITY,
]


def _build_category_list(categories_only: Optional[list[Category]] = None) -> str:
    """Build the numbered category list. When categories_only is given, only
    those categories appear (same wording, just the others omitted)."""
    cats = categories_only if categories_only is not None else _CATEGORY_ORDER
    # Filter to only categories in _ALL_CATEGORY_DESCRIPTIONS
    active = [c for c in _CATEGORY_ORDER if c in cats]
    lines = []
    for i, cat in enumerate(active, start=1):
        lines.append(f"{i}. {_ALL_CATEGORY_DESCRIPTIONS[cat]}")
    return "\n".join(lines)


def _build_context_block(context: NotebookContext) -> str:
    """LOCKED context block wording per spec §3.3."""
    return (
        "Structured context extracted via static analysis (facts only — no conclusions have\n"
        "been drawn from these facts; you must judge for yourself whether any of them indicate\n"
        "an issue):\n"
        f"- train_test_split()/KFold()/GroupKFold()/cross_val_score() calls at lines: {context.split_call_lines}\n"
        f"- fit()/fit_transform() calls on scaler/encoder/imputer-like objects: {context.fit_transform_call_sites}\n"
        f"- Timestamp-like column(s): {context.timestamp_columns}\n"
        f"- Group/ID-like column(s) with repeated values: {context.group_columns}\n"
        f"- Class imbalance: {context.imbalance}"
    )


def _build_prompt(
    pset: ParsedSource,
    context: NotebookContext,
    categories_only: Optional[list[Category]] = None,
) -> str:
    """Build the llm_with_context prompt.

    Category descriptions are identical to llm_only_auditor._build_prompt.
    The ONLY addition is the context block inserted before 'Source code:'.
    """
    if context.columns is not None:
        columns_str = str(context.columns)
        dtypes_str = str(context.dtypes)
        sample_rows_str = str(context.sample_rows)
    else:
        columns_str = "Not provided"
        dtypes_str = "Not provided"
        sample_rows_str = "Not provided"

    category_list = _build_category_list(categories_only)
    context_block = _build_context_block(context)

    return f"""You are auditing a machine learning notebook for methodological problems that would \
silently corrupt results or make them irreproducible.

You are looking specifically for issues in these categories:
{category_list}

Dataset schema (if provided):
Columns: {columns_str}
Dtypes: {dtypes_str}
Sample rows: {sample_rows_str}

{context_block}

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
    from dotenv import load_dotenv
    load_dotenv()
    from langchain_groq import ChatGroq
    from pipelineguardian.llm_config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
    llm = ChatGroq(model=LLM_MODEL, temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS)
    return llm.with_structured_output(LLMAuditResult)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    pset: ParsedSource,
    context: NotebookContext,
    llm: Optional[StructuredLLM] = None,
    categories_only: Optional[list[Category]] = None,
) -> list[Issue]:
    """Run the llm_with_context auditor.

    categories_only — when provided (e.g. by hybrid_router for OVERLAP/MULTI_TEST),
    restricts the prompt's category list to those categories only. Same wording
    per category; others are simply omitted.

    Returns Issues tagged source=LLM_CONTEXT.
    """
    if llm is None:
        llm = build_default_llm()

    prompt = _build_prompt(pset, context, categories_only=categories_only)
    result: LLMAuditResult = llm.invoke(prompt)

    issues = []
    for finding in result.findings:
        cat_val = finding.category.value if hasattr(finding.category, 'value') else finding.category
        issues.append(Issue(
            check_name=f"llm_context_{cat_val}",
            category=finding.category,
            source=IssueSource.LLM_CONTEXT,
            severity=finding.severity,
            confidence=Confidence.INFERRED,
            message=finding.message,
            evidence=finding.evidence,
            line=finding.line,
            cell=None,
            suggested_fix=finding.suggested_fix,
        ))
    return issues
