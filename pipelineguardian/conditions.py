"""
conditions.py — AuditCondition enum and the single run_audit() entrypoint (§1.4).

This is the dispatcher every eval script and condition-specific test calls.
agent.py's existing audit() function is kept unchanged in behavior and
becomes the implementation backing AuditCondition.HYBRID's validation-strategy
sub-path (via hybrid_router.py).

Documented limitation: AuditCondition.RULE_ONLY can never produce a
VALIDATION_STRATEGY finding by construction — no rule detector exists for
that category. Cross-condition comparisons on VALIDATION_STRATEGY are between
LLM_ONLY, LLM_WITH_CONTEXT, and HYBRID only.
"""

from __future__ import annotations

import ast
from enum import Enum
from typing import Optional

import pandas as pd

from pipelineguardian.models import AuditReport, IssueSource, Category
from pipelineguardian.tools.notebook_parser import load_source


class AuditCondition(str, Enum):
    RULE_ONLY = "rule_only"
    LLM_ONLY = "llm_only"
    LLM_WITH_CONTEXT = "llm_with_context"
    HYBRID = "hybrid"


def run_audit(
    source_path: str,
    condition: AuditCondition,
    project_dir: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    target_col: Optional[str] = None,
    llm=None,
) -> AuditReport:
    """Single entrypoint for all four experimental conditions.

    Parameters
    ----------
    source_path : str
        Path to a .ipynb or .py file.
    condition : AuditCondition
        Which of the four conditions to run.
    project_dir : str, optional
        Project root directory for requirements.txt inspection
        (used by rule-based reproducibility checks).
    df : pd.DataFrame, optional
        Dataset for schema_inspector and overlap Check B.
    target_col : str, optional
        Target column name, required for imbalance detection.
    llm : optional
        Pre-built structured LLM for testing (avoids real API calls).
        When None, each LLM-backed component builds the real client lazily.
    """
    pset = load_source(source_path)

    if condition == AuditCondition.RULE_ONLY:
        return _run_rule_only(pset, project_dir=project_dir, df=df, target_col=target_col)

    elif condition == AuditCondition.LLM_ONLY:
        return _run_llm_only(pset, df=df, llm=llm)

    elif condition == AuditCondition.LLM_WITH_CONTEXT:
        return _run_llm_with_context(pset, df=df, target_col=target_col, llm=llm)

    elif condition == AuditCondition.HYBRID:
        from pipelineguardian.tools.hybrid_router import run_hybrid
        return run_hybrid(pset, project_dir=project_dir, df=df, target_col=target_col, llm=llm)

    else:
        raise ValueError(f"Unknown AuditCondition: {condition!r}")


# ---------------------------------------------------------------------------
# Condition implementations
# ---------------------------------------------------------------------------

def _run_rule_only(pset, project_dir=None, df=None, target_col=None) -> AuditReport:
    """rule_only: every deterministic detector, no LLM of any kind.

    DOCUMENTED LIMITATION: Can never produce a VALIDATION_STRATEGY finding.
    Cross-condition comparisons on that category use LLM_ONLY, LLM_WITH_CONTEXT,
    and HYBRID only. This is not a bug — it is a structural property of the condition.
    """
    from pipelineguardian.tools import (
        leakage_detector, reproducibility_checker,
        overlap_detector, multitest_detector, schema_inspector,
    )

    tree = ast.parse(pset.source)
    issues = []

    issues += leakage_detector.run(pset)
    issues += reproducibility_checker.run(pset, project_dir)
    issues += overlap_detector.check_no_split_before_fit_eval(tree, pset)
    issues += multitest_detector.check_repeated_test_evaluation(tree, pset)

    if df is not None:
        signals = schema_inspector.inspect(df, target_col)
        issues += overlap_detector.check_group_unaware_split(tree, pset, signals)

    for i in issues:
        i.source = IssueSource.RULE.value

    return AuditReport(
        source_path=pset.path,
        issues=issues,
        tools_run=["leakage_detector", "reproducibility_checker",
                   "overlap_detector", "multitest_detector"],
        tools_skipped=["validation_strategy_reviewer (rule_only condition — no LLM)"],
    )


def _run_llm_only(pset, df=None, llm=None) -> AuditReport:
    from pipelineguardian.tools.llm_only_auditor import run as llm_only_run
    issues = llm_only_run(pset, df=df, llm=llm)
    return AuditReport(
        source_path=pset.path,
        issues=issues,
        tools_run=["llm_only_auditor"],
        tools_skipped=[],
    )


def _run_llm_with_context(pset, df=None, target_col=None, llm=None) -> AuditReport:
    from pipelineguardian.tools.context_builder import build
    from pipelineguardian.tools.llm_context_auditor import run as context_run
    context = build(pset, df=df, target_col=target_col)
    issues = context_run(pset, context, llm=llm)
    return AuditReport(
        source_path=pset.path,
        issues=issues,
        tools_run=["context_builder", "llm_context_auditor"],
        tools_skipped=[],
    )
