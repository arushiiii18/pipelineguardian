"""
hybrid_router.py — the hybrid condition implementation (§3.4).

Routing criterion (LOCKED — do not invent a confidence-scored alternative):

  PREPROCESSING   → Rule trusted; passes straight through. No LLM.
  REPRODUCIBILITY → Rule trusted; passes straight through. No LLM.
  OVERLAP         → Rule findings kept AND additionally escalated to
                    llm_context_auditor (lower-fidelity static detection).
  MULTI_TEST      → Rule findings kept AND additionally escalated to
                    llm_context_auditor (same rationale as OVERLAP).
  VALIDATION_STR  → Existing schema-signal gate from agent.py, unchanged.

For OVERLAP and MULTI_TEST: rule finding + LLM finding are BOTH kept,
distinctly tagged by confidence (DETERMINISTIC vs INFERRED). Both carry
source=HYBRID. Do NOT deduplicate these — whether they agree or disagree
is the data RQ3 needs.
"""

from __future__ import annotations

import ast
from typing import Optional

import pandas as pd

from pipelineguardian.models import AuditReport, IssueSource, Category
from pipelineguardian.tools.notebook_parser import ParsedSource
from pipelineguardian.tools import leakage_detector, reproducibility_checker
from pipelineguardian.tools import overlap_detector, multitest_detector
from pipelineguardian.tools import schema_inspector, validation_strategy_reviewer
from pipelineguardian.tools import context_builder, llm_context_auditor


def run_hybrid(
    pset: ParsedSource,
    project_dir: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    target_col: Optional[str] = None,
    llm=None,
) -> AuditReport:
    """Run the hybrid condition.

    Trusted categories (PREPROCESSING, REPRODUCIBILITY): rule output is final,
    no LLM call.
    Escalated categories (OVERLAP, MULTI_TEST): rule output kept, LLM also
    consulted via llm_context_auditor with categories_only restriction.
    VALIDATION_STRATEGY: existing schema-signal gate, unchanged.
    """
    tree = ast.parse(pset.source)
    issues = []
    tools_run: list[str] = []
    tools_skipped: list[str] = []

    # -----------------------------------------------------------------------
    # Trusted categories — pass straight through, no LLM call.
    # -----------------------------------------------------------------------
    trusted = leakage_detector.run(pset) + reproducibility_checker.run(pset, project_dir)
    for i in trusted:
        i.source = IssueSource.HYBRID
    issues += trusted
    tools_run += ["leakage_detector", "reproducibility_checker"]

    # -----------------------------------------------------------------------
    # Escalated categories — rule finding kept, LLM additionally consulted.
    # -----------------------------------------------------------------------
    escalated = overlap_detector.check_no_split_before_fit_eval(tree, pset)
    escalated += multitest_detector.check_repeated_test_evaluation(tree, pset)

    signals = None
    if df is not None:
        signals = schema_inspector.inspect(df, target_col)
        escalated += overlap_detector.check_group_unaware_split(tree, pset, signals)

    for i in escalated:
        i.source = IssueSource.HYBRID
    issues += escalated
    tools_run += ["overlap_detector", "multitest_detector"]

    # LLM escalation for OVERLAP and MULTI_TEST only
    context = context_builder.build(pset, df, target_col)
    llm_findings = llm_context_auditor.run(
        pset, context, llm=llm,
        categories_only=[Category.OVERLAP, Category.MULTI_TEST],
    )
    for f in llm_findings:
        f.source = IssueSource.HYBRID
    issues += llm_findings
    tools_run.append("llm_context_auditor (overlap/multi_test only)")

    # -----------------------------------------------------------------------
    # VALIDATION_STRATEGY — existing schema-signal gate, unchanged.
    # -----------------------------------------------------------------------
    if signals is not None and signals.has_trigger:
        vs_issues = validation_strategy_reviewer.review(pset, signals, llm=llm)
        for i in vs_issues:
            i.source = IssueSource.HYBRID
        issues += vs_issues
        tools_run.append("validation_strategy_reviewer")
    else:
        reason = (
            "no timestamp/group/imbalance signal, or no dataframe provided"
            if signals is None
            else "no timestamp/group/imbalance signal found"
        )
        tools_skipped.append(f"validation_strategy_reviewer ({reason})")

    # Defensive catch-all: stamp any remaining un-sourced issues
    for i in issues:
        if i.source is None:
            i.source = IssueSource.HYBRID

    return AuditReport(
        source_path=pset.path,
        issues=issues,
        tools_run=tools_run,
        tools_skipped=tools_skipped,
    )
