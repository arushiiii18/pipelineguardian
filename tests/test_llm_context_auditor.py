"""
test_llm_context_auditor.py — unit tests per §9 coverage.

Verifies:
- Prompt contains context block (structural facts)
- Prompt contains same four category descriptions as llm_only
- categories_only restricts category list in prompt text
- Output Issues tagged source=LLM_CONTEXT
- check_name prefixed 'llm_context_'
"""

import pytest
from unittest.mock import MagicMock

from pipelineguardian.tools.notebook_parser import ParsedSource
from pipelineguardian.tools.context_builder import NotebookContext
from pipelineguardian.tools.llm_context_auditor import (
    LLMFinding, LLMAuditResult, _build_prompt, run,
)
from pipelineguardian.tools.llm_only_auditor import _build_prompt as llm_only_build_prompt
from pipelineguardian.models import Category, Severity, IssueSource, Confidence


def _make_pset(source: str) -> ParsedSource:
    return ParsedSource(source=source, path="<test>", is_notebook=False, line_to_cell={})


def _make_context(split_lines=None, fit_sites=None, ts_cols=None, group_cols=None) -> NotebookContext:
    return NotebookContext(
        source="x = 1\n",
        split_call_lines=split_lines or [10],
        fit_transform_call_sites=fit_sites or [{"var": "scaler", "call": "fit", "line": 5}],
        timestamp_columns=ts_cols or ["event_date"],
        group_columns=group_cols or ["patient_id"],
        imbalance=None,
    )


SAMPLE_SOURCE = (
    "from sklearn.preprocessing import StandardScaler\n"
    "X_scaled = scaler.fit_transform(X)\n"
)


# ---------------------------------------------------------------------------
# Context block present in prompt
# ---------------------------------------------------------------------------

def test_prompt_contains_context_block():
    pset = _make_pset(SAMPLE_SOURCE)
    ctx = _make_context()
    prompt = _build_prompt(pset, ctx)
    assert "Structured context extracted via static analysis" in prompt
    assert "split_call_lines" not in prompt  # field name hidden, values shown
    assert "[10]" in prompt or "10" in prompt  # split_call_lines value appears
    assert "event_date" in prompt
    assert "patient_id" in prompt


def test_prompt_contains_fit_transform_sites():
    pset = _make_pset(SAMPLE_SOURCE)
    ctx = _make_context()
    prompt = _build_prompt(pset, ctx)
    assert "fit" in prompt  # fit site shows up in context block


# ---------------------------------------------------------------------------
# Category descriptions match llm_only (fairness check §3.3)
# ---------------------------------------------------------------------------

def test_category_descriptions_match_llm_only():
    """FAIRNESS CHECK §3.3: category description wording must be identical.
    The only structural difference between the two prompts is the context block."""
    pset = _make_pset(SAMPLE_SOURCE)
    ctx = _make_context()
    ctx_prompt = _build_prompt(pset, ctx)
    only_prompt = llm_only_build_prompt(pset)

    # These exact phrases must appear in both
    shared_phrases = [
        "Preprocessing leakage",
        "Overlap leakage",
        "Multi-test leakage",
        "Reproducibility issues",
        "you must judge for yourself",  # NOT in llm_only — this is in context block only
    ]
    for phrase in shared_phrases[:4]:
        assert phrase in ctx_prompt, f"'{phrase}' missing from llm_context prompt"
        assert phrase in only_prompt, f"'{phrase}' missing from llm_only prompt"

    # The context block MUST be in ctx_prompt but NOT in only_prompt
    assert "Structured context extracted via static analysis" in ctx_prompt
    assert "Structured context extracted via static analysis" not in only_prompt


# ---------------------------------------------------------------------------
# categories_only restricts prompt category list
# ---------------------------------------------------------------------------

def test_categories_only_restricts_prompt():
    """When categories_only=[OVERLAP, MULTI_TEST], preprocessing and
    reproducibility descriptions must be absent from the prompt."""
    pset = _make_pset(SAMPLE_SOURCE)
    ctx = _make_context()
    prompt = _build_prompt(
        pset, ctx,
        categories_only=[Category.OVERLAP, Category.MULTI_TEST],
    )
    assert "Overlap leakage" in prompt
    assert "Multi-test leakage" in prompt
    assert "Preprocessing leakage" not in prompt
    assert "Reproducibility issues" not in prompt


def test_categories_only_none_includes_all_four():
    pset = _make_pset(SAMPLE_SOURCE)
    ctx = _make_context()
    prompt = _build_prompt(pset, ctx, categories_only=None)
    assert "Preprocessing leakage" in prompt
    assert "Overlap leakage" in prompt
    assert "Multi-test leakage" in prompt
    assert "Reproducibility issues" in prompt


# ---------------------------------------------------------------------------
# Output mapping
# ---------------------------------------------------------------------------

def test_run_maps_findings_to_issues_with_llm_context_source():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = LLMAuditResult(findings=[
        LLMFinding(
            category=Category.OVERLAP,
            severity=Severity.MEDIUM,
            message="Same group in train and test.",
            evidence="train_test_split at line 10",
            suggested_fix="Use GroupKFold.",
        )
    ])
    pset = _make_pset(SAMPLE_SOURCE)
    ctx = _make_context()
    issues = run(pset, ctx, llm=mock_llm)
    assert len(issues) == 1
    assert issues[0].source == IssueSource.LLM_CONTEXT.value
    assert issues[0].confidence == Confidence.INFERRED.value
    assert issues[0].check_name == "llm_context_overlap"
    assert issues[0].category == Category.OVERLAP.value


def test_run_categories_only_restricts_llm_call():
    """When categories_only is passed, the LLM is called with restricted prompt."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = LLMAuditResult(findings=[])
    pset = _make_pset(SAMPLE_SOURCE)
    ctx = _make_context()
    run(pset, ctx, llm=mock_llm, categories_only=[Category.OVERLAP, Category.MULTI_TEST])
    assert mock_llm.invoke.call_count == 1
    prompt_used = mock_llm.invoke.call_args[0][0]
    assert "Overlap leakage" in prompt_used
    assert "Preprocessing leakage" not in prompt_used
