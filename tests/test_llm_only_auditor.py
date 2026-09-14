"""
test_llm_only_auditor.py — unit tests per §9 coverage.

Uses a mocked LLM (no real API call). Asserts:
- Prompt contains no NotebookContext fields (split_call_lines etc.)
- Prompt DOES contain the four locked category descriptions
- Output Issues are mapped with source=LLM_ONLY and confidence=INFERRED
- check_name is prefixed with 'llm_only_'
"""

import pytest
from unittest.mock import MagicMock

from pipelineguardian.tools.notebook_parser import ParsedSource
from pipelineguardian.tools.llm_only_auditor import (
    LLMFinding, LLMAuditResult, _build_prompt, run,
)
from pipelineguardian.models import Category, Severity, IssueSource, Confidence


def _make_pset(source: str) -> ParsedSource:
    return ParsedSource(source=source, path="<test>", is_notebook=False, line_to_cell={})


SAMPLE_SOURCE = (
    "from sklearn.preprocessing import StandardScaler\n"
    "from sklearn.model_selection import train_test_split\n"
    "scaler = StandardScaler()\n"
    "X_scaled = scaler.fit_transform(X)\n"
    "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)\n"
)


# ---------------------------------------------------------------------------
# Prompt content assertions
# ---------------------------------------------------------------------------

def test_prompt_contains_four_category_descriptions():
    pset = _make_pset(SAMPLE_SOURCE)
    prompt = _build_prompt(pset)
    assert "Preprocessing leakage" in prompt
    assert "Overlap leakage" in prompt
    assert "Multi-test leakage" in prompt
    assert "Reproducibility issues" in prompt


def test_prompt_contains_source_code():
    pset = _make_pset(SAMPLE_SOURCE)
    prompt = _build_prompt(pset)
    assert "StandardScaler" in prompt
    assert "fit_transform" in prompt


def test_prompt_does_not_contain_context_fields():
    """llm_only prompt must NOT include NotebookContext structural fields.
    Those are reserved for llm_with_context."""
    pset = _make_pset(SAMPLE_SOURCE)
    prompt = _build_prompt(pset)
    forbidden_snippets = [
        "split_call_lines",
        "fit_transform_call_sites",
        "Structured context extracted via static analysis",
    ]
    for snippet in forbidden_snippets:
        assert snippet not in prompt, (
            f"llm_only prompt contains NotebookContext field '{snippet}' — "
            "this is reserved for llm_with_context"
        )


def test_prompt_schema_not_provided_when_no_df():
    pset = _make_pset(SAMPLE_SOURCE)
    prompt = _build_prompt(pset, df=None)
    assert "Not provided" in prompt


def test_prompt_includes_schema_when_df_given():
    import pandas as pd
    pset = _make_pset(SAMPLE_SOURCE)
    df = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})
    prompt = _build_prompt(pset, df=df)
    assert "feature" in prompt
    assert "target" in prompt


# ---------------------------------------------------------------------------
# Output mapping assertions
# ---------------------------------------------------------------------------

def test_run_maps_findings_to_issues_with_correct_source():
    """Mocked LLM returns one finding; assert it maps to Issue with source=LLM_ONLY."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = LLMAuditResult(findings=[
        LLMFinding(
            category=Category.PREPROCESSING,
            severity=Severity.HIGH,
            message="Scaler fit before split.",
            evidence="scaler.fit_transform(X) at line 4",
            line=4,
            suggested_fix="Fit scaler after split.",
        )
    ])
    pset = _make_pset(SAMPLE_SOURCE)
    issues = run(pset, llm=mock_llm)
    assert len(issues) == 1
    issue = issues[0]
    assert issue.source == IssueSource.LLM_ONLY.value
    assert issue.confidence == Confidence.INFERRED.value
    assert issue.check_name.startswith("llm_only_")
    assert issue.category == Category.PREPROCESSING.value


def test_run_returns_empty_list_when_no_findings():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = LLMAuditResult(findings=[])
    pset = _make_pset("x = 1\n")
    issues = run(pset, llm=mock_llm)
    assert issues == []


def test_run_uses_mocked_llm_not_real_api():
    """Verify the llm parameter is actually used (no real API call made)."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = LLMAuditResult(findings=[])
    pset = _make_pset("x = 1\n")
    run(pset, llm=mock_llm)
    assert mock_llm.invoke.call_count == 1


def test_check_name_prefixed_with_llm_only():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = LLMAuditResult(findings=[
        LLMFinding(
            category=Category.OVERLAP,
            severity=Severity.MEDIUM,
            message="Same group in train and test.",
            evidence="train_test_split without groups=",
            suggested_fix="Use GroupKFold.",
        )
    ])
    pset = _make_pset(SAMPLE_SOURCE)
    issues = run(pset, llm=mock_llm)
    assert issues[0].check_name == "llm_only_overlap"
