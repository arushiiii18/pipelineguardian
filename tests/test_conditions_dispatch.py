"""
test_conditions_dispatch.py — unit tests per §9 coverage.

Verifies:
- Each AuditCondition routes to the correct implementation
- Every returned Issue.source matches the condition
- rule_only never produces VALIDATION_STRATEGY findings
"""

import os
import ast
import pytest
from unittest.mock import MagicMock

import pandas as pd

from pipelineguardian.conditions import AuditCondition, run_audit
from pipelineguardian.tools.llm_only_auditor import LLMAuditResult, LLMFinding
from pipelineguardian.models import Category, IssueSource

HERE = os.path.dirname(__file__)
FIXTURES_DIR = os.path.join(HERE, "..", "eval", "synthetic_notebooks")


def _make_mock_llm(findings=None):
    mock = MagicMock()
    mock.invoke.return_value = LLMAuditResult(findings=findings or [])
    return mock


def _scaler_leak_path():
    return os.path.join(FIXTURES_DIR, "01_scaler_leak.py")


def _clean_path():
    return os.path.join(FIXTURES_DIR, "10_all_clean_baseline.py")


# ---------------------------------------------------------------------------
# rule_only
# ---------------------------------------------------------------------------

def test_rule_only_routes_and_issues_tagged_rule():
    mock_llm = _make_mock_llm()
    report = run_audit(_scaler_leak_path(), AuditCondition.RULE_ONLY, llm=mock_llm)
    assert report is not None
    prep = [i for i in report.issues if i.category == Category.PREPROCESSING.value]
    assert prep, "rule_only should detect preprocessing leak"
    for issue in report.issues:
        assert issue.source == IssueSource.RULE.value, (
            f"rule_only: issue {issue.check_name} has source={issue.source!r}"
        )


def test_rule_only_never_produces_validation_strategy():
    """Documented limitation: rule_only has no VALIDATION_STRATEGY detector."""
    mock_llm = _make_mock_llm()
    df = pd.DataFrame({
        "event_date": pd.date_range("2024-01-01", periods=20),
        "feature": range(20),
        "target": [0, 1] * 10,
    })
    report = run_audit(
        _clean_path(), AuditCondition.RULE_ONLY,
        df=df, target_col="target", llm=mock_llm,
    )
    vs_issues = [i for i in report.issues if i.category == Category.VALIDATION_STRATEGY.value]
    assert vs_issues == [], "rule_only must never produce VALIDATION_STRATEGY findings"
    assert any("validation_strategy_reviewer" in s for s in report.tools_skipped)


# ---------------------------------------------------------------------------
# llm_only
# ---------------------------------------------------------------------------

def test_llm_only_routes_and_issues_tagged_llm_only():
    mock_llm = _make_mock_llm(findings=[
        LLMFinding(
            category=Category.PREPROCESSING,
            severity="high",
            message="Scaler fit before split.",
            evidence="line 11",
            suggested_fix="Move fit() after split.",
        )
    ])
    report = run_audit(_scaler_leak_path(), AuditCondition.LLM_ONLY, llm=mock_llm)
    assert report is not None
    assert "llm_only_auditor" in report.tools_run
    for issue in report.issues:
        assert issue.source == IssueSource.LLM_ONLY.value


def test_llm_only_clean_file_no_issues():
    mock_llm = _make_mock_llm(findings=[])
    report = run_audit(_clean_path(), AuditCondition.LLM_ONLY, llm=mock_llm)
    assert report.issues == []


# ---------------------------------------------------------------------------
# llm_with_context
# ---------------------------------------------------------------------------

def test_llm_with_context_routes_and_issues_tagged_llm_context():
    mock_llm = _make_mock_llm(findings=[
        LLMFinding(
            category=Category.OVERLAP,
            severity="medium",
            message="Group-unaware split detected.",
            evidence="train_test_split at line 10",
            suggested_fix="Use GroupKFold.",
        )
    ])
    report = run_audit(_scaler_leak_path(), AuditCondition.LLM_WITH_CONTEXT, llm=mock_llm)
    assert "llm_context_auditor" in report.tools_run
    for issue in report.issues:
        assert issue.source == IssueSource.LLM_CONTEXT.value


# ---------------------------------------------------------------------------
# hybrid
# ---------------------------------------------------------------------------

def test_hybrid_routes_and_issues_tagged_hybrid():
    mock_llm = _make_mock_llm(findings=[])
    report = run_audit(_scaler_leak_path(), AuditCondition.HYBRID, llm=mock_llm)
    assert "leakage_detector" in report.tools_run
    for issue in report.issues:
        assert issue.source == IssueSource.HYBRID.value


# ---------------------------------------------------------------------------
# Invalid condition
# ---------------------------------------------------------------------------

def test_invalid_condition_raises():
    with pytest.raises((ValueError, KeyError)):
        run_audit(_clean_path(), "not_a_valid_condition")
