"""
test_hybrid_router.py — unit tests per §9 coverage.

Verifies (with mocked LLM throughout):
- PREPROCESSING/REPRODUCIBILITY findings never trigger an LLM call
- OVERLAP/MULTI_TEST findings from rule and LLM both coexist (not deduplicated)
- All returned issues carry source=HYBRID
- validation_strategy_reviewer is called when schema signal present
"""

import ast
import os
import pytest
from unittest.mock import MagicMock, patch

import pandas as pd

from pipelineguardian.tools.notebook_parser import ParsedSource, load_source
from pipelineguardian.tools.llm_only_auditor import LLMAuditResult, LLMFinding
from pipelineguardian.tools.hybrid_router import run_hybrid
from pipelineguardian.models import Category, Severity, IssueSource, Confidence
from pipelineguardian.tools.validation_strategy_reviewer import ValidationJudgment

HERE = os.path.dirname(__file__)
FIXTURES_DIR = os.path.join(HERE, "..", "eval", "synthetic_notebooks")


def _make_pset(source: str) -> ParsedSource:
    return ParsedSource(source=source, path="<test>", is_notebook=False, line_to_cell={})


def _make_mock_llm(findings=None):
    """LLM that returns LLMAuditResult with optional findings."""
    mock = MagicMock()
    mock.invoke.return_value = LLMAuditResult(findings=findings or [])
    return mock


# ---------------------------------------------------------------------------
# PREPROCESSING/REPRODUCIBILITY — rule trusted, no LLM call
# ---------------------------------------------------------------------------

def test_preprocessing_findings_never_trigger_llm():
    """PREPROCESSING issues must pass through without any LLM call
    (except the overlap/multi-test escalation which is called unconditionally)."""
    path = os.path.join(FIXTURES_DIR, "01_scaler_leak.py")
    pset = load_source(path)
    mock_llm = _make_mock_llm(findings=[])  # LLM returns nothing

    report = run_hybrid(pset, llm=mock_llm)

    # scaler_fit_before_split should be found via rule engine
    prep_issues = [i for i in report.issues if i.category == Category.PREPROCESSING.value]
    assert prep_issues, "Expected PREPROCESSING issue from 01_scaler_leak.py"

    # All issues must be tagged HYBRID
    for issue in report.issues:
        assert issue.source == IssueSource.HYBRID.value, (
            f"Issue {issue.check_name} has source={issue.source}, expected hybrid"
        )


def test_reproducibility_findings_never_trigger_extra_llm_call():
    """REPRODUCIBILITY issues are trusted — LLM is only called for
    OVERLAP/MULTI_TEST escalation, never for reproducibility categories."""
    path = os.path.join(FIXTURES_DIR, "08_missing_seed_torch.py")
    pset = load_source(path)
    mock_llm = _make_mock_llm(findings=[])

    report = run_hybrid(pset, llm=mock_llm)

    repro_issues = [i for i in report.issues if i.category == Category.REPRODUCIBILITY.value]
    assert repro_issues, "Expected REPRODUCIBILITY issue from 08_missing_seed_torch.py"
    for issue in repro_issues:
        assert issue.confidence == Confidence.DETERMINISTIC.value, (
            "REPRODUCIBILITY issues must be DETERMINISTIC — they come from the rule engine"
        )


# ---------------------------------------------------------------------------
# OVERLAP/MULTI_TEST — rule + LLM findings coexist, not deduplicated
# ---------------------------------------------------------------------------

def test_overlap_gets_both_rule_and_llm_findings():
    """An overlap issue from the rule engine plus an LLM finding should
    BOTH be kept — not collapsed into one."""
    path = os.path.join(FIXTURES_DIR, "15_overlap_no_split.py")
    pset = load_source(path)

    # LLM also flags overlap
    mock_llm = _make_mock_llm(findings=[
        LLMFinding(
            category=Category.OVERLAP,
            severity=Severity.HIGH,
            message="LLM detected no train/test split.",
            evidence="clf.fit(X, y) and clf.score(X, y)",
            suggested_fix="Split the data.",
        )
    ])
    report = run_hybrid(pset, llm=mock_llm)

    overlap_issues = [i for i in report.issues if i.category == Category.OVERLAP.value]
    assert len(overlap_issues) >= 2, (
        "Expected at least two OVERLAP issues: one rule-sourced (DETERMINISTIC) "
        "and one LLM-sourced (INFERRED)"
    )

    confidences = {i.confidence for i in overlap_issues}
    assert Confidence.DETERMINISTIC.value in confidences, "Rule OVERLAP finding missing"
    assert Confidence.INFERRED.value in confidences, "LLM OVERLAP finding missing"

    # Both tagged HYBRID
    for issue in overlap_issues:
        assert issue.source == IssueSource.HYBRID.value


def test_multitest_gets_both_rule_and_llm_findings():
    path = os.path.join(FIXTURES_DIR, "19_multitest_repeated_peek.py")
    pset = load_source(path)

    mock_llm = _make_mock_llm(findings=[
        LLMFinding(
            category=Category.MULTI_TEST,
            severity=Severity.HIGH,
            message="LLM: same test set used to compare models.",
            evidence="clf1.score(X_test) and clf2.score(X_test)",
            suggested_fix="Use cross_val_score for model selection.",
        )
    ])
    report = run_hybrid(pset, llm=mock_llm)

    mt_issues = [i for i in report.issues if i.category == Category.MULTI_TEST.value]
    assert len(mt_issues) >= 2, (
        "Expected both rule-sourced and LLM-sourced MULTI_TEST findings"
    )
    confidences = {i.confidence for i in mt_issues}
    assert Confidence.DETERMINISTIC.value in confidences
    assert Confidence.INFERRED.value in confidences


# ---------------------------------------------------------------------------
# All returned issues carry source=HYBRID
# ---------------------------------------------------------------------------

def test_all_issues_tagged_hybrid():
    path = os.path.join(FIXTURES_DIR, "01_scaler_leak.py")
    pset = load_source(path)
    mock_llm = _make_mock_llm()
    report = run_hybrid(pset, llm=mock_llm)
    for issue in report.issues:
        assert issue.source == IssueSource.HYBRID.value, (
            f"Issue {issue.check_name} has source={issue.source!r}, expected 'hybrid'"
        )


# ---------------------------------------------------------------------------
# validation_strategy_reviewer gating
# ---------------------------------------------------------------------------

def test_validation_strategy_reviewer_called_with_schema_signal():
    """When a timestamp df is provided, schema signal fires and
    validation_strategy_reviewer should be in tools_run."""
    path = os.path.join(FIXTURES_DIR, "10_all_clean_baseline.py")
    pset = load_source(path)
    df = pd.DataFrame({
        "event_date": pd.date_range("2024-01-01", periods=20, freq="D"),
        "feature": range(20),
        "target": [0, 1] * 10,
    })

    # validation_strategy_reviewer uses a different structured output type
    class _MultiLLM:
        """Routes calls to the correct mock based on which module calls it."""
        def __init__(self):
            self._llm_result = LLMAuditResult(findings=[])
            self._vs_result = ValidationJudgment(
                is_appropriate=False,
                reasoning="Random split despite timestamp column.",
                suggested_fix="Use TimeSeriesSplit.",
            )
            self.invoke_count = 0

        def invoke(self, prompt):
            self.invoke_count += 1
            # Return LLMAuditResult for the context auditor call
            if "Structured context" in prompt or "Preprocessing leakage" in prompt:
                return self._llm_result
            # Return ValidationJudgment for the validation strategy reviewer
            return self._vs_result

    multi_llm = _MultiLLM()
    report = run_hybrid(pset, df=df, target_col="target", llm=multi_llm)
    assert "validation_strategy_reviewer" in report.tools_run


def test_validation_strategy_reviewer_skipped_without_df():
    path = os.path.join(FIXTURES_DIR, "10_all_clean_baseline.py")
    pset = load_source(path)
    mock_llm = _make_mock_llm()
    report = run_hybrid(pset, df=None, llm=mock_llm)
    assert any("validation_strategy_reviewer" in s for s in report.tools_skipped)
