"""
These tests prove the CONDITIONAL BRANCH works correctly — the actual
agentic claim of the project — without ever calling a real LLM. A stub
LLM tracks whether it was invoked; that call-count assertion is the real
test. Whether the *real* OpenAI model gives good judgments is a separate,
non-deterministic concern, evaluated qualitatively in eval/results.md,
not asserted in pytest.
"""

import os
import pandas as pd
import pytest

from pipelineguardian import agent
from pipelineguardian.tools.validation_strategy_reviewer import ValidationJudgment

HERE = os.path.dirname(__file__)
FIXTURES_DIR = os.path.join(HERE, "..", "eval", "synthetic_notebooks")


class StubLLM:
    """Records whether it was called and returns a canned judgment."""
    def __init__(self, judgment: ValidationJudgment):
        self.judgment = judgment
        self.call_count = 0
        self.last_prompt = None

    def invoke(self, prompt: str) -> ValidationJudgment:
        self.call_count += 1
        self.last_prompt = prompt
        return self.judgment


def _balanced_no_signal_df():
    return pd.DataFrame({
        "id_but_unique_every_row": range(100),  # pure PK, not a group signal
        "feature": range(100),
        "target": [0] * 50 + [1] * 50,           # balanced
    })


def _timestamp_signal_df():
    return pd.DataFrame({
        "event_date": pd.date_range("2024-01-01", periods=100, freq="D"),
        "feature": range(100),
        "target": [0] * 50 + [1] * 50,
    })


def test_no_signal_skips_llm_entirely():
    stub = StubLLM(ValidationJudgment(is_appropriate=True, reasoning="n/a"))
    path = os.path.join(FIXTURES_DIR, "10_all_clean_baseline.py")

    report = agent.audit(path, df=_balanced_no_signal_df(), target_col="target", llm=stub)

    assert stub.call_count == 0
    assert "validation_strategy_reviewer" not in report.tools_run
    assert any("validation_strategy_reviewer" in s for s in report.tools_skipped)


def test_timestamp_signal_invokes_llm_and_flags_when_inappropriate():
    stub = StubLLM(ValidationJudgment(
        is_appropriate=False,
        reasoning="The code uses a random split despite a timestamp column.",
        suggested_fix="Use a chronological split instead of train_test_split.",
    ))
    path = os.path.join(FIXTURES_DIR, "10_all_clean_baseline.py")

    report = agent.audit(path, df=_timestamp_signal_df(), target_col="target", llm=stub)

    assert stub.call_count == 1
    assert "validation_strategy_reviewer" in report.tools_run
    assert any(i.check_name == "validation_strategy_mismatch" for i in report.issues)


def test_timestamp_signal_but_llm_says_appropriate_yields_no_issue():
    stub = StubLLM(ValidationJudgment(is_appropriate=True, reasoning="Already uses TimeSeriesSplit."))
    path = os.path.join(FIXTURES_DIR, "10_all_clean_baseline.py")

    report = agent.audit(path, df=_timestamp_signal_df(), target_col="target", llm=stub)

    assert stub.call_count == 1  # LLM was consulted...
    assert not any(i.check_name == "validation_strategy_mismatch" for i in report.issues)  # ...but found nothing wrong


def test_deterministic_tools_always_run_regardless_of_df():
    stub = StubLLM(ValidationJudgment(is_appropriate=True, reasoning="n/a"))
    path = os.path.join(FIXTURES_DIR, "01_scaler_leak.py")

    report = agent.audit(path, df=None, llm=stub)

    assert "leakage_detector" in report.tools_run
    assert "reproducibility_checker" in report.tools_run
    assert any(i.check_name == "scaler_fit_before_split" for i in report.issues)
    assert stub.call_count == 0
