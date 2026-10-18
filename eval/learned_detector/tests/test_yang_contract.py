"""
Tests for Yang Evaluation Contract and dev ID exclusion.
"""

import os
import sys
import pytest

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from evaluate_yang_contract import load_dev_ids, check_teammate_readiness, run_yang_evaluation_contract


def test_dev_ids_loading():
    dev_ids = load_dev_ids()
    # 12 fixed development IDs
    assert len(dev_ids) == 12
    assert "GitHub-data/notebooks/2021-09-03/nb_2630.py" in dev_ids


def test_yang_contract_readiness_or_blocker():
    report = run_yang_evaluation_contract()
    assert "status" in report
    if report["status"] == "BLOCKED_ON_TEAMMATE_1":
        assert "missing_inputs" in report
        assert "required_contract_spec" in report
        assert len(report["required_contract_spec"]) == 6
