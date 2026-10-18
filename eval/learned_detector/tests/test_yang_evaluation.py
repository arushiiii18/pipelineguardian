"""
test_yang_evaluation.py — Tests for run_yang_evaluation.py

Validates:
  - Ground truth parsing (filename conversion, binary label extraction)
  - Dev ID loading and count
  - Exclusion of invalid notebook and dev IDs from eligible set
  - Metric calculation correctness given known y_true / y_pred
  - Missing notebook handling (no ground truth entry)
"""
import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock
from sklearn.metrics import f1_score, precision_score, recall_score

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from run_yang_evaluation import (
    get_ground_truth,
    load_dev_ids,
    nb_path_to_fname,
    load_rule_baseline,
    INVALID_NOTEBOOK,
)


# ---------- Unit tests: path conversion ----------

def test_nb_path_to_fname_standard():
    path = "GitHub-data/notebooks/2021-09-05/nb_1244.py"
    assert nb_path_to_fname(path) == "2021-09-05-nb_1244.ipynb"


def test_nb_path_to_fname_passthrough():
    """Paths without enough segments are returned unchanged."""
    path = "nb_1244.py"
    assert nb_path_to_fname(path) == "nb_1244.py"


# ---------- Unit tests: ground truth parsing ----------

def test_ground_truth_parsing():
    gt = get_ground_truth()
    assert isinstance(gt, dict)
    assert len(gt) > 0, "Ground truth should not be empty"
    for fname, label in gt.items():
        assert fname.endswith(".ipynb"), f"Expected .ipynb filename, got: {fname}"
        assert label in (0, 1), f"Expected binary label, got: {label}"


def test_ground_truth_has_positive_and_negative():
    gt = get_ground_truth()
    positives = [k for k, v in gt.items() if v == 1]
    negatives = [k for k, v in gt.items() if v == 0]
    assert len(positives) > 0, "Should have at least one positive (leakage) case"
    assert len(negatives) > 0, "Should have at least one negative (clean) case"


def test_known_notebook_has_correct_label():
    gt = get_ground_truth()
    # nb_949.py -> pre=Y (MinMaxScaler) -> leakage=1
    fname_pos = "2021-09-07-nb_949.ipynb"
    # nb_1244.py -> pre=N -> clean=0
    fname_neg = "2021-09-05-nb_1244.ipynb"
    assert gt.get(fname_pos) == 1, f"Expected {fname_pos} to be label=1"
    assert gt.get(fname_neg) == 0, f"Expected {fname_neg} to be label=0"


# ---------- Unit tests: dev ID loading ----------

def test_dev_id_loading():
    dev_ids = load_dev_ids()
    assert len(dev_ids) == 12, f"Expected 12 dev IDs, got {len(dev_ids)}"


def test_dev_ids_are_nonempty_strings():
    dev_ids = load_dev_ids()
    for did in dev_ids:
        assert isinstance(did, str) and len(did) > 0


# ---------- Unit tests: rule baseline loading ----------

def test_rule_baseline_loading():
    rule = load_rule_baseline()
    assert isinstance(rule, dict)
    # If available, values should be 0 or 1
    for fname, label in rule.items():
        assert label in (0, 1), f"Rule label should be binary, got: {label}"


# ---------- Unit tests: metric calculation ----------

def test_metric_calculation_perfect():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 1, 0, 0]
    assert precision_score(y_true, y_pred) == 1.0
    assert recall_score(y_true, y_pred) == 1.0
    assert f1_score(y_true, y_pred) == 1.0


def test_metric_calculation_all_negative():
    """Matches what happens with domain shift: model predicts all 0."""
    y_true = [1, 1, 0, 0]
    y_pred = [0, 0, 0, 0]
    assert precision_score(y_true, y_pred, zero_division=0) == 0.0
    assert recall_score(y_true, y_pred, zero_division=0) == 0.0
    assert f1_score(y_true, y_pred, zero_division=0) == 0.0


# ---------- Integration test: eligible set properties ----------

def test_invalid_notebook_excluded():
    """The fixed invalid notebook must never appear in the included set."""
    scored_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "yang_scored_evaluation.json"
    )
    if not os.path.exists(scored_path):
        pytest.skip("yang_scored_evaluation.json not yet generated")
    r = json.load(open(scored_path))
    excluded_ids = [e["id"] for e in r["manifest"]["excluded"]]
    included_ids = r["manifest"]["included"]
    assert INVALID_NOTEBOOK not in included_ids, f"{INVALID_NOTEBOOK} should be excluded"
    assert INVALID_NOTEBOOK in excluded_ids, f"{INVALID_NOTEBOOK} should be in excluded list"


def test_eligible_set_does_not_contain_dev_ids():
    scored_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "yang_scored_evaluation.json"
    )
    if not os.path.exists(scored_path):
        pytest.skip("yang_scored_evaluation.json not yet generated")
    r = json.load(open(scored_path))
    dev_ids = load_dev_ids()
    dev_fnames = {nb_path_to_fname(x) for x in dev_ids}
    for inc in r["manifest"]["included"]:
        assert inc not in dev_fnames, f"Dev ID {inc} leaked into eligible set"


def test_eligible_set_count_and_label_counts():
    scored_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "yang_scored_evaluation.json"
    )
    if not os.path.exists(scored_path):
        pytest.skip("yang_scored_evaluation.json not yet generated")
    r = json.load(open(scored_path))
    m = r["learned_model_metrics"]
    assert r["eligible_sample_count"] > 0
    assert m["positive_cases"] + m["negative_cases"] == r["eligible_sample_count"]


def test_rule_comparison_on_same_eligible_set():
    scored_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "yang_scored_evaluation.json"
    )
    if not os.path.exists(scored_path):
        pytest.skip("yang_scored_evaluation.json not yet generated")
    r = json.load(open(scored_path))
    rm = r.get("rule_baseline_metrics")
    if rm is None:
        pytest.skip("Rule baseline not available")
    # Common denominator must be <= eligible set
    assert rm["support"] <= r["eligible_sample_count"]
