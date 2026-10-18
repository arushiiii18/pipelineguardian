"""
test_yang_evaluation_v2.py — Test suite for run_yang_evaluation_v2.py.

Validates:
- Wilson score interval mathematics and edge cases (0 total, 0 successes, total successes)
- Trivial baseline functions (_always_pred with Wilson CI)
- Regex heuristic detector
- Abstention accounting and conservative metric adjustment
- Full evaluation integrity: coverage, manifest sum == 100 notebooks, frozen configuration
"""

import os
import sys
import pytest

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from run_yang_evaluation_v2 import (
    wilson_score_interval,
    _always_pred,
    _regex_heuristic_pred,
    _compute_metrics,
    _conservative_metrics,
    get_ground_truth,
    load_dev_ids,
    run_evaluation,
    INVALID_NOTEBOOK,
)


# ---------- Wilson score tests ----------

def test_wilson_score_zero_total():
    ci = wilson_score_interval(0, 0)
    assert ci["lower"] is None
    assert ci["upper"] is None
    assert ci["center"] is None


def test_wilson_score_bounds_within_unit_interval():
    ci = wilson_score_interval(10, 20)
    assert 0.0 <= ci["lower"] <= ci["center"] <= ci["upper"] <= 1.0


def test_wilson_score_zero_successes():
    ci = wilson_score_interval(0, 50)
    assert ci["lower"] == 0.0
    assert ci["upper"] > 0.0
    assert ci["upper"] <= 1.0


def test_wilson_score_all_successes():
    ci = wilson_score_interval(50, 50)
    assert ci["upper"] == 1.0
    assert ci["lower"] < 1.0
    assert ci["lower"] >= 0.0


def test_wilson_score_center_monotonicity():
    ci1 = wilson_score_interval(10, 100)
    ci2 = wilson_score_interval(50, 100)
    ci3 = wilson_score_interval(90, 100)
    assert ci1["center"] < ci2["center"] < ci3["center"]


# ---------- Trivial baselines tests ----------

def test_always_zero_baseline():
    y_true = [0, 0, 1, 1]
    res = _always_pred(y_true, 0)
    assert res["precision"] == 0.0
    assert res["recall"] == 0.0
    assert res["f1"] == 0.0
    assert res["accuracy"] == 0.5
    assert res["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 2, "tp": 0}
    assert "wilson_ci" in res


def test_always_one_baseline():
    y_true = [0, 0, 1, 1]
    res = _always_pred(y_true, 1)
    assert res["recall"] == 1.0
    assert res["accuracy"] == 0.5
    assert res["confusion_matrix"] == {"tn": 0, "fp": 2, "fn": 0, "tp": 2}


# ---------- Regex heuristic tests ----------

def test_regex_heuristic_clean():
    clean_code = """
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
scaler.fit(X_train)
"""
    assert _regex_heuristic_pred(clean_code) == 0


def test_regex_heuristic_leakage():
    leaky_code = """
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
scaler.fit(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)
"""
    assert _regex_heuristic_pred(leaky_code) == 1


# ---------- Conservative metrics tests ----------

def test_conservative_metrics_increases_fn():
    y_true = [0, 1, 1]
    y_pred = [0, 1, 0]  # TP=1, FP=0, FN=1, TN=1
    normal = _compute_metrics(y_true, y_pred)
    conserv = _conservative_metrics(y_true, y_pred, n_abstained_positives=2)

    assert normal["confusion_matrix"]["fn"] == 1
    assert conserv["adjusted_fn"] == 3
    assert conserv["recall"] < normal["recall"]


# ---------- Full evaluation integration tests ----------

def test_v2_manifest_and_counts_integrity():
    report = run_evaluation()
    assert report["status"] == "EVALUATED"

    manifest = report["manifest"]
    n_inc = len(manifest["included"])
    n_inv = len(manifest["excluded_invalid_json"])
    n_dev = len(manifest["excluded_dev_id"])
    n_unl = len(manifest["unlabeled_in_gt"])
    n_abs = len(manifest["abstained"])

    # Total notebooks in corpus must be exactly 100
    total_accounted = n_inc + n_inv + n_dev + n_unl + n_abs
    assert total_accounted == 100, f"Expected 100 total accounted notebooks, got {total_accounted}"

    assert n_inv == 1
    assert n_dev == 12
    assert report["eligible_sample_count"] == n_inc + n_abs
    assert report["coverage_pct"] == 100.0
    assert report["coverage_wilson_ci"]["lower"] is not None


def test_v2_configuration_is_frozen():
    report = run_evaluation()
    cfg = report["config"]
    assert cfg["adapter_version"] == "1.0.0"
    assert cfg["checkpoint_sha256"] is not None
    assert cfg["ground_truth_sha256"] is not None
    assert cfg["invalid_notebook_excluded"] == INVALID_NOTEBOOK
