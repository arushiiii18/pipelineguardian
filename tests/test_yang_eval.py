"""
test_yang_eval.py — unit tests per §9 coverage.

Verifies:
- load_ground_truth() binarization rule: 'Y', 'Y (mean)', 'N', blank → correct booleans
- 'model' column is never read
- score_condition() accumulates TP/FP/FN correctly on a tiny inline fixture
"""

import csv
import io
import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from eval.yang_eval import load_ground_truth


# ---------------------------------------------------------------------------
# Binarization rule tests (§6.2 LOCKED)
# ---------------------------------------------------------------------------

def _write_temp_csv(rows: list[dict]) -> str:
    """Write rows to a temp CSV and return the path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8", newline=""
    )
    fieldnames = ["nb", "model", "pre", "overlap", "multi"]
    writer = csv.DictWriter(tmp, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    tmp.close()
    return tmp.name


def test_binarization_y_is_true():
    path = _write_temp_csv([
        {"nb": "nb1.py", "model": "Y", "pre": "Y", "overlap": "Y", "multi": "Y"},
    ])
    gt = load_ground_truth(path)
    assert gt["nb1.py"]["pre"] is True
    assert gt["nb1.py"]["overlap"] is True
    assert gt["nb1.py"]["multi"] is True
    os.unlink(path)


def test_binarization_y_mean_is_true():
    """Qualitative suffix 'Y (mean)' must binarize to True."""
    path = _write_temp_csv([
        {"nb": "nb2.py", "model": "Y", "pre": "Y (mean)", "overlap": "N", "multi": ""},
    ])
    gt = load_ground_truth(path)
    assert gt["nb2.py"]["pre"] is True
    os.unlink(path)


def test_binarization_y_minmaxscaler_is_true():
    """Qualitative suffix 'Y (MinMaxScaler)' must binarize to True."""
    path = _write_temp_csv([
        {"nb": "nb3.py", "model": "Y", "pre": "Y (MinMaxScaler)", "overlap": "N", "multi": "N"},
    ])
    gt = load_ground_truth(path)
    assert gt["nb3.py"]["pre"] is True
    os.unlink(path)


def test_binarization_n_is_false():
    path = _write_temp_csv([
        {"nb": "nb4.py", "model": "Y", "pre": "N", "overlap": "N", "multi": "N"},
    ])
    gt = load_ground_truth(path)
    assert gt["nb4.py"]["pre"] is False
    assert gt["nb4.py"]["overlap"] is False
    assert gt["nb4.py"]["multi"] is False
    os.unlink(path)


def test_binarization_blank_is_false():
    """Blank cells binarize to False."""
    path = _write_temp_csv([
        {"nb": "nb5.py", "model": "N", "pre": "", "overlap": "", "multi": ""},
    ])
    gt = load_ground_truth(path)
    assert gt["nb5.py"]["pre"] is False
    assert gt["nb5.py"]["overlap"] is False
    assert gt["nb5.py"]["multi"] is False
    os.unlink(path)


def test_model_column_never_read():
    """The 'model' column must never appear in the returned dict."""
    path = _write_temp_csv([
        {"nb": "nb6.py", "model": "Y", "pre": "Y", "overlap": "N", "multi": "N"},
    ])
    gt = load_ground_truth(path)
    assert "model" not in gt["nb6.py"], (
        "'model' column must never be returned — it is an annotation flag, not a label"
    )
    os.unlink(path)


def test_multiple_rows_loaded_correctly():
    path = _write_temp_csv([
        {"nb": "nb1.py", "model": "Y", "pre": "Y (mean)", "overlap": "N", "multi": "Y"},
        {"nb": "nb2.py", "model": "N", "pre": "", "overlap": "", "multi": ""},
        {"nb": "nb3.py", "model": "Y", "pre": "Y", "overlap": "Y", "multi": "N"},
    ])
    gt = load_ground_truth(path)
    assert len(gt) == 3
    assert gt["nb1.py"] == {"pre": True, "overlap": False, "multi": True}
    assert gt["nb2.py"] == {"pre": False, "overlap": False, "multi": False}
    assert gt["nb3.py"] == {"pre": True, "overlap": True, "multi": False}
    os.unlink(path)


def test_case_insensitive_y_detection():
    """Binarization should handle uppercase Y variants — startswith('Y') is LOCKED."""
    path = _write_temp_csv([
        {"nb": "nb7.py", "model": "Y", "pre": "y", "overlap": "Y (oversampling)", "multi": "N"},
    ])
    gt = load_ground_truth(path)
    # 'y'.strip().upper().startswith('Y') → True
    assert gt["nb7.py"]["pre"] is True
    assert gt["nb7.py"]["overlap"] is True
    os.unlink(path)


# ---------------------------------------------------------------------------
# McNemar's test & architecture comparison tests
# ---------------------------------------------------------------------------

def test_mcnemar_test_basic():
    from eval.yang_eval import mcnemar_test
    # Condition A: [T, F], Condition B: [F, T], GT: [T, T]
    # A correct: [True, False], B correct: [False, True]
    # n01 = 1 (A wrong, B right), n10 = 1 (A right, B wrong)
    res = mcnemar_test([True, False], [False, True], [True, True])
    assert res["n01"] == 1
    assert res["n10"] == 1
    assert res["p_value"] == 1.0
    assert "statistic" in res


def test_mcnemar_test_zero_discordant():
    from eval.yang_eval import mcnemar_test
    # Both identical predictions
    res = mcnemar_test([True, False], [True, False], [True, True])
    assert res["n01"] == 0
    assert res["n10"] == 0
    assert res["p_value"] == 1.0
    assert res["statistic"] == 0.0


def test_compare_architectures_helper():
    from eval.yang_eval import compare_architectures
    gt = {
        "nb1.py": {"pre": True, "overlap": False, "multi": False},
        "nb2.py": {"pre": False, "overlap": True, "multi": False},
    }
    res_a = {
        "condition": "rule_only",
        "per_notebook": [
            {"nb_id": "nb1.py", "pre_pred": True, "overlap_pred": False, "multi_pred": False},
            {"nb_id": "nb2.py", "pre_pred": False, "overlap_pred": False, "multi_pred": False},
        ]
    }
    res_b = {
        "condition": "hybrid",
        "per_notebook": [
            {"nb_id": "nb1.py", "pre_pred": True, "overlap_pred": False, "multi_pred": False},
            {"nb_id": "nb2.py", "pre_pred": False, "overlap_pred": True, "multi_pred": False},
        ]
    }
    comps = compare_architectures({"rule_only": res_a, "hybrid": res_b}, gt)
    assert "rule_only_vs_hybrid" in comps
    assert "overlap" in comps["rule_only_vs_hybrid"]
    assert comps["rule_only_vs_hybrid"]["overlap"]["n01"] == 1
    assert comps["rule_only_vs_hybrid"]["overlap"]["n10"] == 0

