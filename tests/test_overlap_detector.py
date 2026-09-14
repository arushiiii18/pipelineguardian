"""
test_overlap_detector.py — unit tests per §7.3.

Check A (no_split_before_fit_eval): fixture + clean counter-example.
Check B (group_unaware_split): fixture + clean counter-examples for
  groups= keyword and GroupKFold.
"""

import ast
import os
import pytest
import pandas as pd

from pipelineguardian.tools.notebook_parser import load_source, ParsedSource
from pipelineguardian.tools.overlap_detector import (
    check_no_split_before_fit_eval,
    check_group_unaware_split,
    run,
)
from pipelineguardian.tools.schema_inspector import inspect

HERE = os.path.dirname(__file__)
FIXTURES_DIR = os.path.join(HERE, "..", "eval", "synthetic_notebooks")


def _make_pset(source: str) -> ParsedSource:
    return ParsedSource(source=source, path="<test>", is_notebook=False, line_to_cell={})


def _group_df():
    """DataFrame with a patient_id group column (repeating values)."""
    return pd.DataFrame({
        "patient_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
                        6, 6, 7, 7, 8, 8, 9, 9, 10, 10],
        "feature": range(20),
        "target": [0, 1] * 10,
    })


# ---------------------------------------------------------------------------
# Check A — no_split_before_fit_eval
# ---------------------------------------------------------------------------

def test_check_a_fires_on_fixture_15():
    path = os.path.join(FIXTURES_DIR, "15_overlap_no_split.py")
    pset = load_source(path)
    tree = ast.parse(pset.source)
    issues = check_no_split_before_fit_eval(tree, pset)
    assert len(issues) >= 1
    assert issues[0].check_name == "no_split_before_fit_eval"
    assert issues[0].category == "overlap"
    assert issues[0].source == "rule"


def test_check_a_silent_on_fixture_16_clean_split():
    path = os.path.join(FIXTURES_DIR, "16_overlap_clean_split.py")
    pset = load_source(path)
    tree = ast.parse(pset.source)
    issues = check_no_split_before_fit_eval(tree, pset)
    assert issues == []


def test_check_a_silent_when_kfold_present():
    source = (
        "from sklearn.model_selection import KFold, cross_val_score\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "import pandas as pd\n"
        "X = pd.DataFrame({'a': range(10)})\n"
        "y = [0]*5 + [1]*5\n"
        "kf = KFold(n_splits=5)\n"
        "clf = RandomForestClassifier()\n"
        "clf.fit(X, y)\n"
        "clf.score(X, y)\n"
    )
    pset = _make_pset(source)
    tree = ast.parse(source)
    issues = check_no_split_before_fit_eval(tree, pset)
    # KFold is in SPLIT_CALLS so has_any_split == True → no issue
    assert issues == []


# ---------------------------------------------------------------------------
# Check B — group_unaware_split
# ---------------------------------------------------------------------------

def test_check_b_fires_when_group_column_no_groups_arg():
    source = (
        "from sklearn.model_selection import train_test_split\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
    )
    pset = _make_pset(source)
    tree = ast.parse(source)
    df = _group_df()
    signals = inspect(df, target_col="target")
    assert signals.group_columns  # confirm signal is detected
    issues = check_group_unaware_split(tree, pset, signals)
    assert len(issues) >= 1
    assert issues[0].check_name == "group_unaware_split"


def test_check_b_silent_when_groups_kwarg_present():
    source = (
        "from sklearn.model_selection import train_test_split\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, groups=patient_id, random_state=42)\n"
    )
    pset = _make_pset(source)
    tree = ast.parse(source)
    df = _group_df()
    signals = inspect(df, target_col="target")
    issues = check_group_unaware_split(tree, pset, signals)
    assert issues == []


def test_check_b_silent_when_groupkfold_used():
    source = (
        "from sklearn.model_selection import GroupKFold\n"
        "from sklearn.model_selection import train_test_split\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)\n"
        "gkf = GroupKFold(n_splits=5)\n"
    )
    pset = _make_pset(source)
    tree = ast.parse(source)
    df = _group_df()
    signals = inspect(df, target_col="target")
    issues = check_group_unaware_split(tree, pset, signals)
    assert issues == []


def test_check_b_silent_when_no_group_signal():
    source = (
        "from sklearn.model_selection import train_test_split\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)\n"
    )
    pset = _make_pset(source)
    tree = ast.parse(source)
    # No group columns → no signal → no issue
    df = pd.DataFrame({"feature": range(10), "target": [0, 1] * 5})
    signals = inspect(df, target_col="target")
    issues = check_group_unaware_split(tree, pset, signals)
    assert issues == []


def test_fixture_17_fires_check_b_with_group_signals():
    """Fixture 17 uses a group-bearing CSV — verify Check B fires when df is provided."""
    path = os.path.join(FIXTURES_DIR, "17_overlap_group_unaware.py")
    csv_path = os.path.join(FIXTURES_DIR, "17_group_data.csv")
    pset = load_source(path)
    df = pd.read_csv(csv_path)
    signals = inspect(df, target_col="target")
    tree = ast.parse(pset.source)
    issues = run(tree, pset, signals=signals)
    check_names = [i.check_name for i in issues]
    assert "group_unaware_split" in check_names


def test_fixture_18_no_check_b_with_groupkfold():
    """Fixture 18 uses GroupKFold — Check B must be silent."""
    path = os.path.join(FIXTURES_DIR, "18_overlap_group_aware.py")
    csv_path = os.path.join(FIXTURES_DIR, "18_group_data.csv")
    pset = load_source(path)
    df = pd.read_csv(csv_path)
    signals = inspect(df, target_col="target")
    tree = ast.parse(pset.source)
    issues = run(tree, pset, signals=signals)
    check_names = [i.check_name for i in issues]
    assert "group_unaware_split" not in check_names
