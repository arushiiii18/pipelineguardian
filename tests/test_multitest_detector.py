"""
test_multitest_detector.py — unit tests per §7.2 (highest priority).

Priority regression case (§7.2): CV-based model selection followed by
exactly one final .score() on a genuine held-out test set must produce
ZERO MULTI_TEST issues. This test must NOT be weakened — both positive
and negative cases are required.
"""

import ast
import os
import pytest

from pipelineguardian.tools.notebook_parser import load_source, ParsedSource
from pipelineguardian.tools.multitest_detector import (
    check_repeated_test_evaluation,
    run,
)

HERE = os.path.dirname(__file__)
FIXTURES_DIR = os.path.join(HERE, "..", "eval", "synthetic_notebooks")


def _make_pset(source: str) -> ParsedSource:
    return ParsedSource(source=source, path="<test>", is_notebook=False, line_to_cell={})


# ---------------------------------------------------------------------------
# §7.2 Priority regression case — CV then final eval → ZERO MULTI_TEST
# ---------------------------------------------------------------------------

def test_cv_then_final_eval_produces_zero_multitest_issues():
    """THE §7.2 PRIORITY REGRESSION CASE.

    cross_val_score for model selection + exactly one final .score() on a
    single genuinely held-out test variable must produce ZERO MULTI_TEST issues.
    If this test ever fails, the has_cv_selection guard is broken."""
    path = os.path.join(FIXTURES_DIR, "20_multitest_cv_then_final_eval.py")
    pset = load_source(path)
    tree = ast.parse(pset.source)
    issues = check_repeated_test_evaluation(tree, pset)
    multitest_issues = [i for i in issues if i.check_name == "repeated_test_evaluation"]
    assert multitest_issues == [], (
        f"§7.2 regression FAILED: expected zero MULTI_TEST issues but got "
        f"{len(multitest_issues)}: {[i.message for i in multitest_issues]}"
    )


# ---------------------------------------------------------------------------
# Positive case — repeated peeking DOES fire without CV
# ---------------------------------------------------------------------------

def test_repeated_test_evaluation_fires_on_fixture_19():
    path = os.path.join(FIXTURES_DIR, "19_multitest_repeated_peek.py")
    pset = load_source(path)
    tree = ast.parse(pset.source)
    issues = check_repeated_test_evaluation(tree, pset)
    assert len(issues) >= 1
    assert issues[0].check_name == "repeated_test_evaluation"
    assert issues[0].category == "multi_test"
    assert issues[0].source == "rule"


def test_repeated_test_evaluation_fires_on_inline_source():
    """Two different model vars both .score() the same X_test — no CV present."""
    source = (
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n"
        "clf1 = RandomForestClassifier()\n"
        "clf1.fit(X_train, y_train)\n"
        "s1 = clf1.score(X_test, y_test)\n"
        "clf2 = GradientBoostingClassifier()\n"
        "clf2.fit(X_train, y_train)\n"
        "s2 = clf2.score(X_test, y_test)\n"
    )
    pset = _make_pset(source)
    tree = ast.parse(source)
    issues = check_repeated_test_evaluation(tree, pset)
    assert len(issues) == 1
    assert issues[0].check_name == "repeated_test_evaluation"


# ---------------------------------------------------------------------------
# Negative cases — no fire when CV is present
# ---------------------------------------------------------------------------

def test_silent_when_cross_val_score_present():
    """cross_val_score on trainval + single final score on X_test → only 1 model scores X_test → no MULTI_TEST issue."""
    source = (
        "from sklearn.model_selection import train_test_split, cross_val_score\n"
        "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n"
        "X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2)\n"
        "cv1 = cross_val_score(RandomForestClassifier(), X_trainval, y_trainval, cv=5)\n"
        "cv2 = cross_val_score(GradientBoostingClassifier(), X_trainval, y_trainval, cv=5)\n"
        "best = RandomForestClassifier()\n"
        "best.fit(X_trainval, y_trainval)\n"
        "score = best.score(X_test, y_test)\n"
    )
    pset = _make_pset(source)
    tree = ast.parse(source)
    issues = check_repeated_test_evaluation(tree, pset)
    assert issues == []


def test_silent_when_gridsearchcv_present():
    source = (
        "from sklearn.model_selection import train_test_split, GridSearchCV\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n"
        "gs = GridSearchCV(RandomForestClassifier(), {'n_estimators': [10, 50]})\n"
        "gs.fit(X_train, y_train)\n"
        "score = gs.score(X_test, y_test)\n"
    )
    pset = _make_pset(source)
    tree = ast.parse(source)
    issues = check_repeated_test_evaluation(tree, pset)
    assert issues == []


def test_silent_when_single_model_on_test_set():
    """Only one model evaluated on X_test — not repeated peeking."""
    source = (
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n"
        "clf = RandomForestClassifier()\n"
        "clf.fit(X_train, y_train)\n"
        "score = clf.score(X_test, y_test)\n"
    )
    pset = _make_pset(source)
    tree = ast.parse(source)
    issues = check_repeated_test_evaluation(tree, pset)
    assert issues == []


def test_cv_and_repeated_test_evaluation_triggers_issue():
    """Even if cross_val_score is present, evaluating X_test on >1 model variables MUST be flagged."""
    source = (
        "from sklearn.model_selection import train_test_split, cross_val_score\n"
        "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n"
        "cv = cross_val_score(RandomForestClassifier(), X_train, y_train, cv=5)\n"
        "clf1 = RandomForestClassifier()\n"
        "clf1.fit(X_train, y_train)\n"
        "s1 = clf1.score(X_test, y_test)\n"
        "clf2 = GradientBoostingClassifier()\n"
        "clf2.fit(X_train, y_train)\n"
        "s2 = clf2.score(X_test, y_test)\n"
    )
    pset = _make_pset(source)
    tree = ast.parse(source)
    issues = check_repeated_test_evaluation(tree, pset)
    assert len(issues) == 1
    assert issues[0].check_name == "repeated_test_evaluation"

