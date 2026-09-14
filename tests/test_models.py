"""
test_models.py — asserts every existing detector's Issues carry valid
category and source fields per §9 coverage requirement.
"""

import ast
import os
import pandas as pd
import pytest

from pipelineguardian.tools.notebook_parser import load_source, ParsedSource
from pipelineguardian.tools import leakage_detector, reproducibility_checker
from pipelineguardian.tools import overlap_detector, multitest_detector
from pipelineguardian.models import Category, IssueSource

HERE = os.path.dirname(__file__)
FIXTURES_DIR = os.path.join(HERE, "..", "eval", "synthetic_notebooks")


def _make_pset(source: str) -> ParsedSource:
    return ParsedSource(source=source, path="<test>", is_notebook=False, line_to_cell={})


# ---------------------------------------------------------------------------
# leakage_detector: all issues have category and source set
# ---------------------------------------------------------------------------

def test_leakage_scaler_issue_has_category_and_source():
    path = os.path.join(FIXTURES_DIR, "01_scaler_leak.py")
    pset = load_source(path)
    issues = leakage_detector.run(pset)
    assert issues, "Expected at least one issue from 01_scaler_leak.py"
    for issue in issues:
        assert issue.category is not None, f"category is None on {issue.check_name}"
        assert issue.source is not None, f"source is None on {issue.check_name}"
        assert issue.source == IssueSource.RULE.value


def test_leakage_fillna_issue_has_preprocessing_category():
    path = os.path.join(FIXTURES_DIR, "03_fillna_leak.py")
    pset = load_source(path)
    issues = leakage_detector.run(pset)
    assert issues
    for issue in issues:
        assert issue.category == Category.PREPROCESSING.value


def test_leakage_target_in_features_has_other_category():
    path = os.path.join(FIXTURES_DIR, "05_target_in_features.py")
    pset = load_source(path)
    issues = leakage_detector.run(pset)
    assert issues
    for issue in issues:
        assert issue.category == Category.OTHER.value


# ---------------------------------------------------------------------------
# reproducibility_checker: all issues have category and source set
# ---------------------------------------------------------------------------

def test_reproducibility_issues_have_category_and_source():
    source = (
        "import torch\n"
        "import numpy as np\n"
        "import random\n"
        "from sklearn.model_selection import KFold\n"
        "kf = KFold(n_splits=5)\n"
        "x = np.random.rand(10)\n"
        "y = random.choice([1, 2])\n"
        "model = torch.nn.Linear(1, 1)\n"
    )
    pset = _make_pset(source)
    issues = reproducibility_checker.run(pset)
    assert issues
    for issue in issues:
        assert issue.category == Category.REPRODUCIBILITY.value
        assert issue.source == IssueSource.RULE.value


# ---------------------------------------------------------------------------
# overlap_detector: all issues have category=OVERLAP and source=RULE
# ---------------------------------------------------------------------------

def test_overlap_no_split_issue_has_correct_fields():
    path = os.path.join(FIXTURES_DIR, "15_overlap_no_split.py")
    pset = load_source(path)
    tree = ast.parse(pset.source)
    issues = overlap_detector.run(tree, pset)
    assert issues
    for issue in issues:
        assert issue.category == Category.OVERLAP.value
        assert issue.source == IssueSource.RULE.value


# ---------------------------------------------------------------------------
# multitest_detector: all issues have category=MULTI_TEST and source=RULE
# ---------------------------------------------------------------------------

def test_multitest_issue_has_correct_fields():
    path = os.path.join(FIXTURES_DIR, "19_multitest_repeated_peek.py")
    pset = load_source(path)
    tree = ast.parse(pset.source)
    issues = multitest_detector.run(tree, pset)
    assert issues
    for issue in issues:
        assert issue.category == Category.MULTI_TEST.value
        assert issue.source == IssueSource.RULE.value


# ---------------------------------------------------------------------------
# Category and IssueSource enum values are correct
# ---------------------------------------------------------------------------

def test_category_enum_values():
    assert Category.PREPROCESSING.value == "preprocessing"
    assert Category.OVERLAP.value == "overlap"
    assert Category.MULTI_TEST.value == "multi_test"
    assert Category.REPRODUCIBILITY.value == "reproducibility"
    assert Category.VALIDATION_STRATEGY.value == "validation_strategy"
    assert Category.OTHER.value == "other"


def test_issue_source_enum_values():
    assert IssueSource.RULE.value == "rule"
    assert IssueSource.LLM_ONLY.value == "llm_only"
    assert IssueSource.LLM_CONTEXT.value == "llm_context"
    assert IssueSource.HYBRID.value == "hybrid"
