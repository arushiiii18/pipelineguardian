"""
Every synthetic fixture in eval/synthetic_notebooks/ is run through both
detectors and checked against eval/expected.json ground truth. This is the
same fixture set that eval.py will report precision/recall over — writing
it once, testing it twice.
"""

import json
import os
from collections import Counter

import pytest

import ast
from pipelineguardian.tools.notebook_parser import load_source
from pipelineguardian.tools import leakage_detector, reproducibility_checker, overlap_detector, multitest_detector

HERE = os.path.dirname(__file__)
FIXTURES_DIR = os.path.join(HERE, "..", "eval", "synthetic_notebooks")
EXPECTED_PATH = os.path.join(HERE, "..", "eval", "expected.json")

with open(EXPECTED_PATH) as f:
    EXPECTED = json.load(f)


def _all_issue_names(path):
    pset = load_source(path)
    tree = ast.parse(pset.source)
    issues = (
        leakage_detector.run(pset)
        + reproducibility_checker.run(pset)
        + overlap_detector.run(tree, pset)
        + multitest_detector.run(tree, pset)
    )
    return [i.check_name for i in issues]


@pytest.mark.parametrize("filename", list(EXPECTED.keys()))
def test_fixture_matches_expected(filename):
    path = os.path.join(FIXTURES_DIR, filename)
    got = Counter(_all_issue_names(path))
    want = Counter(EXPECTED[filename])
    assert got == want, f"{filename}: expected {dict(want)}, got {dict(got)}"
