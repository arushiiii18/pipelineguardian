"""
test_reproducibility_checker.py — unit tests per §7.1.

Covers:
- check_missing_seed_python_random fires on a fixture using random.choice()
  with no random.seed() call
- check_missing_seed_python_random does NOT fire when random.seed() is present
- run() now exposes five possible check types (not four)
"""

import ast
import os
import pytest

from pipelineguardian.tools.notebook_parser import load_source
from pipelineguardian.tools.reproducibility_checker import (
    check_missing_seed_python_random,
    run,
)

HERE = os.path.dirname(__file__)
FIXTURES_DIR = os.path.join(HERE, "..", "eval", "synthetic_notebooks")


def _make_pset(source: str):
    """Minimal ParsedSource-like object from a source string."""
    from pipelineguardian.tools.notebook_parser import ParsedSource
    return ParsedSource(source=source, path="<test>", is_notebook=False, line_to_cell={})


# ---------------------------------------------------------------------------
# check_missing_seed_python_random: fires on unseeded usage
# ---------------------------------------------------------------------------

def test_missing_seed_python_random_fires_without_seed():
    source = "import random\nresult = random.choice([1, 2, 3])\n"
    pset = _make_pset(source)
    tree = ast.parse(source)
    issues = check_missing_seed_python_random(source, tree, pset)
    assert len(issues) == 1
    assert issues[0].check_name == "missing_seed_python_random"


def test_missing_seed_python_random_silent_when_seeded():
    source = "import random\nrandom.seed(42)\nresult = random.choice([1, 2, 3])\n"
    pset = _make_pset(source)
    tree = ast.parse(source)
    issues = check_missing_seed_python_random(source, tree, pset)
    assert issues == []


def test_missing_seed_python_random_silent_when_no_import():
    """If 'import random' is absent, the check must not fire."""
    source = "result = [1, 2, 3]  # no import random at all\n"
    pset = _make_pset(source)
    tree = ast.parse(source)
    issues = check_missing_seed_python_random(source, tree, pset)
    assert issues == []


# ---------------------------------------------------------------------------
# Fixture-based tests (§7.1)
# ---------------------------------------------------------------------------

def test_fixture_13_triggers_missing_seed_python_random():
    path = os.path.join(FIXTURES_DIR, "13_missing_seed_python_random.py")
    pset = load_source(path)
    issues = run(pset)
    check_names = [i.check_name for i in issues]
    assert "missing_seed_python_random" in check_names


def test_fixture_14_no_missing_seed_python_random():
    path = os.path.join(FIXTURES_DIR, "14_clean_seeded_python_random.py")
    pset = load_source(path)
    issues = run(pset)
    check_names = [i.check_name for i in issues]
    assert "missing_seed_python_random" not in check_names


# ---------------------------------------------------------------------------
# run() now has five possible check types
# ---------------------------------------------------------------------------

def test_run_exposes_five_check_types():
    """Verify run() calls all five checks. We do this by building a synthetic
    source that triggers every reproducibility check and confirming we get
    exactly five distinct check_names from run()."""
    # Source that triggers: missing_random_state (KFold), missing_seed_torch,
    # missing_seed_numpy, missing_seed_python_random.
    # missing_requirements_pins is filesystem-based — tested separately; not
    # triggered in this in-memory fixture.
    source = (
        "import torch\n"
        "import numpy as np\n"
        "import random\n"
        "from sklearn.model_selection import KFold\n"
        "kf = KFold(n_splits=5)\n"           # triggers missing_random_state
        "x = np.random.rand(10)\n"           # triggers missing_seed_numpy
        "y = random.choice([1, 2])\n"        # triggers missing_seed_python_random
        "model = torch.nn.Linear(1, 1)\n"    # triggers missing_seed_torch
    )
    pset = _make_pset(source)
    issues = run(pset)
    check_names = {i.check_name for i in issues}
    expected_checks = {
        "missing_random_state",
        "missing_seed_torch",
        "missing_seed_numpy",
        "missing_seed_python_random",
    }
    assert expected_checks.issubset(check_names), (
        f"Expected all of {expected_checks}, got {check_names}"
    )
