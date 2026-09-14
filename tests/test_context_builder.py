"""
test_context_builder.py — unit tests per §9 coverage.

Verifies:
- NotebookContext shape is correct for a fixture with timestamp+group signals
- No forbidden fields are present on the returned object
- split_call_lines and fit_transform_call_sites are populated correctly
"""

import os
import pandas as pd
import pytest

from pipelineguardian.tools.notebook_parser import load_source, ParsedSource
from pipelineguardian.tools.context_builder import build, NotebookContext

HERE = os.path.dirname(__file__)
FIXTURES_DIR = os.path.join(HERE, "..", "eval", "synthetic_notebooks")

FORBIDDEN_FIELD_NAMES = {
    "predicted_category",
    "rule_verdict",
    "leakage_findings",
    "detector_output",
}


def _make_pset(source: str) -> ParsedSource:
    return ParsedSource(source=source, path="<test>", is_notebook=False, line_to_cell={})


def _timestamp_group_df():
    """DataFrame with a timestamp column and a group/ID column with repeated values."""
    import numpy as np
    return pd.DataFrame({
        "event_date": pd.date_range("2024-01-01", periods=30, freq="D"),
        "patient_id": [i // 3 for i in range(30)],
        "feature": range(30),
        "target": [0, 1] * 15,
    })


# ---------------------------------------------------------------------------
# Shape correctness
# ---------------------------------------------------------------------------

def test_context_shape_no_df():
    source = "import pandas as pd\ndf = pd.read_csv('data.csv')\n"
    pset = _make_pset(source)
    ctx = build(pset)
    assert isinstance(ctx, NotebookContext)
    assert ctx.source == source
    assert ctx.columns is None
    assert ctx.dtypes is None
    assert ctx.sample_rows is None
    assert isinstance(ctx.split_call_lines, list)
    assert isinstance(ctx.fit_transform_call_sites, list)


def test_context_shape_with_df_and_signals():
    source = (
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.model_selection import train_test_split\n"
        "scaler = StandardScaler()\n"
        "scaler.fit_transform(X)\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n"
    )
    pset = _make_pset(source)
    df = _timestamp_group_df()
    ctx = build(pset, df=df, target_col="target")

    # Schema fields populated
    assert ctx.columns is not None
    assert "event_date" in ctx.columns
    assert ctx.dtypes is not None
    assert ctx.sample_rows is not None
    assert len(ctx.sample_rows) <= 3

    # Structural fields populated
    assert len(ctx.split_call_lines) >= 1         # train_test_split
    assert len(ctx.fit_transform_call_sites) >= 1  # scaler.fit_transform

    # Derived signals from schema_inspector
    assert "event_date" in ctx.timestamp_columns
    assert "patient_id" in ctx.group_columns


def test_split_call_lines_populated():
    source = (
        "from sklearn.model_selection import train_test_split, KFold\n"
        "X_train, X_test = train_test_split(X, y)\n"
        "kf = KFold(n_splits=5)\n"
    )
    pset = _make_pset(source)
    ctx = build(pset)
    assert len(ctx.split_call_lines) == 2  # train_test_split + KFold


def test_fit_transform_sites_populated():
    source = (
        "from sklearn.preprocessing import StandardScaler, MinMaxScaler\n"
        "scaler1 = StandardScaler()\n"
        "scaler2 = MinMaxScaler()\n"
        "scaler1.fit_transform(X)\n"
        "scaler2.fit(X_train)\n"
    )
    pset = _make_pset(source)
    ctx = build(pset)
    assert len(ctx.fit_transform_call_sites) == 2
    call_names = {s["call"] for s in ctx.fit_transform_call_sites}
    assert "fit_transform" in call_names
    assert "fit" in call_names


# ---------------------------------------------------------------------------
# No forbidden fields
# ---------------------------------------------------------------------------

def test_no_forbidden_fields_on_context():
    """Explicitly verify that forbidden field names are not present."""
    source = "import pandas as pd\n"
    pset = _make_pset(source)
    ctx = build(pset)
    ctx_dict = vars(ctx)
    for forbidden in FORBIDDEN_FIELD_NAMES:
        assert forbidden not in ctx_dict, f"Forbidden field '{forbidden}' found on NotebookContext"


def test_sample_rows_limited_to_three():
    source = "x = 1\n"
    pset = _make_pset(source)
    df = pd.DataFrame({"a": range(100), "b": range(100)})
    ctx = build(pset, df=df)
    assert ctx.sample_rows is not None
    assert len(ctx.sample_rows) <= 3
