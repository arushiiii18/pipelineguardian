import pandas as pd
from pipelineguardian.tools.schema_inspector import inspect


def test_detects_timestamp_column_by_dtype():
    df = pd.DataFrame({
        "created_at": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "value": [1, 2, 3],
    })
    signals = inspect(df)
    assert "created_at" in signals.timestamp_columns
    assert signals.has_trigger


def test_detects_timestamp_column_by_parseable_string():
    df = pd.DataFrame({
        "event_date": ["2023-05-01", "2023-05-02", "2023-05-03"] * 10,
        "value": list(range(30)),
    })
    signals = inspect(df)
    assert "event_date" in signals.timestamp_columns


def test_detects_group_id_column_with_repeats():
    df = pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 3],
        "value": [10, 11, 12, 13, 14, 15],
    })
    signals = inspect(df)
    assert "user_id" in signals.group_columns


def test_pure_primary_key_not_flagged_as_group():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],  # unique every row -> not a validation-strategy concern
        "value": [10, 11, 12, 13, 14],
    })
    signals = inspect(df)
    assert "id" not in signals.group_columns


def test_detects_class_imbalance():
    df = pd.DataFrame({
        "target": [0] * 95 + [1] * 5,
        "value": range(100),
    })
    signals = inspect(df, target_col="target")
    assert signals.imbalance is not None
    assert signals.imbalance["column"] == "target"
    assert signals.has_trigger


def test_balanced_target_no_trigger():
    df = pd.DataFrame({
        "target": [0] * 50 + [1] * 50,
        "value": range(100),
    })
    signals = inspect(df, target_col="target")
    assert signals.imbalance is None
    assert not signals.has_trigger
