"""
schema_inspector.py — pandas-based, deterministic signal extraction.

This tool does NOT emit Issues. It emits a SchemaSignals object that the
agent reads to decide whether validation_strategy_reviewer (the one LLM
tool) is worth invoking. No timestamp/group/imbalance signal -> agent skips
the LLM call entirely. This is the actual "agentic" branch point in the
whole system — everything else here is a fixed pipeline.
"""

from dataclasses import dataclass, field
import pandas as pd

TIMESTAMP_NAME_HINTS = {"date", "time", "timestamp", "created", "updated", "ds"}
GROUP_NAME_HINTS = {"id", "user_id", "customer_id", "group", "session", "patient_id", "subject_id"}
IMBALANCE_THRESHOLD = 0.10  # minority class < 10% of majority -> flag


@dataclass
class SchemaSignals:
    timestamp_columns: list[str] = field(default_factory=list)
    group_columns: list[str] = field(default_factory=list)
    imbalance: dict | None = None  # {"column": ..., "minority_ratio": ...}

    @property
    def has_trigger(self) -> bool:
        return bool(self.timestamp_columns or self.group_columns or self.imbalance)


def _looks_like_timestamp(colname: str, series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    name_hit = any(h in colname.lower() for h in TIMESTAMP_NAME_HINTS)
    if not name_hit:
        return False
    # confirm a reasonable fraction actually parses as dates before trusting the name
    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() > 0.8


def _looks_like_group_id(colname: str, series: pd.Series) -> bool:
    name_hit = any(h in colname.lower() for h in GROUP_NAME_HINTS)
    if not name_hit:
        return False
    # a real group/ID column repeats values (multiple rows per group);
    # a pure primary key (unique every row) isn't a validation-strategy concern
    n = len(series)
    if n == 0:
        return False
    uniqueness = series.nunique(dropna=True) / n
    return uniqueness < 0.98


def inspect(df: pd.DataFrame, target_col: str | None = None) -> SchemaSignals:
    signals = SchemaSignals()

    for col in df.columns:
        if col == target_col:
            continue
        if _looks_like_timestamp(col, df[col]):
            signals.timestamp_columns.append(col)
        elif _looks_like_group_id(col, df[col]):
            signals.group_columns.append(col)

    if target_col and target_col in df.columns:
        counts = df[target_col].value_counts(normalize=True, dropna=True)
        if len(counts) >= 2:
            minority_ratio = counts.min()
            if minority_ratio < IMBALANCE_THRESHOLD:
                signals.imbalance = {
                    "column": target_col,
                    "minority_ratio": round(float(minority_ratio), 4),
                }

    return signals
