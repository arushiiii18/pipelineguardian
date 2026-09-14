"""
context_builder.py — builds NotebookContext from a ParsedSource + optional DataFrame.

NotebookContext contains ONLY structural facts extractable from the source code
and schema-inspector signals. It MUST NEVER contain rule-engine verdicts,
Issue objects, or any output from leakage_detector or reproducibility_checker.

Allowed imports (enforced by test_context_builder_no_rule_import.py):
  ast, notebook_parser, schema_inspector, pandas — ONLY.

FORBIDDEN (import ban, automatically tested):
  leakage_detector, reproducibility_checker — do NOT import these anywhere
  in this file, including inside function bodies.

AST walks here duplicate ~10 lines from leakage_detector.py intentionally.
The duplication keeps the import ban a complete guarantee: context_builder
cannot accidentally surface rule verdicts via a shared helper.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from pipelineguardian.tools.notebook_parser import ParsedSource
from pipelineguardian.tools.schema_inspector import inspect as inspect_schema


@dataclass
class NotebookContext:
    """Structural facts passed to llm_with_context and the escalated half of hybrid.

    ALLOWED fields (spec §1.3):
      source                  — verbatim flattened source
      columns / dtypes / sample_rows — raw schema, if df supplied
      split_call_lines        — line numbers of split-related calls
      fit_transform_call_sites — scaler/encoder/imputer .fit()/.fit_transform() locations
      timestamp_columns       — derived from schema_inspector (allowed)
      group_columns           — derived from schema_inspector (allowed)
      imbalance               — derived from schema_inspector (allowed)

    FORBIDDEN fields (never add):
      predicted_category, rule_verdict, leakage_findings, detector_output,
      or any field whose value originates from calling leakage_detector.run()
      or reproducibility_checker.run() or from reading an Issue object.
    """
    source: str
    columns: Optional[list[str]] = None
    dtypes: Optional[dict[str, str]] = None
    sample_rows: Optional[list[dict]] = None
    split_call_lines: list[int] = field(default_factory=list)
    fit_transform_call_sites: list[dict] = field(default_factory=list)
    timestamp_columns: list[str] = field(default_factory=list)
    group_columns: list[str] = field(default_factory=list)
    imbalance: Optional[dict] = None


# ---------------------------------------------------------------------------
# AST helpers — local copies, NOT imported from leakage_detector or overlap_detector
# (required by the import ban in §9.5)
# ---------------------------------------------------------------------------

_SPLIT_CALL_NAMES = {
    "train_test_split", "KFold", "StratifiedKFold",
    "GroupKFold", "cross_val_score",
}

_FIT_CLASSES = {
    "StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer",
    "LabelEncoder", "OneHotEncoder", "OrdinalEncoder", "SimpleImputer",
    "KNNImputer", "TfidfVectorizer", "CountVectorizer", "PowerTransformer",
}


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _find_split_call_lines(tree: ast.AST) -> list[int]:
    """Return line numbers of all train_test_split/KFold/GroupKFold/cross_val_score calls."""
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) in _SPLIT_CALL_NAMES:
            lines.append(node.lineno)
    return sorted(lines)


def _track_fit_class_vars(tree: ast.AST) -> dict[str, int]:
    """Map variable name → line it was assigned a Scaler/Encoder/Imputer instance."""
    tracked: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            callee = node.value.func
            cls_name = callee.id if isinstance(callee, ast.Name) else getattr(callee, "attr", None)
            if cls_name in _FIT_CLASSES:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        tracked[target.id] = node.lineno
    return tracked


def _find_fit_transform_sites(tree: ast.AST, fit_class_vars: dict[str, int]) -> list[dict]:
    """Return [{var, call, line}] for .fit()/.fit_transform() on known scaler/encoder vars."""
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in ("fit", "fit_transform"):
            continue
        base = node.func.value
        if isinstance(base, ast.Name) and base.id in fit_class_vars:
            sites.append({
                "var": base.id,
                "call": node.func.attr,
                "line": node.lineno,
            })
    return sorted(sites, key=lambda s: s["line"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build(
    pset: ParsedSource,
    df: Optional[pd.DataFrame] = None,
    target_col: Optional[str] = None,
) -> NotebookContext:
    """Build a NotebookContext from a ParsedSource and an optional DataFrame.

    Uses its own local AST walk — does NOT call any function from
    leakage_detector.py or overlap_detector.py (import ban, §9.5).
    """
    tree = ast.parse(pset.source)

    # Raw schema facts — only if df is supplied
    columns: Optional[list[str]] = None
    dtypes: Optional[dict[str, str]] = None
    sample_rows: Optional[list[dict]] = None
    timestamp_columns: list[str] = []
    group_columns: list[str] = []
    imbalance: Optional[dict] = None

    if df is not None:
        columns = list(df.columns)
        dtypes = {col: str(df[col].dtype) for col in df.columns}
        sample_rows = df.head(3).to_dict(orient="records")

        # Derived signals from schema_inspector — explicitly allowed in NotebookContext
        signals = inspect_schema(df, target_col)
        timestamp_columns = signals.timestamp_columns
        group_columns = signals.group_columns
        imbalance = signals.imbalance

    # Structural facts via local AST walk
    split_call_lines = _find_split_call_lines(tree)
    fit_class_vars = _track_fit_class_vars(tree)
    fit_transform_call_sites = _find_fit_transform_sites(tree, fit_class_vars)

    return NotebookContext(
        source=pset.source,
        columns=columns,
        dtypes=dtypes,
        sample_rows=sample_rows,
        split_call_lines=split_call_lines,
        fit_transform_call_sites=fit_transform_call_sites,
        timestamp_columns=timestamp_columns,
        group_columns=group_columns,
        imbalance=imbalance,
    )
