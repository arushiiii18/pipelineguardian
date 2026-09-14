"""
overlap_detector.py — AST-based, fully deterministic.

Scope (see spec §2.3 — exactly two checks, no additional heuristics):
  A. no_split_before_fit_eval  — a model is fit and evaluated on the identical
                                   argument variable with no train_test_split/KFold/
                                   StratifiedKFold/GroupKFold/cross_val_score call
                                   anywhere in the file.
  B. group_unaware_split       — a group-like column is detected (via SchemaSignals)
                                   AND train_test_split() has no groups= keyword AND
                                   no GroupKFold/StratifiedGroupKFold call exists.

The two checks are intentionally mechanical and may disagree with
validation_strategy_reviewer's judgment about whether group-aware splitting is
actually *necessary* — that disagreement is experimental data, not a bug.
"""

import ast
from pipelineguardian.models import Issue, Severity, Confidence, Category, IssueSource
from pipelineguardian.tools.schema_inspector import SchemaSignals

SPLIT_CALLS = {"train_test_split", "KFold", "StratifiedKFold", "GroupKFold", "cross_val_score"}


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _line_to_cell(pset, line):
    return pset.line_to_cell.get(line) if pset.line_to_cell else None


def check_no_split_before_fit_eval(tree, pset) -> list[Issue]:
    """Check A — §2.3: flag when a model is fit and evaluated on the identical
    variable with no split call of any kind present in the file."""
    has_any_split = any(
        isinstance(n, ast.Call) and _call_name(n) in SPLIT_CALLS
        for n in ast.walk(tree)
    )
    if has_any_split:
        return []

    issues = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("score", "predict", "evaluate")):
            continue
        if not node.args or not isinstance(node.args[0], ast.Name):
            continue
        eval_var = node.args[0].id
        # find a .fit(same_var) call on the same base object earlier in the tree
        base = node.func.value
        if not isinstance(base, ast.Name):
            continue
        for other in ast.walk(tree):
            if (isinstance(other, ast.Call) and isinstance(other.func, ast.Attribute)
                    and other.func.attr == "fit" and isinstance(other.func.value, ast.Name)
                    and other.func.value.id == base.id and other.args
                    and isinstance(other.args[0], ast.Name) and other.args[0].id == eval_var
                    and other.lineno < node.lineno):
                issues.append(Issue(
                    check_name="no_split_before_fit_eval",
                    category=Category.OVERLAP,
                    source=IssueSource.RULE,
                    severity=Severity.HIGH,
                    confidence=Confidence.DETERMINISTIC,
                    message=f"'{base.id}' is fit and evaluated on the same variable "
                            f"'{eval_var}' with no train_test_split/KFold call anywhere "
                            f"in the file — training and evaluation data overlap completely.",
                    evidence=f"{base.id}.fit({eval_var}) ... {base.id}.{node.func.attr}({eval_var})",
                    line=node.lineno,
                    cell=_line_to_cell(pset, node.lineno),
                    suggested_fix="Split the data with train_test_split() (or use "
                                  "cross-validation) before fitting, and evaluate only "
                                  "on the held-out portion.",
                ))
    return issues


def check_group_unaware_split(tree, pset, signals: SchemaSignals) -> list[Issue]:
    """Check B — §2.3: flag when group-like columns exist but train_test_split
    is called without groups= and no GroupKFold/StratifiedGroupKFold is present."""
    if not signals.group_columns:
        return []

    has_group_aware = any(
        isinstance(n, ast.Call) and _call_name(n) in ("GroupKFold", "StratifiedGroupKFold")
        for n in ast.walk(tree)
    )
    if has_group_aware:
        return []

    issues = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) == "train_test_split":
            has_groups_kw = any(kw.arg == "groups" for kw in node.keywords)
            if not has_groups_kw:
                issues.append(Issue(
                    check_name="group_unaware_split",
                    category=Category.OVERLAP,
                    source=IssueSource.RULE,
                    severity=Severity.MEDIUM,
                    confidence=Confidence.DETERMINISTIC,
                    message=f"Column(s) {signals.group_columns} look like repeated group/entity "
                            f"identifiers, but train_test_split() on line {node.lineno} has no "
                            f"groups= argument and no GroupKFold is used elsewhere — the same "
                            f"group may appear in both train and test.",
                    evidence=f"train_test_split(...) at line {node.lineno}, no groups=",
                    line=node.lineno,
                    cell=_line_to_cell(pset, node.lineno),
                    suggested_fix="Use GroupKFold or pass groups=<group_column> to "
                                  "train_test_split so the same entity never appears in "
                                  "both splits.",
                ))
    return issues


def run(tree, pset, signals: SchemaSignals | None = None) -> list[Issue]:
    """Run overlap checks. Check A runs unconditionally; Check B only when
    signals (from schema_inspector) are provided."""
    issues = []
    issues += check_no_split_before_fit_eval(tree, pset)
    if signals is not None:
        issues += check_group_unaware_split(tree, pset, signals)
    return issues
