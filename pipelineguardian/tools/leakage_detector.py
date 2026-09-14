"""
leakage_detector.py — AST-based, fully deterministic.

Scope (see README non-goals — this is NOT exhaustive leakage coverage):
  1. scaler_fit_before_split   — Scaler/Encoder/Imputer .fit() or .fit_transform()
                                  called before the first train_test_split() call.
  2. fillna_uses_full_dataset_stat — df.fillna(df[...].mean()/median()/std()/...)
                                  pattern, computed before the split line.
  3. target_column_in_feature_list — the y-target column name literally present
                                  in an explicit X = df[[...]] feature list.

Each check is independent. Order-of-operations is inferred from source line
number, which is valid because notebook_parser.py flattens cells in
*execution order*, not file order.
"""

import ast
from pipelineguardian.models import Issue, Severity, Confidence, Category, IssueSource

FIT_CLASSES = {
    "StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer",
    "LabelEncoder", "OneHotEncoder", "OrdinalEncoder", "SimpleImputer",
    "KNNImputer", "TfidfVectorizer", "CountVectorizer", "PowerTransformer",
}
AGG_FUNCS = {"mean", "median", "std", "min", "max", "mode", "sum"}


def _line_to_cell(pset, line):
    return pset.line_to_cell.get(line) if pset.line_to_cell else None


def _find_split_line(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) == "train_test_split":
            return node.lineno
    return None


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _track_fit_class_vars(tree) -> dict[str, int]:
    """Map variable name -> line it was assigned a Scaler/Encoder/Imputer instance."""
    tracked = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            callee = node.value.func
            cls_name = callee.id if isinstance(callee, ast.Name) else getattr(callee, "attr", None)
            if cls_name in FIT_CLASSES:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        tracked[target.id] = node.lineno
    return tracked


def check_scaler_fit_before_split(tree, pset) -> list[Issue]:
    issues = []
    split_line = _find_split_line(tree)
    if split_line is None:
        return issues  # no split call found — order can't be judged

    tracked_vars = _track_fit_class_vars(tree)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in ("fit", "fit_transform"):
            continue
        base = node.func.value
        if not isinstance(base, ast.Name):
            continue
        if base.id not in tracked_vars:
            continue
        if node.lineno < split_line:
            issues.append(Issue(
                check_name="scaler_fit_before_split",
                category=Category.PREPROCESSING,
                source=IssueSource.RULE,
                severity=Severity.HIGH,
                confidence=Confidence.DETERMINISTIC,
                message=f"'{base.id}.{node.func.attr}()' is called on line {node.lineno}, "
                        f"before train_test_split() on line {split_line}. Statistics from "
                        f"the test set leak into training.",
                evidence=f"{base.id}.{node.func.attr}(...) at line {node.lineno}",
                line=node.lineno,
                cell=_line_to_cell(pset, node.lineno),
                suggested_fix=(
                    f"Move '{base.id}.fit(...)' to after train_test_split(), fit only on "
                    f"the training split, then use .transform() (not .fit_transform()) on test data."
                ),
            ))
    return issues


def check_fillna_uses_full_dataset_stat(tree, pset) -> list[Issue]:
    issues = []
    split_line = _find_split_line(tree)

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "fillna"):
            continue
        if not node.args:
            continue
        arg = node.args[0]
        if not (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute)
                and arg.func.attr in AGG_FUNCS):
            continue
        if split_line is not None and node.lineno >= split_line:
            continue  # happens after split — not necessarily leakage, skip
        issues.append(Issue(
            check_name="fillna_uses_full_dataset_stat",
            category=Category.PREPROCESSING,
            source=IssueSource.RULE,
            severity=Severity.HIGH,
            confidence=Confidence.DETERMINISTIC,
            message=f"fillna() on line {node.lineno} imputes using a statistic "
                    f"('{arg.func.attr}') computed over the full dataset before the split.",
            evidence=ast.unparse(node),
            line=node.lineno,
            cell=_line_to_cell(pset, node.lineno),
            suggested_fix=(
                f"Compute the '{arg.func.attr}()' only on the training split after "
                f"train_test_split(), then apply that same value to fill both train and test."
            ),
        ))
    return issues


def _extract_target_col_name(tree) -> str | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "y":
            val = node.value
            if isinstance(val, ast.Subscript):
                slice_node = val.slice
                if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
                    return slice_node.value
    return None


def check_target_column_in_feature_list(tree, pset) -> list[Issue]:
    issues = []
    target_col = _extract_target_col_name(tree)
    if target_col is None:
        return issues

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "X"):
            continue
        val = node.value
        if not isinstance(val, ast.Subscript):
            continue
        slice_node = val.slice
        if isinstance(slice_node, ast.List):
            col_names = [e.value for e in slice_node.elts
                         if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            if target_col in col_names:
                issues.append(Issue(
                    check_name="target_column_in_feature_list",
                    category=Category.OTHER,
                    source=IssueSource.RULE,
                    severity=Severity.HIGH,
                    confidence=Confidence.DETERMINISTIC,
                    message=f"Target column '{target_col}' (used to build y) also appears "
                            f"in the explicit feature list assigned to X on line {node.lineno}.",
                    evidence=ast.unparse(node),
                    line=node.lineno,
                    cell=_line_to_cell(pset, node.lineno),
                    suggested_fix=f"Remove '{target_col}' from the X feature list.",
                ))
    return issues


def run(pset) -> list[Issue]:
    tree = ast.parse(pset.source)
    issues = []
    issues += check_scaler_fit_before_split(tree, pset)
    issues += check_fillna_uses_full_dataset_stat(tree, pset)
    issues += check_target_column_in_feature_list(tree, pset)
    return issues
