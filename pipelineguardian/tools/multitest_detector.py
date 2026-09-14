"""
multitest_detector.py — AST-based, fully deterministic.

Scope (see spec §2.4 — exactly one check, no additional heuristics):
  repeated_test_evaluation — a variable produced as the test-half of
    train_test_split() is passed to .score()/.predict()/.evaluate() on
    MORE THAN ONE distinctly-named model variable, WITH NO cross_val_score/
    GridSearchCV/RandomizedSearchCV call present anywhere in the file.

Priority regression case (§7.2): CV-based model selection followed by exactly
one final .score() on a genuine held-out test set must produce ZERO MULTI_TEST
findings. The has_cv_selection guard is the mechanism that implements this.
"""

import ast
from pipelineguardian.models import Issue, Severity, Confidence, Category, IssueSource


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _line_to_cell(pset, line):
    return pset.line_to_cell.get(line) if pset.line_to_cell else None


def check_repeated_test_evaluation(tree, pset) -> list[Issue]:
    """Flag when the same held-out test variable is scored against multiple
    model variables.

    If CV is used for model selection and only ONE model evaluates the held-out
    test set (spec §7.2 priority case), len(model_vars) is 1, producing ZERO
    issues. If multiple models evaluate the test set, it is flagged as
    repeated test evaluation regardless of whether CV appears elsewhere."""

    # Find variables assigned as the test-half of train_test_split()
    # Test-half positions are odd indices in the unpacked tuple: (X_train, X_test, y_train, y_test)
    test_vars: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and _call_name(node.value) == "train_test_split"):
            for target in node.targets:
                if isinstance(target, ast.Tuple):
                    for i, elt in enumerate(target.elts):
                        if isinstance(elt, ast.Name) and i % 2 == 1:  # test-half positions
                            test_vars.add(elt.id)

    # Count how many distinct model variables evaluate each test variable
    evaluators_per_test_var: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("score", "predict", "evaluate")):
            continue
        if not node.args or not isinstance(node.args[0], ast.Name):
            continue
        arg_name = node.args[0].id
        if arg_name not in test_vars:
            continue
        base = node.func.value
        if not isinstance(base, ast.Name):
            continue
        evaluators_per_test_var.setdefault(arg_name, set()).add(base.id)

    issues = []
    for test_var, model_vars in evaluators_per_test_var.items():
        if len(model_vars) > 1:
            issues.append(Issue(
                check_name="repeated_test_evaluation",
                category=Category.MULTI_TEST,
                source=IssueSource.RULE,
                severity=Severity.HIGH,
                confidence=Confidence.DETERMINISTIC,
                message=f"'{test_var}' is evaluated against {len(model_vars)} different "
                        f"models ({sorted(model_vars)}) with no cross-validation call in the "
                        f"file — the same held-out set is being used to pick among candidates, "
                        f"which is no longer a genuinely independent test.",
                evidence=f"{sorted(model_vars)} all called .score/.predict/.evaluate on '{test_var}'",
                line=None,
                cell=None,
                suggested_fix="Use cross_val_score/GridSearchCV for model selection, and "
                              "reserve a separate, single final evaluation on the test set.",
            ))
    return issues


def run(tree, pset) -> list[Issue]:
    return check_repeated_test_evaluation(tree, pset)
