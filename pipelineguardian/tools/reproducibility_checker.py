"""
reproducibility_checker.py — AST + regex, fully deterministic.

Scope:
  1. missing_random_state   — calls to train_test_split / common sklearn
                               estimators / KMeans that omit random_state.
  2. missing_seed_torch      — 'import torch' present, but no torch.manual_seed() call.
  3. missing_seed_numpy      — numpy.random.* used, but no np.random.seed() call.
  4. missing_seed_python_random — random module used, but no random.seed() call.
  5. missing_requirements_pins — no requirements.txt in project dir, or one
                               exists but has zero '==' version pins.
"""

import ast
import os
from pipelineguardian.models import Issue, Severity, Confidence, Category, IssueSource

NEEDS_RANDOM_STATE = {
    "train_test_split", "KFold", "StratifiedKFold", "RandomForestClassifier",
    "RandomForestRegressor", "KMeans", "GradientBoostingClassifier",
    "GradientBoostingRegressor", "XGBClassifier", "XGBRegressor",
    "LogisticRegression", "DecisionTreeClassifier", "DecisionTreeRegressor",
    "train_test_split", "shuffle",
}


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _line_to_cell(pset, line):
    return pset.line_to_cell.get(line) if pset.line_to_cell else None


def check_missing_random_state(tree, pset) -> list[Issue]:
    issues = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name not in NEEDS_RANDOM_STATE:
            continue
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "random":
            continue
        has_rs = any(kw.arg == "random_state" for kw in node.keywords)
        if has_rs:
            continue
        issues.append(Issue(
            check_name="missing_random_state",
            category=Category.REPRODUCIBILITY,
            source=IssueSource.RULE,
            severity=Severity.MEDIUM,
            confidence=Confidence.DETERMINISTIC,
            message=f"'{name}(...)' on line {node.lineno} has no random_state — "
                    f"results won't be reproducible run to run.",
            evidence=ast.unparse(node),
            line=node.lineno,
            cell=_line_to_cell(pset, node.lineno),
            suggested_fix=f"Add an explicit random_state=<int> to '{name}(...)'.",
        ))
    return issues


def check_missing_seed_torch(source, tree, pset) -> list[Issue]:
    imports_torch = "import torch" in source
    if not imports_torch:
        return []
    seeded = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "manual_seed"
        for n in ast.walk(tree)
    )
    if seeded:
        return []
    return [Issue(
        check_name="missing_seed_torch",
        category=Category.REPRODUCIBILITY,
        source=IssueSource.RULE,
        severity=Severity.MEDIUM,
        confidence=Confidence.DETERMINISTIC,
        message="torch is imported and used, but torch.manual_seed(...) is never called.",
        evidence="import torch  (no torch.manual_seed found anywhere in the source)",
        line=None,
        cell=None,
        suggested_fix="Call torch.manual_seed(<int>) once near the top of the script, "
                       "before any model init or data shuffling.",
    )]


def check_missing_seed_numpy(source, tree, pset) -> list[Issue]:
    uses_np_random = ("np.random." in source) or ("numpy.random." in source)
    if not uses_np_random:
        return []
    seeded = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "seed"
        and isinstance(n.func.value, ast.Attribute) and n.func.value.attr == "random"
        for n in ast.walk(tree)
    )
    if seeded:
        return []
    return [Issue(
        check_name="missing_seed_numpy",
        category=Category.REPRODUCIBILITY,
        source=IssueSource.RULE,
        severity=Severity.MEDIUM,
        confidence=Confidence.DETERMINISTIC,
        message="numpy.random is used, but np.random.seed(...) is never called.",
        evidence="np.random.* used without np.random.seed(...)",
        line=None,
        cell=None,
        suggested_fix="Call np.random.seed(<int>) near the top of the script.",
    )]


def check_missing_seed_python_random(source, tree, pset) -> list[Issue]:
    uses_random = ("random." in source) and ("import random" in source)
    if not uses_random:
        return []
    seeded = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "seed" and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "random"
        for n in ast.walk(tree)
    )
    if seeded:
        return []
    return [Issue(
        check_name="missing_seed_python_random",
        category=Category.REPRODUCIBILITY,
        source=IssueSource.RULE,
        severity=Severity.MEDIUM,
        confidence=Confidence.DETERMINISTIC,
        message="Python's random module is used, but random.seed(...) is never called.",
        evidence="random.* used without random.seed(...)",
        line=None,
        cell=None,
        suggested_fix="Call random.seed(<int>) near the top of the script.",
    )]


def check_missing_requirements_pins(project_dir: str) -> list[Issue]:
    req_path = os.path.join(project_dir, "requirements.txt")
    if not os.path.exists(req_path):
        return [Issue(
            check_name="missing_requirements_file",
            category=Category.REPRODUCIBILITY,
            source=IssueSource.RULE,
            severity=Severity.LOW,
            confidence=Confidence.DETERMINISTIC,
            message="No requirements.txt found in the project directory.",
            evidence=f"checked path: {req_path}",
            suggested_fix="Add a requirements.txt (ideally via `pip freeze > requirements.txt`).",
        )]
    with open(req_path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    pinned = [l for l in lines if "==" in l]
    if lines and not pinned:
        return [Issue(
            check_name="unpinned_requirements",
            category=Category.REPRODUCIBILITY,
            source=IssueSource.RULE,
            severity=Severity.LOW,
            confidence=Confidence.DETERMINISTIC,
            message="requirements.txt exists but has no '==' version pins.",
            evidence=f"{len(lines)} dependencies listed, 0 pinned",
            suggested_fix="Pin exact versions, e.g. `pandas==2.2.2`, so the environment is reproducible.",
        )]
    return []


def run(pset, project_dir: str | None = None) -> list[Issue]:
    tree = ast.parse(pset.source)
    issues = []
    issues += check_missing_random_state(tree, pset)
    issues += check_missing_seed_torch(pset.source, tree, pset)
    issues += check_missing_seed_numpy(pset.source, tree, pset)
    issues += check_missing_seed_python_random(pset.source, tree, pset)
    if project_dir:
        issues += check_missing_requirements_pins(project_dir)
    return issues
