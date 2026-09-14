"""
test_context_builder_no_rule_import.py — §9.5 import-ban test.

Exact implementation specified in the spec §9.5.
Parses context_builder.py's AST and asserts that neither leakage_detector
nor reproducibility_checker appears in any import statement.
"""

import ast


def test_context_builder_never_imports_rule_detectors():
    with open("pipelineguardian/tools/context_builder.py") as f:
        tree = ast.parse(f.read())
    banned = {"leakage_detector", "reproducibility_checker"}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[-1] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[-1])
            imported.update(a.name for a in node.names)
    assert not (imported & banned), (
        f"context_builder.py imports banned module(s): {imported & banned}"
    )
