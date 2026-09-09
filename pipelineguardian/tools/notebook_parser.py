"""
notebook_parser.py

Turns a .ipynb or .py file into one flat Python source string in execution
order, plus a line -> cell_index map so downstream tools can report which
cell an issue came from. All the AST-based tools operate on the flattened
source, not on notebook JSON directly — this keeps them file-format agnostic.
"""

from __future__ import annotations
from dataclasses import dataclass
import nbformat


@dataclass
class ParsedSource:
    source: str                 # flattened python source
    path: str
    is_notebook: bool
    line_to_cell: dict[int, int]  # 1-indexed source line -> 0-indexed cell number


def _execution_order(nb) -> list:
    """
    Cells are ordered by execution_count when available (this is what the
    author actually ran, which is what matters for leakage order-of-operations).
    Cells that were never run (execution_count is None) are kept in their
    original notebook position, appended after the executed ones, since we
    can't know when they'd run.
    """
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    executed = [c for c in code_cells if c.get("execution_count") is not None]
    unexecuted = [c for c in code_cells if c.get("execution_count") is None]
    executed.sort(key=lambda c: c["execution_count"])
    return executed + unexecuted


def parse_notebook(path: str) -> ParsedSource:
    nb = nbformat.read(path, as_version=4)
    cells = _execution_order(nb)

    lines: list[str] = []
    line_to_cell: dict[int, int] = {}
    for cell_idx, cell in enumerate(cells):
        cell_source = cell.source
        cell_lines = cell_source.splitlines() or [""]
        for line in cell_lines:
            lines.append(line)
            line_to_cell[len(lines)] = cell_idx
        lines.append("")  # blank separator so cell boundaries don't merge statements
        line_to_cell[len(lines)] = cell_idx

    return ParsedSource(
        source="\n".join(lines),
        path=path,
        is_notebook=True,
        line_to_cell=line_to_cell,
    )


def parse_script(path: str) -> ParsedSource:
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    return ParsedSource(source=source, path=path, is_notebook=False, line_to_cell={})


def load_source(path: str) -> ParsedSource:
    if path.endswith(".ipynb"):
        return parse_notebook(path)
    return parse_script(path)
