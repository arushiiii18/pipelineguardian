"""
notebook_adapter.py — Minimal adapter to extract normalised Python source
from Jupyter notebook (.ipynb) files for use with the frozen RF classifier.

Design principles:
- The frozen model was trained on synthetic .py *script* text.
- We extract only code-cell source in cell order, strip IPython magics/shell
  commands, and join cells with a single blank line separator — producing text
  that structurally resembles a flat Python script.
- Model weights, TF-IDF vocabulary, feature definitions, and threshold are
  NOT changed.
- Documented limitations are listed at the bottom of this module.

ADAPTER VERSION: 1.0.0
"""
import json
import re
import hashlib
from typing import List, Tuple, Optional

ADAPTER_VERSION = "1.0.0"

# Regex patterns for IPython-specific constructs to strip before featurising
_MAGIC_LINE_RE = re.compile(r"^\s*%%?\w+.*$", re.MULTILINE)  # line/cell magics
_SHELL_LINE_RE = re.compile(r"^\s*!.*$", re.MULTILINE)        # shell commands


def _normalize_source_lines(source) -> str:
    """
    Accept either a list of strings (nbformat 4 style) or a single string
    (nbformat 3 / some generators) and return a single normalized string.
    """
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def _strip_magics(code: str) -> str:
    """
    Remove IPython line magics (%command), cell magics (%%command),
    and shell commands (!command). Lines are removed entirely; indentation
    is preserved for the remaining lines.
    """
    lines = code.splitlines(keepends=True)
    out = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("%%") or stripped.startswith("%") or stripped.startswith("!"):
            # Replace the line with an empty line to preserve line numbers
            out.append("\n" if line.endswith("\n") else "")
        else:
            out.append(line)
    return "".join(out)


def _strip_magics_counted(code: str):
    """
    Same as _strip_magics but also returns the count of stripped magic lines.
    Returns: (n_stripped, cleaned_source)
    """
    lines = code.splitlines(keepends=True)
    out = []
    n_stripped = 0
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("%%") or stripped.startswith("%") or stripped.startswith("!"):
            out.append("\n" if line.endswith("\n") else "")
            n_stripped += 1
        else:
            out.append(line)
    return n_stripped, "".join(out)


class NotebookParseError(Exception):
    """Raised when the notebook JSON cannot be parsed or is structurally invalid."""
    pass


class NotebookAdapter:
    """
    Converts a .ipynb file (as a raw string) into a normalised Python source
    string compatible with the frozen RF classifier's feature extraction.
    """

    def __init__(self, strip_magics: bool = True, cell_separator: str = "\n\n"):
        """
        Args:
            strip_magics: If True, remove IPython magic and shell lines.
            cell_separator: String inserted between successive code cells.
                            A blank line ("\n\n") matches typical script style.
        """
        self.strip_magics = strip_magics
        self.cell_separator = cell_separator

    def extract_code_source(self, raw_content: str) -> Tuple[str, dict]:
        """
        Parse raw .ipynb content and return:
          (normalised_python_source, metadata_dict)

        Raises:
            NotebookParseError: if JSON is invalid or notebook structure is
                                 unrecognised (e.g. missing 'cells' key in
                                 nbformat 4, or missing 'worksheets' in nbformat 3).
        """
        try:
            nb = json.loads(raw_content)
        except (json.JSONDecodeError, ValueError) as exc:
            raise NotebookParseError(f"JSON parse failure: {exc}") from exc

        nbformat = nb.get("nbformat", None)

        if nbformat == 4 or (nbformat is None and "cells" in nb):
            cells = nb.get("cells", [])
        elif nbformat == 3 or (nbformat is None and "worksheets" in nb):
            worksheets = nb.get("worksheets", [])
            cells = []
            for ws in worksheets:
                cells.extend(ws.get("cells", []))
        else:
            raise NotebookParseError(
                f"Unsupported or ambiguous notebook format (nbformat={nbformat})"
            )

        code_cells = [c for c in cells if c.get("cell_type") == "code"]
        total_cells = len(cells)
        code_cell_count = len(code_cells)

        if code_cell_count == 0:
            # Notebook exists but has no code cells; return empty string
            return "", {
                "nbformat": nbformat,
                "total_cells": total_cells,
                "code_cells": 0,
                "magic_lines_stripped": 0,
                "empty_cells_skipped": 0,
            }

        extracted_parts: List[str] = []
        magic_count = 0
        empty_count = 0

        for cell in code_cells:
            raw_src = _normalize_source_lines(cell.get("source", ""))

            if not raw_src.strip():
                empty_count += 1
                continue

            if self.strip_magics:
                n_stripped, raw_src = _strip_magics_counted(raw_src)
                magic_count += n_stripped

            # After stripping magics the cell might be empty — still include
            # so structural positions are preserved (matches training convention)
            extracted_parts.append(raw_src)

        normalised = self.cell_separator.join(extracted_parts)

        return normalised, {
            "nbformat": nbformat,
            "total_cells": total_cells,
            "code_cells": code_cell_count,
            "empty_cells_skipped": empty_count,
            "magic_lines_stripped": magic_count,
            "adapter_version": ADAPTER_VERSION,
        }

    def extract_from_file(self, path: str) -> Tuple[str, dict]:
        """
        Convenience wrapper: read a file (UTF-8, replace errors) and call
        extract_code_source.
        """
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
        return self.extract_code_source(raw)


def adapter_input_hash(raw_content: str) -> str:
    """SHA-256 of the raw notebook content for reproducibility records."""
    return hashlib.sha256(raw_content.encode("utf-8", errors="replace")).hexdigest()


# -------------------------------------------------------------------------
# KNOWN LIMITATIONS OF THIS ADAPTER (v1.0.0)
# -------------------------------------------------------------------------
# 1. LINE NUMBER FIDELITY: Joining cells with a separator shifts line numbers
#    relative to a flat script. The structural features that depend on
#    absolute line numbers (feat[13] fit_before_split_flag, feat[16]
#    split_to_fit_line_delta) will fire based on the concatenated position, not
#    the original notebook cell order. This is a known approximation.
#
# 2. CELL-LOCAL SCOPE: Notebooks can shadow variables across cells. The adapter
#    concatenates cells in order; if a variable is redefined in a later cell
#    the AST-order indicators may differ from runtime behaviour.
#
# 3. MAGIC STRIPPING: Only line (%cmd) and shell (!cmd) magics are stripped.
#    Object-access magics (e.g. %load_ext, %%time) are removed at the line
#    level.  Code that spans a cell magic (%%cython blocks, etc.) is stripped
#    at the first line only; remaining lines are retained as plain code.
#
# 4. TF-IDF VOCABULARY: The frozen TF-IDF was fitted on synthetic .py scripts.
#    Token frequency distributions in ipynb-sourced text may differ, causing
#    TF-IDF features to underestimate coverage of real-world API names.
#    This limitation cannot be resolved without retraining.
#
# 5. OUTPUT-ONLY CELLS: print() outputs, widget outputs, and matplotlib show()
#    are not present in the source; they are in cell 'outputs', which this
#    adapter ignores.  The model was not trained on output content.
# -------------------------------------------------------------------------
