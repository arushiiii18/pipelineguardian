"""
eval.py — real, runnable evaluation against eval/synthetic_notebooks.

Scope: this scores leakage_detector + reproducibility_checker only, since
those are deterministic and their ground truth is unambiguous. The LLM tool
(validation_strategy_reviewer) is intentionally NOT scored here — a
precision/recall number for a non-deterministic LLM judgment, computed
against a handful of hand-labeled cases, would be exactly the kind of
unfalsifiable-sounding metric the project's non-goals rule out. That tool's
behavior is instead demonstrated qualitatively via the before/after demo
pair (see README).

Run: python -m eval.eval
"""

import json
import os
from collections import Counter

from pipelineguardian.tools.notebook_parser import load_source
from pipelineguardian.tools import leakage_detector, reproducibility_checker

HERE = os.path.dirname(__file__)
FIXTURES_DIR = os.path.join(HERE, "synthetic_notebooks")
EXPECTED_PATH = os.path.join(HERE, "expected.json")
RESULTS_PATH = os.path.join(HERE, "results.md")


def run_detectors(path: str) -> list[str]:
    pset = load_source(path)
    issues = leakage_detector.run(pset) + reproducibility_checker.run(pset)
    return [i.check_name for i in issues]


def score() -> dict:
    with open(EXPECTED_PATH) as f:
        expected = json.load(f)

    total_tp = total_fp = total_fn = 0
    per_file_rows = []

    for filename, expected_checks in expected.items():
        path = os.path.join(FIXTURES_DIR, filename)
        got = Counter(run_detectors(path))
        want = Counter(expected_checks)

        # multiset comparison: an extra instance of a check the file DOES
        # exhibit still counts as a false positive if it exceeds the
        # expected count, and vice versa for false negatives.
        tp = sum((got & want).values())
        fp = sum((got - want).values())
        fn = sum((want - got).values())

        total_tp += tp
        total_fp += fp
        total_fn += fn

        per_file_rows.append({
            "file": filename,
            "expected": dict(want),
            "got": dict(got),
            "tp": tp, "fp": fp, "fn": fn,
        })

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 1.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "n_files": len(expected),
        "per_file": per_file_rows,
    }


def write_report(results: dict):
    lines = [
        "# Evaluation Results\n",
        f"Scored against `{os.path.basename(FIXTURES_DIR)}` "
        f"({results['n_files']} synthetic files, deterministic tools only — "
        f"see eval.py docstring for why the LLM tool isn't scored here).\n",
        "## Aggregate\n",
        f"- Precision: **{results['precision']}**",
        f"- Recall: **{results['recall']}**",
        f"- F1: **{results['f1']}**",
        f"- True positives: {results['total_tp']}, False positives: {results['total_fp']}, "
        f"False negatives: {results['total_fn']}\n",
        "## Per-file breakdown\n",
        "| File | Expected | Got | TP | FP | FN |",
        "|---|---|---|---|---|---|",
    ]
    for row in results["per_file"]:
        lines.append(
            f"| {row['file']} | {row['expected']} | {row['got']} | "
            f"{row['tp']} | {row['fp']} | {row['fn']} |"
        )
    with open(RESULTS_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    results = score()
    write_report(results)
    print(f"Precision: {results['precision']}  Recall: {results['recall']}  F1: {results['f1']}")
    print(f"TP={results['total_tp']} FP={results['total_fp']} FN={results['total_fn']}")
    print(f"Full breakdown written to {RESULTS_PATH}")
