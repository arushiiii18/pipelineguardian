"""
run_yang_evaluation_v2.py — Corrected scored external Yang evaluation.

Changes from v1 (run_yang_evaluation.py):
  - Uses NotebookAdapter to extract code-cell source from .ipynb files
    instead of passing raw JSON strings to the model.
  - Adds explicit ABSTAIN outcome for parse failures / empty notebooks.
  - Reports coverage, processed/abstained counts, and a conservative
    accounting where abstained positives count as missed (FN).
  - Computes trivial baselines (always-0, always-1, regex heuristic).
  - Generates a full eligibility manifest with reasons.
  - Records checkpoint hash, adapter version, GT hash.
  - Does NOT retrain, retune, or select features/threshold using Yang outcomes.

Original v1 result (raw-JSON input representation failure) is preserved in
results/yang_scored_evaluation.json. This file writes to
results/yang_scored_evaluation_v2.json.

ADAPTER VERSION: 1.0.0 (frozen before any Yang outcomes were inspected)
"""
import os
import sys
import json
import csv
import hashlib
import joblib
import re
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import math

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from notebook_adapter import NotebookAdapter, NotebookParseError, ADAPTER_VERSION

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
YANG_NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "eval", "yang_corpus", "notebooks")
DEV_IDS_PATH       = os.path.join(PROJECT_ROOT, "eval", "task2_dev_ids.txt")
GROUND_TRUTH_PATH  = os.path.join(PROJECT_ROOT, "eval", "yang_corpus", "ground_truth", "ground-truth.csv")
CHECKPOINT_PATH    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "baseline_rf.joblib")
RULE_BASELINE_PATH = os.path.join(PROJECT_ROOT, "eval", "yang_results_rule_only.json")
INVALID_NOTEBOOK   = "2021-09-09-nb_334.ipynb"
OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "yang_scored_evaluation_v2.json")


# ---------- helpers ----------

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()


def load_dev_ids() -> set:
    if not os.path.exists(DEV_IDS_PATH):
        return set()
    with open(DEV_IDS_PATH, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def nb_path_to_fname(nb_path: str) -> str:
    parts = nb_path.split("/")
    if len(parts) >= 3:
        return f"{parts[-2]}-{parts[-1].replace('.py', '.ipynb')}"
    return nb_path


def get_ground_truth() -> Tuple[Dict[str, int], Dict[str, str]]:
    """
    Returns:
        labels: {fname -> 0 or 1} — only rows where 'pre' is Y* or N*
        unlabeled: {fname -> raw_pre_value} — rows with empty/unclassifiable 'pre'
    """
    labels = {}
    unlabeled = {}
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx == 0 and "nb" in str(row[0]):
                continue
            if len(row) < 3:
                continue
            fname = nb_path_to_fname(row[0])
            pre_val = row[2].strip().upper()
            if pre_val.startswith("Y"):
                labels[fname] = 1
            elif pre_val.startswith("N"):
                labels[fname] = 0
            else:
                unlabeled[fname] = row[2].strip()  # empty or non-standard
    return labels, unlabeled


def load_rule_baseline() -> Dict[str, int]:
    if not os.path.exists(RULE_BASELINE_PATH):
        return {}
    with open(RULE_BASELINE_PATH, "r", encoding="utf-8") as f:
        r = json.load(f)
    out = {}
    for item in r.get("per_notebook", []):
        fname = nb_path_to_fname(item["nb_id"])
        out[fname] = 1 if item.get("pre_pred") else 0
    return out


# ---------- Wilson score confidence intervals ----------

def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Dict[str, Optional[float]]:
    """
    Computes Wilson score confidence interval for a binomial proportion.
    Returns dict with lower, upper, and center bounds, or None if total == 0.
    """
    if total <= 0:
        return {"lower": None, "upper": None, "center": None}
    z = 1.959963984540054  # 95% two-sided normal quantile
    p = successes / total
    denom = 1.0 + (z ** 2) / total
    center = (p + (z ** 2) / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / total) + ((z ** 2) / (4.0 * (total ** 2))))
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return {
        "lower": round(float(lower), 4),
        "upper": round(float(upper), 4),
        "center": round(float(center), 4)
    }


# ---------- trivial baselines ----------

def _always_pred(y_true: List[int], label: int) -> Dict:
    y_pred = [label] * len(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tp = int(cm[1, 1])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tn = int(cm[0, 0])
    total = len(y_true)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy":  float((tp + tn) / total) if total > 0 else 0.0,
        "support":   total,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "wilson_ci": {
            "precision": wilson_score_interval(tp, tp + fp),
            "recall":    wilson_score_interval(tp, tp + fn),
            "accuracy":  wilson_score_interval(tp + tn, total),
        }
    }


_LEAKAGE_REGEX = re.compile(
    r"(?:\.fit(?:_transform)?\s*\([^)]*\).*?train_test_split|"
    r"fit_transform\s*\(\s*(?:X|df|data)\s*\))",
    re.DOTALL
)

def _regex_heuristic_pred(source: str) -> int:
    """
    Simple regex: flag leakage if a .fit or .fit_transform call appears
    and train_test_split(...) call is absent or appears after the fit call.
    """
    has_fit = bool(re.search(r"\.fit(?:_transform)?\s*\(", source))
    split_call_matches = [m.start() for m in re.finditer(r"train_test_split\s*\(", source)]
    if not has_fit:
        return 0
    if not split_call_matches:
        return 1  # fit with no split call at all — suspicious
    # Check if any fit occurs before the first split call
    fit_pos = [m.start() for m in re.finditer(r"\.fit(?:_transform)?\s*\(", source)]
    if fit_pos and min(fit_pos) < min(split_call_matches):
        return 1
    return 0


# ---------- metrics ----------

def _compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tp = int(cm[1, 1])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tn = int(cm[0, 0])
    total = len(y_true)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy":  float((tp + tn) / total) if total > 0 else 0.0,
        "support":   total,
        "positive_cases": int(sum(y_true)),
        "negative_cases": int(len(y_true) - sum(y_true)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "wilson_ci": {
            "precision": wilson_score_interval(tp, tp + fp),
            "recall":    wilson_score_interval(tp, tp + fn),
            "accuracy":  wilson_score_interval(tp + tn, total),
        }
    }


def _conservative_metrics(y_true_processed: List[int], y_pred_processed: List[int],
                           n_abstained_positives: int) -> Dict:
    """
    Conservative accounting: treat each abstained case as a missed positive (FN).
    y_true_processed and y_pred_processed are only for the processed subset.
    """
    cm = confusion_matrix(y_true_processed, y_pred_processed, labels=[0, 1])
    tp = int(cm[1, 1])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0]) + n_abstained_positives
    tn = int(cm[0, 0])
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    return {
        "convention": "abstained_positives_counted_as_fn",
        "n_abstained_positives_added_to_fn": n_abstained_positives,
        "adjusted_tp": tp, "adjusted_fp": fp, "adjusted_fn": fn, "adjusted_tn": tn,
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "accuracy":  float(accuracy),
        "support":   total,
        "wilson_ci": {
            "precision": wilson_score_interval(tp, tp + fp),
            "recall":    wilson_score_interval(tp, tp + fn),
            "accuracy":  wilson_score_interval(tp + tn, total),
        }
    }


# ---------- main evaluation ----------

def run_evaluation(model_path: str = CHECKPOINT_PATH) -> Dict[str, Any]:
    if not os.path.exists(GROUND_TRUTH_PATH):
        return {"status": "BLOCKED", "reason": "ground-truth.csv not found"}

    # --- Freeze evaluation configuration ---
    config = {
        "checkpoint_path": model_path,
        "checkpoint_sha256": _sha256_file(model_path) if os.path.exists(model_path) else None,
        "ground_truth_path": GROUND_TRUTH_PATH,
        "ground_truth_sha256": _sha256_file(GROUND_TRUTH_PATH),
        "adapter_version": ADAPTER_VERSION,
        "threshold": "default (argmax from RandomForestClassifier.predict)",
        "invalid_notebook_excluded": INVALID_NOTEBOOK,
    }

    gt_labels, unlabeled_gt = get_ground_truth()
    dev_ids_raw = load_dev_ids()
    dev_fnames  = {nb_path_to_fname(x) for x in dev_ids_raw} | dev_ids_raw
    model       = joblib.load(model_path)
    rule_preds  = load_rule_baseline()
    adapter     = NotebookAdapter(strip_magics=True)

    manifest = {
        "included": [],
        "excluded_invalid_json": [],
        "excluded_dev_id": [],
        "unlabeled_in_gt": [],    # present in GT file but 'pre' is empty/unclassifiable
        "abstained": [],          # eligible but adapter/prediction failed
    }

    results:  Dict[str, Dict] = {}   # fname -> {pred, true, adapter_meta}
    abstained: Dict[str, str]  = {}  # fname -> reason

    for fname in sorted(os.listdir(YANG_NOTEBOOKS_DIR)):
        if fname == INVALID_NOTEBOOK:
            manifest["excluded_invalid_json"].append({"id": fname, "reason": "invalid JSON per spec"})
            continue
        if fname in dev_fnames:
            manifest["excluded_dev_id"].append({"id": fname, "reason": "fixed development ID"})
            continue
        if fname in unlabeled_gt:
            manifest["unlabeled_in_gt"].append({"id": fname, "raw_pre": unlabeled_gt[fname]})
            continue
        if fname not in gt_labels:
            # Should not occur given audit, but guard anyway
            manifest["unlabeled_in_gt"].append({"id": fname, "raw_pre": "missing_from_csv"})
            continue

        fpath = os.path.join(YANG_NOTEBOOKS_DIR, fname)
        try:
            source, meta = adapter.extract_from_file(fpath)
            if not source.strip():
                abstained[fname] = "empty_after_extraction"
                manifest["abstained"].append({"id": fname, "reason": "empty_after_extraction", "meta": meta})
                continue
            pred = int(model.predict([source])[0])
            regex_pred = _regex_heuristic_pred(source)
            results[fname] = {
                "pred": pred,
                "true": gt_labels[fname],
                "regex_pred": regex_pred,
                "adapter_meta": meta,
            }
            manifest["included"].append(fname)
        except NotebookParseError as exc:
            abstained[fname] = f"parse_error: {exc}"
            manifest["abstained"].append({"id": fname, "reason": str(exc)})
        except Exception as exc:
            abstained[fname] = f"unexpected: {exc}"
            manifest["abstained"].append({"id": fname, "reason": str(exc)})

    included = manifest["included"]
    y_true  = [results[f]["true"] for f in included]
    y_pred  = [results[f]["pred"] for f in included]
    y_regex = [results[f]["regex_pred"] for f in included]

    # Trivial baselines
    always0_metrics = _always_pred(y_true, 0)
    always1_metrics = _always_pred(y_true, 1)
    regex_metrics   = _compute_metrics(y_true, y_regex)
    learned_metrics = _compute_metrics(y_true, y_pred)

    # Abstention accounting
    n_abstained = len(abstained)
    n_abstained_pos = sum(
        1 for f in abstained if f in gt_labels and gt_labels[f] == 1
    )
    conservative = _conservative_metrics(y_true, y_pred, n_abstained_pos)

    # Rule baseline on same included set
    rule_metrics = None
    common = [f for f in included if f in rule_preds]
    if common:
        yt_r = [results[f]["true"] for f in common]
        yp_r = [rule_preds[f] for f in common]
        rule_metrics = _compute_metrics(yt_r, yp_r)
        rule_metrics["common_denominator_count"] = len(common)

    # Coverage
    n_eligible = len(included) + n_abstained
    coverage_pct = 100.0 * len(included) / n_eligible if n_eligible > 0 else 0.0

    report = {
        "status": "EVALUATED",
        "config": config,
        "eligible_sample_count": n_eligible,
        "processed_count": len(included),
        "abstained_count": n_abstained,
        "abstained_positive_count": n_abstained_pos,
        "coverage_pct": round(coverage_pct, 2),
        "coverage_wilson_ci": wilson_score_interval(len(included), n_eligible),
        "positive_cases_in_processed": int(sum(y_true)),
        "negative_cases_in_processed": int(len(y_true) - sum(y_true)),
        "unlabeled_in_gt_count": len(manifest["unlabeled_in_gt"]),
        "baselines": {
            "always_negative": always0_metrics,
            "always_positive": always1_metrics,
            "regex_heuristic": regex_metrics,
        },
        "learned_rf_metrics": learned_metrics,
        "learned_rf_conservative": conservative,
        "rule_only_metrics": rule_metrics,
        "rule_comparison_status": "COMPLETE" if rule_metrics else "PENDING",
        "manifest": manifest,
    }
    return report


if __name__ == "__main__":
    report = run_evaluation()
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved to {OUT_PATH}\n")
    print(f"Eligible: {report['eligible_sample_count']}  "
          f"Processed: {report['processed_count']}  "
          f"Abstained: {report['abstained_count']}  "
          f"Coverage: {report['coverage_pct']}%")
    print(f"GT positives (leakage=1) in processed: {report['positive_cases_in_processed']}")
    print(f"GT negatives (leakage=0) in processed: {report['negative_cases_in_processed']}")
    print(f"Unlabeled in GT (empty 'pre' column): {report['unlabeled_in_gt_count']}")

    def _fmt(name, m):
        if m is None:
            return
        print(f"\n  {name}: P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f}  support={m['support']}"
              f"  CM={m['confusion_matrix']}")

    print("\n=== Baselines ===")
    _fmt("Always-0  ", report["baselines"]["always_negative"])
    _fmt("Always-1  ", report["baselines"]["always_positive"])
    _fmt("Regex     ", report["baselines"]["regex_heuristic"])
    print("\n=== Learned RF ===")
    _fmt("Learned RF", report["learned_rf_metrics"])
    c = report["learned_rf_conservative"]
    print(f"  Conservative (abstained pos as FN): P={c['precision']:.3f} "
          f"R={c['recall']:.3f} F1={c['f1']:.3f}  support={c['support']}")
    if report["rule_only_metrics"]:
        print("\n=== Rule-Only Baseline ===")
        _fmt("Rule-Only ", report["rule_only_metrics"])
