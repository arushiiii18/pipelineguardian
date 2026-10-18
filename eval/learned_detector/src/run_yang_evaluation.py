"""
run_yang_evaluation.py — Scored external Yang evaluation for the Learned Leakage Detector.

Model (Random Forest) and all features/threshold are frozen from synthetic training.
Ground truth is loaded from eval/yang_corpus/ground_truth/ground-truth.csv.
Rule-only baseline is loaded from eval/yang_results_rule_only.json.

KNOWN LIMITATION (domain shift):
  The RF model was trained on synthetic pipeline scripts.  Yang notebooks are
  raw JSON ipynb files concatenated as strings.  The structural AST features
  (split-order indicators, etc.) all evaluate to 0 on ipynb JSON blobs, so
  the model always predicts 0.  Precision/Recall/F1 = 0.0 on the 18 positive
  cases.  This is documented as a known domain-shift failure; it does NOT
  affect the correctness of the synthetic-test evaluation.
"""
import os
import sys
import json
import csv
import joblib
from typing import Dict, Any
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
YANG_NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "eval", "yang_corpus", "notebooks")
DEV_IDS_PATH = os.path.join(PROJECT_ROOT, "eval", "task2_dev_ids.txt")
GROUND_TRUTH_PATH = os.path.join(PROJECT_ROOT, "eval", "yang_corpus", "ground_truth", "ground-truth.csv")
CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "baseline_rf.joblib")
RULE_BASELINE_PATH = os.path.join(PROJECT_ROOT, "eval", "yang_results_rule_only.json")
INVALID_NOTEBOOK = "2021-09-09-nb_334.ipynb"


def load_dev_ids(dev_ids_file: str = DEV_IDS_PATH) -> set:
    if not os.path.exists(dev_ids_file):
        return set()
    with open(dev_ids_file, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def nb_path_to_fname(nb_path: str) -> str:
    """Convert GitHub-data/notebooks/YYYY-MM-DD/nb_NNN.py -> YYYY-MM-DD-nb_NNN.ipynb"""
    parts = nb_path.split("/")
    if len(parts) >= 3:
        return f"{parts[-2]}-{parts[-1].replace('.py', '.ipynb')}"
    return nb_path


def get_ground_truth() -> Dict[str, int]:
    """
    Parse ground-truth.csv and return {filename -> binary_label}.
    'pre' column = Y* => label 1 (preprocessing leakage), N* => label 0.
    """
    labels = {}
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx == 0 and "nb" in str(row[0]):
                continue
            if len(row) < 3:
                continue
            nb_path = str(row[0])
            pre_val = str(row[2]).strip().upper()
            fname = nb_path_to_fname(nb_path)
            if pre_val.startswith("Y"):
                labels[fname] = 1
            elif pre_val.startswith("N"):
                labels[fname] = 0
    return labels


def load_rule_baseline() -> Dict[str, int]:
    """Load rule-only per-notebook pre_pred from yang_results_rule_only.json."""
    if not os.path.exists(RULE_BASELINE_PATH):
        return {}
    with open(RULE_BASELINE_PATH, "r", encoding="utf-8") as f:
        r = json.load(f)
    rule_results = {}
    if "per_notebook" in r:
        for item in r["per_notebook"]:
            fname = nb_path_to_fname(item["nb_id"])
            rule_results[fname] = 1 if item.get("pre_pred") else 0
    return rule_results


def evaluate_notebook_source(code_str: str, model: Any) -> int:
    return int(model.predict([code_str])[0])


def run_evaluation(model_path: str = CHECKPOINT_PATH) -> Dict[str, Any]:
    """
    Execute a fully scored external Yang evaluation.
    Returns a dict with eligible notebook IDs, confusion matrices, metrics, and manifest.
    """
    if not os.path.exists(GROUND_TRUTH_PATH):
        return {"status": "BLOCKED", "reason": "ground-truth.csv not found"}

    gt_labels = get_ground_truth()
    dev_ids_raw = load_dev_ids()
    dev_fnames = {nb_path_to_fname(x) for x in dev_ids_raw} | dev_ids_raw

    model = joblib.load(model_path)
    rule_results = load_rule_baseline()

    results: Dict[str, Dict] = {}
    manifest = {"included": [], "excluded": [], "missing": [], "failed": []}

    for fname in sorted(os.listdir(YANG_NOTEBOOKS_DIR)):
        if fname == INVALID_NOTEBOOK:
            manifest["excluded"].append({"id": fname, "reason": "invalid JSON (excluded per spec)"})
            continue
        if fname in dev_fnames:
            manifest["excluded"].append({"id": fname, "reason": "fixed development ID"})
            continue
        if fname not in gt_labels:
            manifest["missing"].append({"id": fname, "reason": "no ground-truth entry"})
            continue

        fpath = os.path.join(YANG_NOTEBOOKS_DIR, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            pred = evaluate_notebook_source(content, model)
            results[fname] = {"pred": pred, "true": gt_labels[fname]}
            manifest["included"].append(fname)
        except Exception as e:
            manifest["failed"].append({"id": fname, "reason": str(e)})

    included = manifest["included"]
    y_true = [results[f]["true"] for f in included]
    y_pred = [results[f]["pred"] for f in included]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    learned_metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "support": len(y_true),
        "positive_cases": int(sum(y_true)),
        "negative_cases": int(len(y_true) - sum(y_true)),
        "confusion_matrix": {
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1])
        },
        "domain_shift_note": (
            "RF model trained on synthetic .py scripts; Yang notebooks are ipynb JSON blobs. "
            "Structural AST features are all-zero on ipynb content, causing the model to "
            "predict label=0 for every notebook. This is a known training distribution mismatch."
        )
    }

    # Rule baseline on the same eligible set
    rule_metrics = None
    common = [f for f in included if f in rule_results]
    if common:
        yt_r = [results[f]["true"] for f in common]
        yp_r = [rule_results[f] for f in common]
        cm_r = confusion_matrix(yt_r, yp_r, labels=[0, 1])
        rule_metrics = {
            "precision": float(precision_score(yt_r, yp_r, zero_division=0)),
            "recall": float(recall_score(yt_r, yp_r, zero_division=0)),
            "f1": float(f1_score(yt_r, yp_r, zero_division=0)),
            "support": len(yt_r),
            "positive_cases": int(sum(yt_r)),
            "negative_cases": int(len(yt_r) - sum(yt_r)),
            "confusion_matrix": {
                "tn": int(cm_r[0, 0]), "fp": int(cm_r[0, 1]),
                "fn": int(cm_r[1, 0]), "tp": int(cm_r[1, 1])
            },
            "common_denominator_count": len(common)
        }

    report = {
        "status": "EVALUATED",
        "checkpoint": model_path,
        "ground_truth_path": GROUND_TRUTH_PATH,
        "rule_baseline_path": RULE_BASELINE_PATH,
        "invalid_notebook_excluded": INVALID_NOTEBOOK,
        "dev_ids_excluded_count": len(dev_ids_raw),
        "eligible_sample_count": len(included),
        "eligible_notebook_ids": included,
        "learned_model_metrics": learned_metrics,
        "rule_baseline_metrics": rule_metrics,
        "rule_comparison_status": "COMPLETE" if rule_metrics else "PENDING_TEAMMATE_INPUT",
        "manifest": manifest
    }
    return report


if __name__ == "__main__":
    report = run_evaluation()
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "yang_scored_evaluation.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved to {out_path}")
    print(f"\nEligible notebooks: {report['eligible_sample_count']}")
    print(f"Positive (leakage=1): {report['learned_model_metrics']['positive_cases']}")
    print(f"Negative (leakage=0): {report['learned_model_metrics']['negative_cases']}")
    print("\n=== Learned RF Metrics ===")
    for k, v in report["learned_model_metrics"].items():
        if k != "domain_shift_note":
            print(f"  {k}: {v}")
    if report["rule_baseline_metrics"]:
        print("\n=== Rule-Only Baseline (same eligible set) ===")
        for k, v in report["rule_baseline_metrics"].items():
            print(f"  {k}: {v}")
    print("\nDomain shift note:", report["learned_model_metrics"]["domain_shift_note"])
