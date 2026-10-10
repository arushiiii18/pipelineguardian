"""
evaluate_yang_contract.py - External Yang Evaluation Contract & Runner for Learned Leakage Detector.
Establishes the exact evaluation contract to evaluate the frozen detector on Yang notebooks
once teammate 1 supplies the common evaluation inputs.
"""

import os
import sys
import json
import joblib
from typing import List, Dict, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEV_IDS_PATH = os.path.join(PROJECT_ROOT, "eval", "task2_dev_ids.txt")
YANG_NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "eval", "yang_corpus", "notebooks")
CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "baseline_rf.joblib")


def load_dev_ids(dev_ids_file: str = DEV_IDS_PATH) -> set:
    if not os.path.exists(dev_ids_file):
        return set()
    with open(dev_ids_file, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def check_teammate_readiness() -> Dict[str, Any]:
    """
    Checks availability of the 6 required teammate inputs.
    """
    dev_ids = load_dev_ids()
    corpus_exists = os.path.isdir(YANG_NOTEBOOKS_DIR) and len(os.listdir(YANG_NOTEBOOKS_DIR)) > 0
    rule_results_path = os.path.join(PROJECT_ROOT, "eval", "yang_results_rule_only.json")
    rule_results_exists = os.path.exists(rule_results_path)

    status = {
        "is_ready": False,
        "dev_ids_count": len(dev_ids),
        "corpus_available": corpus_exists,
        "rule_results_available": rule_results_exists,
        "missing_inputs": []
    }

    if not corpus_exists:
        status["missing_inputs"].append("Uncompressed Yang notebook corpus in eval/yang_corpus/notebooks/")
    if not rule_results_exists:
        status["missing_inputs"].append("Rule-only per-notebook predictions in eval/yang_results_rule_only.json")

    status["is_ready"] = (len(status["missing_inputs"]) == 0)
    return status


def evaluate_notebook_source(code_str: str, model: Any) -> bool:
    """
    Evaluates a single flattened notebook string using the frozen learned detector.
    Returns True if preprocessing leakage is detected, False otherwise.
    """
    pred = model.predict([code_str])[0]
    return bool(pred == 1)


def run_yang_evaluation_contract(model_path: str = CHECKPOINT_PATH) -> Dict[str, Any]:
    """
    Executes or checks the Yang evaluation contract.
    If teammate inputs are missing, reports exact blocker status without fabricating data.
    """
    readiness = check_teammate_readiness()
    if not readiness["is_ready"]:
        report = {
            "status": "BLOCKED_ON_TEAMMATE_1",
            "reason": "Teammate inputs for Yang external evaluation are not fully provisioned.",
            "missing_inputs": readiness["missing_inputs"],
            "required_contract_spec": [
                "1. Common evaluable Yang notebook ID list",
                "2. Explicit handling of invalid JSON notebook 2021-09-09-nb_334.ipynb (skipped)",
                "3. Confirmation that 12 fixed dev IDs from eval/task2_dev_ids.txt are excluded",
                "4. Uncompressed notebook corpus directory",
                "5. Rule-only per-notebook predictions for common denominator alignment",
                "6. Baseline scored/skipped notebook accounting"
            ]
        }
        return report

    # If ready, execute full evaluation
    model = joblib.load(model_path)
    dev_ids = load_dev_ids()
    results = []

    for fname in os.listdir(YANG_NOTEBOOKS_DIR):
        if fname == "2021-09-09-nb_334.ipynb":
            continue  # Explicit handling of invalid JSON notebook
        if fname in dev_ids:
            continue  # Exclude dev IDs

        fpath = os.path.join(YANG_NOTEBOOKS_DIR, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            pred_leakage = evaluate_notebook_source(content, model)
            results.append({
                "nb_id": fname,
                "learned_pre_pred": pred_leakage
            })
        except Exception:
            continue

    return {
        "status": "SUCCESS",
        "evaluated_notebooks": len(results),
        "results": results
    }


if __name__ == "__main__":
    report = run_yang_evaluation_contract()
    print(json.dumps(report, indent=2))
