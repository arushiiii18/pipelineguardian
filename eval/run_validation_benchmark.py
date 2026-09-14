"""
run_validation_benchmark.py — Executes validation_strategy_reviewer against ground_truth.json
"""

import json
import os
from dotenv import load_dotenv
load_dotenv()

from pipelineguardian.tools.schema_inspector import SchemaSignals
from pipelineguardian.tools.notebook_parser import ParsedSource
from pipelineguardian.tools.validation_strategy_reviewer import review

def main():
    gt_path = os.path.join(os.path.dirname(__file__), "validation_benchmark", "ground_truth.json")
    with open(gt_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    print(f"Running validation strategy benchmark on {len(benchmark['test_cases'])} cases...\n")
    results = []
    tp = fp = tn = fn = 0

    for tc in benchmark["test_cases"]:
        pset = ParsedSource(source=tc["code"], path=tc["id"], is_notebook=False, line_to_cell={})
        sig = SchemaSignals(
            timestamp_columns=tc["signals"]["timestamp_columns"],
            group_columns=tc["signals"]["group_columns"],
            imbalance=tc["signals"]["imbalance"]
        )
        expected_app = tc["expected_is_appropriate"]
        
        if not sig.has_trigger:
            pred_app = True
            reasoning = "Skipped by schema_inspector gate (no signals)"
            abstained = True
        else:
            abstained = False
            try:
                issues = review(pset, sig)
                pred_app = (len(issues) == 0)
                reasoning = issues[0].message if issues else "LLM judged strategy appropriate."
            except Exception as e:
                pred_app = True
                reasoning = f"FAILED: {e}"
                abstained = True

        is_correct = (pred_app == expected_app)

        gt_issue = not expected_app
        pred_issue = not pred_app

        if gt_issue and pred_issue:
            tp += 1
        elif not gt_issue and pred_issue:
            fp += 1
        elif gt_issue and not pred_issue:
            fn += 1
        else:
            tn += 1

        results.append({
            "id": tc["id"],
            "signal_type": tc["signal_type"],
            "expected_appropriate": expected_app,
            "predicted_appropriate": pred_app,
            "correct": is_correct,
            "abstained_gate": abstained,
            "reasoning": reasoning
        })

        status_str = "PASS" if is_correct else "FAIL"
        print(f"[{status_str}] {tc['id']} (signal: {tc['signal_type']})")
        print(f"       Expected Appropriate: {expected_app} | Predicted Appropriate: {pred_app}")
        print(f"       Reasoning: {reasoning}\n")

    accuracy = sum(r["correct"] for r in results) / len(results)
    print("=" * 70)
    print(f"VALIDATION STRATEGY BENCHMARK SUMMARY")
    print(f"Accuracy: {sum(r['correct'] for r in results)} / {len(results)} = {accuracy:.4f}")
    print(f"Confusion Matrix (Target = Strategy Mismatch / Inappropriate):")
    print(f"  TP={tp} (Correctly caught mismatch)")
    print(f"  FP={fp} (False alarm on appropriate strategy)")
    print(f"  FN={fn} (Missed actual mismatch)")
    print(f"  TN={tn} (Correctly recognized appropriate strategy)")
    print("=" * 70)

if __name__ == "__main__":
    main()
