"""
compute_agreement.py — Script to compute inter-rater or self-relabeling agreement.

Takes two ground-truth JSON files (annotator_1.json and annotator_2.json)
and computes percent agreement and Cohen's Kappa over labeled categories.

Usage:
    python -m eval.compute_agreement --ann1 path/to/ann1.json --ann2 path/to/ann2.json
"""

from __future__ import annotations

import argparse
import json
import os


def compute_kappa(po: float, pe: float) -> float:
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def _extract_label_map(data: dict) -> dict[str, dict]:
    if "notebooks" in data and isinstance(data["notebooks"], dict):
        res = {}
        for k, v in data["notebooks"].items():
            res[k] = v.get("labels", {})
        return res
    if "test_cases" in data and isinstance(data["test_cases"], list):
        res = {}
        for item in data["test_cases"]:
            item_id = item.get("id")
            if not item_id:
                continue
            if "labels" in item:
                res[item_id] = item["labels"]
            elif "sub_check" in item:
                res[item_id] = {item["sub_check"]: item.get("expected_issue")}
        return res
    return {}


def evaluate_agreement(ann1_path: str, ann2_path: str) -> dict:
    with open(ann1_path, "r", encoding="utf-8") as f:
        data1 = json.load(f)
    with open(ann2_path, "r", encoding="utf-8") as f:
        data2 = json.load(f)

    nbs1 = _extract_label_map(data1)
    nbs2 = _extract_label_map(data2)

    common_ids = sorted(set(nbs1.keys()) & set(nbs2.keys()))
    if not common_ids:
        return {"error": "No common notebook IDs found between the two files."}

    total_items = 0
    agreements = 0

    p1_pos = 0
    p2_pos = 0

    subcheck_stats = {}

    for nb_id in common_ids:
        l1 = nbs1[nb_id]
        l2 = nbs2[nb_id]

        for check_name, val1 in l1.items():
            if val1 is None:
                continue
            val2 = l2.get(check_name)
            if val2 is None:
                continue

            total_items += 1
            if val1 == val2:
                agreements += 1

            if val1:
                p1_pos += 1
            if val2:
                p2_pos += 1

            if check_name not in subcheck_stats:
                subcheck_stats[check_name] = {"total": 0, "agree": 0}
            subcheck_stats[check_name]["total"] += 1
            if val1 == val2:
                subcheck_stats[check_name]["agree"] += 1

    if total_items == 0:
        return {"error": "No overlapping non-null labels found to compute agreement."}

    po = agreements / total_items

    # Chance agreement
    p1 = p1_pos / total_items
    q1 = 1.0 - p1
    p2 = p2_pos / total_items
    q2 = 1.0 - p2

    pe = (p1 * p2) + (q1 * q2)
    kappa = compute_kappa(po, pe)

    per_subcheck = {}
    for sc, stats in subcheck_stats.items():
        per_subcheck[sc] = {
            "percent_agreement": round(stats["agree"] / stats["total"], 4) if stats["total"] else 0.0,
            "count": stats["total"]
        }

    return {
        "notebooks_evaluated": len(common_ids),
        "total_label_pairs": total_items,
        "percent_agreement": round(po, 4),
        "cohens_kappa": round(kappa, 4),
        "per_subcheck": per_subcheck
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute agreement between two annotation sets")
    parser.add_argument("--ann1", required=True, help="Path to Annotator 1 JSON file")
    parser.add_argument("--ann2", required=True, help="Path to Annotator 2 JSON file")
    args = parser.parse_args()

    results = evaluate_agreement(args.ann1, args.ann2)
    print(json.dumps(results, indent=2))
