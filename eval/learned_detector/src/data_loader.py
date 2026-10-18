"""
data_loader.py - Dataset loading and Grouped CV splitting for Learned Leakage Detector.
Loads verified examples, ensures clean train_val vs held_out_test isolation by base_id.
"""

import os
import json
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.model_selection import GroupKFold

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")


def load_dataset(manifest_path: str = MANIFEST_PATH, only_verified: bool = True) -> Tuple[List[str], np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Loads code samples, binary labels (1=leakage, 0=clean), base_id groups, and raw metadata records.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data["records"]
    if only_verified:
        records = [r for r in records if r["verification_status"] in ["verified_clean", "verified_leakage"]]

    texts = []
    labels = []
    groups = []
    filtered_records = []

    for r in records:
        ref_path = os.path.join(DATA_DIR, r["source_reference"])
        if not os.path.exists(ref_path):
            continue
        with open(ref_path, "r", encoding="utf-8") as f:
            code = f.read()

        texts.append(code)
        labels.append(r["label"])
        groups.append(r["base_id"])
        filtered_records.append(r)

    return texts, np.array(labels, dtype=int), np.array(groups), filtered_records


def get_train_val_and_test_splits(manifest_path: str = MANIFEST_PATH) -> Dict[str, Any]:
    """
    Partitions dataset into:
    1. train_val pool (24 base pipelines)
    2. held_out_test set (6 held-out base pipelines: base_25 to base_30)
    """
    texts, labels, groups, records = load_dataset(manifest_path=manifest_path, only_verified=True)

    train_val_idx = [i for i, r in enumerate(records) if r["split"] == "train_val"]
    test_idx = [i for i, r in enumerate(records) if r["split"] == "test"]

    return {
        "train_val": {
            "texts": [texts[i] for i in train_val_idx],
            "labels": labels[train_val_idx],
            "groups": groups[train_val_idx],
            "records": [records[i] for i in train_val_idx]
        },
        "held_out_test": {
            "texts": [texts[i] for i in test_idx],
            "labels": labels[test_idx],
            "groups": groups[test_idx],
            "records": [records[i] for i in test_idx]
        }
    }
