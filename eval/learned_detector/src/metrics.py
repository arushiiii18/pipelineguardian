"""
metrics.py - Metric computation routines for binary preprocessing leakage detection.
Computes Macro-F1, per-class Precision/Recall/F1, Confusion Matrix, and PR-AUC.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve, auc
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    records: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Evaluates predictions against ground truth binary labels.
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Per-class metrics
    pos_p = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    pos_r = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    pos_f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    pos_support = int(np.sum(y_true == 1))

    neg_p = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    neg_r = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    neg_f1 = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))
    neg_support = int(np.sum(y_true == 0))

    macro_f1 = float((pos_f1 + neg_f1) / 2.0)

    res = {
        "macro_f1": macro_f1,
        "positive_class": {
            "precision": pos_p,
            "recall": pos_r,
            "f1": pos_f1,
            "support": pos_support
        },
        "negative_class": {
            "precision": neg_p,
            "recall": neg_r,
            "f1": neg_f1,
            "support": neg_support
        },
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "total_samples": len(y_true)
    }

    if y_prob is not None:
        try:
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            res["pr_auc"] = float(auc(rec, prec))
        except Exception:
            res["pr_auc"] = None

    # Per-mutation breakdown if records supplied
    if records is not None and len(records) == len(y_true):
        per_mut = {}
        mut_types = sorted(list({r["mutation_type"] for r in records}))
        for mt in mut_types:
            idx = [i for i, r in enumerate(records) if r["mutation_type"] == mt]
            sub_true = y_true[idx]
            sub_pred = y_pred[idx]
            acc = float(np.mean(sub_true == sub_pred))
            per_mut[mt] = {
                "support": len(idx),
                "accuracy": acc,
                "expected_label": int(sub_true[0]) if len(sub_true) > 0 else None
            }
        res["per_mutation_type"] = per_mut

    return res
