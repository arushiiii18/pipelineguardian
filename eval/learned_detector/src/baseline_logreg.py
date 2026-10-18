"""
baseline_logreg.py - Baseline A: TF-IDF + Logistic Regression.
Trains, tunes, and evaluates source-code TF-IDF + Logistic Regression using GroupKFold by base_id.
Evaluates on held-out test base pipelines and leave-one-mutation-type-out stress test.
"""

import os
import sys
import json
import time
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

# Ensure src in sys.path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_loader import get_train_val_and_test_splits
from metrics import evaluate_predictions

TOKEN_PATTERN = r"(?u)\b[A-Za-z_]\w*\b"


def run_grouped_cv(texts, labels, groups, config, n_splits=5):
    """
    Evaluates a specific pipeline configuration across GroupKFold splits.
    Guarantees no variants of any base_id appear across train and val folds.
    """
    gkf = GroupKFold(n_splits=n_splits)
    oof_preds = np.zeros(len(labels), dtype=int)
    oof_probs = np.zeros(len(labels), dtype=float)
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(texts, labels, groups)):
        tr_texts = [texts[i] for i in train_idx]
        tr_labels = labels[train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = labels[val_idx]

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                token_pattern=TOKEN_PATTERN,
                ngram_range=config["ngram_range"],
                max_features=config["max_features"],
                sublinear_tf=True
            )),
            ("clf", LogisticRegression(
                C=config["C"],
                class_weight=config["class_weight"],
                solver="liblinear",
                random_state=42
            ))
        ])

        pipe.fit(tr_texts, tr_labels)
        preds = pipe.predict(val_texts)
        probs = pipe.predict_proba(val_texts)[:, 1]

        oof_preds[val_idx] = preds
        oof_probs[val_idx] = probs

        val_eval = evaluate_predictions(val_labels, preds, probs)
        fold_f1s.append(val_eval["macro_f1"])

    oof_eval = evaluate_predictions(labels, oof_preds, oof_probs)
    oof_eval["fold_macro_f1s"] = fold_f1s
    oof_eval["fold_f1_std"] = float(np.std(fold_f1s))
    return oof_eval, oof_preds, oof_probs


def tune_logreg(train_val_data):
    texts = train_val_data["texts"]
    labels = train_val_data["labels"]
    groups = train_val_data["groups"]

    param_grid = [
        {"ngram_range": (1, 1), "max_features": 1000, "C": 0.1, "class_weight": None},
        {"ngram_range": (1, 1), "max_features": 1000, "C": 1.0, "class_weight": None},
        {"ngram_range": (1, 2), "max_features": 1000, "C": 1.0, "class_weight": None},
        {"ngram_range": (1, 2), "max_features": 2000, "C": 1.0, "class_weight": None},
        {"ngram_range": (1, 2), "max_features": 2000, "C": 5.0, "class_weight": "balanced"},
        {"ngram_range": (1, 2), "max_features": 2000, "C": 10.0, "class_weight": None},
    ]

    tuning_records = []
    best_config = None
    best_macro_f1 = -1.0
    best_oof_eval = None

    print(f"Starting TF-IDF + LogReg tuning across {len(param_grid)} configurations...")
    for idx, cfg in enumerate(param_grid):
        start_time = time.time()
        oof_eval, _, _ = run_grouped_cv(texts, labels, groups, cfg, n_splits=5)
        duration = time.time() - start_time

        record = {
            "trial_id": idx + 1,
            "config": cfg,
            "macro_f1": oof_eval["macro_f1"],
            "fold_std": oof_eval["fold_f1_std"],
            "positive_f1": oof_eval["positive_class"]["f1"],
            "negative_f1": oof_eval["negative_class"]["f1"],
            "pr_auc": oof_eval.get("pr_auc"),
            "duration_sec": duration
        }
        tuning_records.append(record)
        print(f"Trial {idx+1}: cfg={cfg} -> Macro-F1={oof_eval['macro_f1']:.4f} (std={oof_eval['fold_f1_std']:.4f}) in {duration:.2f}s")

        if oof_eval["macro_f1"] > best_macro_f1:
            best_macro_f1 = oof_eval["macro_f1"]
            best_config = cfg
            best_oof_eval = oof_eval

    return best_config, best_oof_eval, tuning_records


def run_stress_test(train_val_data, best_config):
    """
    Leave-one-mutation-type-out evaluation:
    Holds out one mutation type completely from training, assesses if model can detect it unseen.
    """
    texts = train_val_data["texts"]
    labels = train_val_data["labels"]
    records = train_val_data["records"]

    mutation_types = sorted(list({r["mutation_type"] for r in records}))
    stress_results = {}

    for mt in mutation_types:
        train_idx = [i for i, r in enumerate(records) if r["mutation_type"] != mt]
        test_idx = [i for i, r in enumerate(records) if r["mutation_type"] == mt]

        # Check if test set contains samples
        if not test_idx or len(set(labels[train_idx])) < 2:
            continue

        tr_texts = [texts[i] for i in train_idx]
        tr_labels = labels[train_idx]
        te_texts = [texts[i] for i in test_idx]
        te_labels = labels[test_idx]

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                token_pattern=TOKEN_PATTERN,
                ngram_range=best_config["ngram_range"],
                max_features=best_config["max_features"],
                sublinear_tf=True
            )),
            ("clf", LogisticRegression(
                C=best_config["C"],
                class_weight=best_config["class_weight"],
                solver="liblinear",
                random_state=42
            ))
        ])

        pipe.fit(tr_texts, tr_labels)
        preds = pipe.predict(te_texts)
        acc = float(np.mean(preds == te_labels))
        stress_results[mt] = {
            "held_out_samples": len(test_idx),
            "expected_label": int(te_labels[0]),
            "accuracy": acc
        }

    return stress_results


def main():
    splits = get_train_val_and_test_splits()
    train_val = splits["train_val"]
    held_out = splits["held_out_test"]

    print(f"Train/Val samples: {len(train_val['texts'])} (Bases: {len(set(train_val['groups']))})")
    print(f"Held-out test samples: {len(held_out['texts'])} (Bases: {len(set(held_out['groups']))})")

    # 1. Tuning & Grouped Cross-Validation
    best_config, best_oof_eval, tuning_history = tune_logreg(train_val)
    print(f"\nBest Config: {best_config} with Grouped CV Macro-F1 = {best_oof_eval['macro_f1']:.4f}")

    # 2. Fit frozen model on entire train_val pool
    start_fit = time.time()
    final_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            token_pattern=TOKEN_PATTERN,
            ngram_range=best_config["ngram_range"],
            max_features=best_config["max_features"],
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            C=best_config["C"],
            class_weight=best_config["class_weight"],
            solver="liblinear",
            random_state=42
        ))
    ])
    final_pipe.fit(train_val["texts"], train_val["labels"])
    fit_duration = time.time() - start_fit

    # 3. Final Synthetic Held-Out Test Evaluation
    start_infer = time.time()
    held_out_preds = final_pipe.predict(held_out["texts"])
    held_out_probs = final_pipe.predict_proba(held_out["texts"])[:, 1]
    infer_duration = time.time() - start_infer

    held_out_eval = evaluate_predictions(
        held_out["labels"],
        held_out_preds,
        held_out_probs,
        records=held_out["records"]
    )
    print(f"\nSynthetic Held-out Test Macro-F1 = {held_out_eval['macro_f1']:.4f}")
    print(f"Held-out Confusion Matrix: {held_out_eval['confusion_matrix']}")

    # 4. Leave-one-mutation-type-out stress test
    stress_results = run_stress_test(train_val, best_config)

    # 5. Save Artifacts
    res_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    model_path = os.path.join(ckpt_dir, "baseline_logreg.joblib")
    joblib.dump(final_pipe, model_path)

    full_report = {
        "model_name": "TF-IDF + Logistic Regression",
        "best_config": best_config,
        "fit_time_sec": fit_duration,
        "infer_time_sec": infer_duration,
        "grouped_cv_eval": best_oof_eval,
        "held_out_test_eval": held_out_eval,
        "tuning_history": tuning_history,
        "stress_test_eval": stress_results
    }

    report_path = os.path.join(res_dir, "baseline_logreg_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"Saved artifacts to {report_path} and {model_path}")


if __name__ == "__main__":
    main()
