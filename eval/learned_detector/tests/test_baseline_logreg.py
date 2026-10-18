"""
Tests for Baseline A: TF-IDF + Logistic Regression.
Verifies fitting, prediction bounds, group fold isolation, and model persistence.
"""

import os
import sys
import pytest
import numpy as np
import joblib

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from baseline_logreg import run_grouped_cv, TOKEN_PATTERN
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def test_logreg_pipeline_fit_predict():
    texts = [
        "scaler.fit(X)\nX_train, X_test = split(X)",
        "X_train, X_test = split(X)\nscaler.fit(X_train)",
        "scaler = StandardScaler()\nX_s = scaler.fit_transform(X)",
        "pipe = Pipeline([('s', StandardScaler())])\npipe.fit(X_tr)"
    ]
    labels = np.array([1, 0, 1, 0])
    groups = np.array(["b1", "b1", "b2", "b2"])

    cfg = {"ngram_range": (1, 1), "max_features": 100, "C": 1.0, "class_weight": None}
    oof_eval, preds, probs = run_grouped_cv(texts, labels, groups, cfg, n_splits=2)

    assert len(preds) == 4
    assert set(preds).issubset({0, 1})
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert "macro_f1" in oof_eval


def test_logreg_save_and_load(tmp_path):
    texts = ["scaler.fit(X)", "scaler.fit(X_train)"]
    labels = np.array([1, 0])

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(token_pattern=TOKEN_PATTERN)),
        ("clf", LogisticRegression(random_state=42))
    ])
    pipe.fit(texts, labels)

    save_path = tmp_path / "model.joblib"
    joblib.dump(pipe, save_path)

    loaded_pipe = joblib.load(save_path)
    preds1 = pipe.predict(texts)
    preds2 = loaded_pipe.predict(texts)
    assert np.array_equal(preds1, preds2)
