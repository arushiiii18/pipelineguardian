"""
Tests for Baseline B: Random Forest Classifier and Feature Extraction.
Verifies feature extraction dimensions, model fitting, predict bounds, and persistence.
"""

import os
import sys
import pytest
import numpy as np
import joblib

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from features import extract_pipeline_features, FEATURE_NAMES
from baseline_rf import PipelineFeatureUnionModel, run_grouped_cv_rf


def test_feature_extraction_vector_dimensions():
    sample_code = """import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = np.random.randn(10, 2)
scaler = StandardScaler()
X_s = scaler.fit_transform(X)
X_tr, X_te = train_test_split(X_s, test_size=0.2)
"""
    vec = extract_pipeline_features(sample_code)
    assert len(vec) == len(FEATURE_NAMES)
    assert vec[13] == 1.0  # fit before split flag detected
    assert vec[10] >= 1.0  # StandardScaler detected


def test_rf_model_fit_predict_save(tmp_path):
    texts = [
        "scaler.fit(X)\nX_tr, X_te = split(X)",
        "X_tr, X_te = split(X)\nscaler.fit(X_tr)",
        "imputer.fit(X)\nX_tr, X_te = split(X)",
        "X_tr, X_te = split(X)\nimputer.fit(X_tr)"
    ]
    labels = np.array([1, 0, 1, 0])

    model = PipelineFeatureUnionModel(n_estimators=10, max_depth=3, random_state=42)
    model.fit(texts, labels)

    preds = model.predict(texts)
    probs = model.predict_proba(texts)

    assert len(preds) == 4
    assert probs.shape == (4, 2)

    save_path = tmp_path / "rf_model.joblib"
    joblib.dump(model, save_path)
    loaded = joblib.load(save_path)
    assert np.array_equal(model.predict(texts), loaded.predict(texts))
