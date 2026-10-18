"""
test_synthetic_robustness.py — Unit and integration tests for Phase 5 synthetic robustness checks.

Validates:
- AblatedFeatureUnionModel shapes and feature selection
- Rejection audit dictionary structure and non-zero counts
- Contrast pairs consistency
- Cluster bootstrap execution on small synthetic cluster subset
"""

import os
import sys
import numpy as np
import pytest

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from run_synthetic_robustness import (
    AblatedFeatureUnionModel,
    run_rejection_audit,
    run_contrast_pairs_evaluation,
    run_cluster_bootstrap
)


SAMPLE_TEXTS = [
    """
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X = np.random.randn(100, 5)
y = np.random.randint(0, 2, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
scaler.fit(X_train)
""",
    """
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X = np.random.randn(100, 5)
y = np.random.randint(0, 2, 100)
scaler = StandardScaler()
scaler.fit(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)
"""
]
SAMPLE_LABELS = np.array([0, 1])


def test_ablated_model_feature_dimensions():
    # Full mode: 17 struct + tfidf
    m_full = AblatedFeatureUnionModel(ablation_mode="full", model_type="rf")
    feat_full = m_full._transform_features(SAMPLE_TEXTS, fit_tfidf=True)
    assert feat_full.shape[0] == 2
    assert feat_full.shape[1] > 17

    # No shortcuts: 13 struct + tfidf
    m_nosc = AblatedFeatureUnionModel(ablation_mode="no_shortcuts", model_type="rf")
    feat_nosc = m_nosc._transform_features(SAMPLE_TEXTS, fit_tfidf=True)
    assert feat_nosc.shape[0] == 2
    assert feat_nosc.shape[1] == feat_full.shape[1] - 4  # 4 shortcut features removed

    # Structural only full: exactly 17
    m_struct = AblatedFeatureUnionModel(ablation_mode="structural_only_full", model_type="rf")
    feat_struct = m_struct._transform_features(SAMPLE_TEXTS, fit_tfidf=False)
    assert feat_struct.shape == (2, 17)

    # Structural only no shortcuts: exactly 13
    m_struct_nosc = AblatedFeatureUnionModel(ablation_mode="structural_only_no_shortcuts", model_type="rf")
    feat_struct_nosc = m_struct_nosc._transform_features(SAMPLE_TEXTS, fit_tfidf=False)
    assert feat_struct_nosc.shape == (2, 13)


def test_ablated_model_fit_predict():
    m = AblatedFeatureUnionModel(ablation_mode="no_shortcuts", model_type="rf")
    m.fit(SAMPLE_TEXTS, SAMPLE_LABELS)
    preds = m.predict(SAMPLE_TEXTS)
    assert len(preds) == 2
    assert set(preds).issubset({0, 1})


def test_rejection_audit_counts():
    audit = run_rejection_audit()
    assert audit["total_generated_candidates"] == 720
    assert audit["verified_sound_count"] == 551
    assert audit["total_rejected"] == 161
    assert audit["total_inconclusive"] == 8
    assert audit["verified_sound_count"] + audit["total_rejected"] + audit["total_inconclusive"] == 720
    assert "clean_function_wrapped" in audit["rejection_root_causes"]
