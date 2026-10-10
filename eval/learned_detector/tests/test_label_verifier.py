"""
Unit tests for the LabelVerificationOracle.
Tests execution sandboxing, statistical difference detection, and rejection of mismatched or malformed cases.
"""

import os
import sys
import pytest

# Ensure src is importable
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from verifier import LabelVerificationOracle


def test_verifier_detects_scaler_leakage():
    leakage_code = """import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
X = np.random.randn(100, 4)
y = (X[:, 0] > 0).astype(int)

# LEAKAGE: fit on full X before split
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.25, random_state=42)
clf = LogisticRegression()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
"""
    oracle = LabelVerificationOracle()
    res = oracle.verify_code(leakage_code, expected_label=1, mutation_type="scaler_fit_before_split")
    assert res["verified"] is True
    assert res["verification_status"] == "verified_leakage"


def test_verifier_detects_clean_pipeline():
    clean_code = """import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
X = np.random.randn(100, 4)
y = (X[:, 0] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train)
X_te_s = scaler.transform(X_test)

clf = LogisticRegression()
clf.fit(X_tr_s, y_train)
score = clf.score(X_te_s, y_test)
"""
    oracle = LabelVerificationOracle()
    res = oracle.verify_code(clean_code, expected_label=0, mutation_type="clean_base")
    assert res["verified"] is True
    assert res["verification_status"] == "verified_clean"
    assert res["stat_matches_train"] is True
    assert res["stat_matches_full"] is False


def test_verifier_rejects_syntax_error():
    broken_code = """def invalid_python(
        print('missing closing paren')
"""
    oracle = LabelVerificationOracle()
    res = oracle.verify_code(broken_code, expected_label=0, mutation_type="clean_base")
    assert res["verified"] is False
    assert res["verification_status"] == "rejected"
    assert "Syntax error" in res["reason"]


def test_verifier_rejects_mislabeled_sample():
    # Sound code, but fraudulently claimed to be leakage (label 1)
    clean_code = """import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
X = np.random.randn(100, 4)
y = (X[:, 0] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train)
X_te_s = scaler.transform(X_test)

clf = LogisticRegression()
clf.fit(X_tr_s, y_train)
score = clf.score(X_te_s, y_test)
"""
    oracle = LabelVerificationOracle()
    res = oracle.verify_code(clean_code, expected_label=1, mutation_type="fraudulent_leakage_claim")
    assert res["verified"] is False
    assert res["verification_status"] == "rejected"
    assert "Expected leakage" in res["reason"]
