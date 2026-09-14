# 37_overlap_clean_train_eval_split.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
X = np.random.rand(100, 4)
y = np.random.randint(0, 2, 100)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(random_state=42)
clf.fit(X_tr, y_tr)
score = clf.score(X_te, y_te)
