# 36_overlap_no_split_eval_score.py
import numpy as np
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
X = np.random.rand(100, 4)
y = np.random.randint(0, 2, 100)

clf = LogisticRegression(random_state=42)
clf.fit(X, y)
score = clf.score(X, y)
