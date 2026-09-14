# 39_multitest_clean_validation_test.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
X = np.random.rand(100, 4)
y = np.random.randint(0, 2, 100)

X_tr_val, X_te, y_tr_val, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(X_tr_val, y_tr_val, test_size=0.25, random_state=42)

clf1 = LogisticRegression(random_state=42).fit(X_tr, y_tr)
score1 = clf1.score(X_val, y_val)

clf2 = RandomForestClassifier(random_state=42).fit(X_tr, y_tr)
score2 = clf2.score(X_val, y_val)

best_model = clf1 if score1 >= score2 else clf2
final_test_score = best_model.score(X_te, y_te)
