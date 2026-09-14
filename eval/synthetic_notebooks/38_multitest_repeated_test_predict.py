# 38_multitest_repeated_test_predict.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
X = np.random.rand(100, 4)
y = np.random.randint(0, 2, 100)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

clf1 = LogisticRegression(random_state=42)
clf1.fit(X_tr, y_tr)
p1 = clf1.predict(X_te)

clf2 = RandomForestClassifier(random_state=42)
clf2.fit(X_tr, y_tr)
p2 = clf2.predict(X_te)
