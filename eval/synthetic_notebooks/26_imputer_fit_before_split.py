# 26_imputer_fit_before_split.py
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

np.random.seed(42)
X = np.random.rand(100, 4)
y = np.random.randint(0, 2, 100)

imp = SimpleImputer(strategy='mean')
imp.fit(X)
X_imp = imp.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_imp, y, test_size=0.2, random_state=42)
