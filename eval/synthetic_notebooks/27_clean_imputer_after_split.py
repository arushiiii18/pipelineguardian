# 27_clean_imputer_after_split.py
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

np.random.seed(42)
X = np.random.rand(100, 4)
y = np.random.randint(0, 2, 100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imp = SimpleImputer(strategy='mean')
X_train_imp = imp.fit_transform(X_train)
X_test_imp = imp.transform(X_test)
