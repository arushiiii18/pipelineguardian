# 34_clean_random_state_kfold.py
from sklearn.model_selection import KFold
import numpy as np

np.random.seed(42)
X = np.random.rand(100, 4)
kf = KFold(n_splits=5, shuffle=True, random_state=99)
for train, test in kf.split(X):
    pass
