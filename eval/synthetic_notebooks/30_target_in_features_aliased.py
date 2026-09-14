# 30_target_in_features_aliased.py
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.DataFrame({'feat1': [1, 2] * 50, 'feat2': [3, 4] * 50, 'label': [0, 1] * 50})

features = ['feat1', 'feat2', 'label']
target_col = 'label'

X = df[features]
y = df[target_col]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
