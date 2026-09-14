# 21_manual_boolean_mask_split.py
# Limitation demonstration: manual boolean mask splitting df[mask] / df[~mask]
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
df = pd.DataFrame({'a': np.random.rand(100), 'b': np.random.rand(100)})
mask = np.random.rand(len(df)) < 0.8
train = df[mask]
test = df[~mask]

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train)
