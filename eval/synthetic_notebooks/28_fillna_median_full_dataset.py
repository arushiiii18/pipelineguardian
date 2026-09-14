# 28_fillna_median_full_dataset.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.DataFrame({'a': [1.0, 2.0, np.nan, 4.0] * 25, 'target': [0, 1] * 50})
df['a'] = df['a'].fillna(df['a'].median())

X_tr, X_te, y_tr, y_te = train_test_split(df[['a']], df['target'], test_size=0.2, random_state=42)
