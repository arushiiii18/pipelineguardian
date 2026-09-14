# 24_label_encoder_before_split.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

np.random.seed(42)
df = pd.DataFrame({'cat': ['a', 'b', 'a', 'c'] * 25, 'val': np.random.rand(100), 'target': np.random.randint(0, 2, 100)})

le = LabelEncoder()
df['cat_enc'] = le.fit_transform(df['cat'])

X_tr, X_te, y_tr, y_te = train_test_split(df[['cat_enc', 'val']], df['target'], test_size=0.2, random_state=42)
