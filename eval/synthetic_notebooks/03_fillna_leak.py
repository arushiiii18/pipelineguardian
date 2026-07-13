# Injected issue: fillna_uses_full_dataset_stat
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")
df["income"] = df["income"].fillna(df["income"].mean())  # LEAK: full-dataset mean

X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
