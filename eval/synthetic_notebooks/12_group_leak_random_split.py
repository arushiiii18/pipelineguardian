# True positive for validation_strategy_reviewer: group/ID column present
# (same customer has multiple rows), but code uses plain random split,
# so the same customer can appear in both train and test.
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
