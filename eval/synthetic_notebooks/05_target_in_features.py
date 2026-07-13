# Injected issue: target_column_in_feature_list
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")
y = df["churn"]
X = df[["age", "income", "tenure", "churn"]]  # LEAK: 'churn' left in feature list

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
