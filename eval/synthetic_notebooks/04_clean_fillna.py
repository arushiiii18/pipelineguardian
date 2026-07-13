# Clean version — impute using train-split statistic only
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

train_mean = X_train["income"].mean()
X_train["income"] = X_train["income"].fillna(train_mean)
X_test["income"] = X_test["income"].fillna(train_mean)
