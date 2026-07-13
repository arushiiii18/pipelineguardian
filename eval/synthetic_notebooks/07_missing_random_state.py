# Injected issue: missing_random_state
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data.csv")
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # LEAK: no random_state

clf = RandomForestClassifier(n_estimators=100)  # LEAK: no random_state
clf.fit(X_train, y_train)
