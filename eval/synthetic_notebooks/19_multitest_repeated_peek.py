# Injected issue: repeated_test_evaluation
# Two different models are both evaluated on the same X_test — no CV present
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LEAK: both models are scored on X_test — same held-out set used to pick among candidates
clf1 = RandomForestClassifier(random_state=42)
clf1.fit(X_train, y_train)
score1 = clf1.score(X_test, y_test)

clf2 = GradientBoostingClassifier(random_state=42)
clf2.fit(X_train, y_train)
score2 = clf2.score(X_test, y_test)

print(f"RF: {score1}, GBM: {score2}")
best = clf1 if score1 > score2 else clf2
