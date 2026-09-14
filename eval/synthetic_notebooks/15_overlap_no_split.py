# Injected issue: no_split_before_fit_eval
# Model is fit and scored on the SAME data — no train_test_split anywhere
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data.csv")
X = df.drop(columns=["target"])
y = df["target"]

# LEAK: no split of any kind — clf is fit and scored on identical X
clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)
score = clf.score(X, y)  # evaluating on training data — optimistic and meaningless
print(f"Score: {score}")
