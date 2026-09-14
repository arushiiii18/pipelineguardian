# Clean: uses GroupKFold so the same patient never appears in both splits
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score

df = pd.read_csv("17_group_data.csv")
X = df.drop(columns=["target", "patient_id"])
y = df["target"]
groups = df["patient_id"]

# CLEAN: GroupKFold ensures same patient stays in train or val, not both
gkf = GroupKFold(n_splits=4)
clf = RandomForestClassifier(random_state=42)
scores = cross_val_score(clf, X, y, cv=gkf, groups=groups)
print(f"CV scores: {scores}")
