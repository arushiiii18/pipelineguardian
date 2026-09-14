# Clean: CV for model selection, then exactly ONE final score on genuine held-out test
# This is the §7.2 priority regression case — must produce ZERO MULTI_TEST issues
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_csv("data.csv")
X = df.drop(columns=["target"])
y = df["target"]

# First hold out a genuine test set — never touched during model selection
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CLEAN: use cross_val_score on train+val for model selection — X_test not touched here
cv_rf = cross_val_score(RandomForestClassifier(random_state=42), X_trainval, y_trainval, cv=5)
cv_gbm = cross_val_score(GradientBoostingClassifier(random_state=42), X_trainval, y_trainval, cv=5)

# Pick best model based on CV, not test set
if cv_rf.mean() > cv_gbm.mean():
    best = RandomForestClassifier(random_state=42)
else:
    best = GradientBoostingClassifier(random_state=42)

# CLEAN: only ONE final evaluation on the genuinely held-out test set
best.fit(X_trainval, y_trainval)
final_score = best.score(X_test, y_test)
print(f"Final test score: {final_score}")
