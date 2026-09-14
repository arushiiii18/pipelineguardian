# Injected issue: group_unaware_split
# Uses train_test_split with a repeated patient_id column — same patient may
# appear in both train and test splits
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("17_group_data.csv")
X = df.drop(columns=["target"])
y = df["target"]

# LEAK: patient_id repeats across rows; random split may put same patient in train AND test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
