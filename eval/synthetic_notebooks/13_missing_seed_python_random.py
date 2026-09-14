# Injected issue: missing_seed_python_random
# Uses random module but never calls random.seed()
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data.csv")
X = df.drop(columns=["target"])
y = df["target"]

# Randomly shuffle indices without seeding — results vary run-to-run
indices = list(range(len(df)))
random.shuffle(indices)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
