# Clean: uses random module WITH random.seed() — should not trigger
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

random.seed(42)  # CLEAN: seed set before any random usage

df = pd.read_csv("data.csv")
X = df.drop(columns=["target"])
y = df["target"]

indices = list(range(len(df)))
random.shuffle(indices)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
