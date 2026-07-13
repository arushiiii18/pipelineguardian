# True negative: timestamp column present, but the code ALREADY uses a
# correct chronological split (not a plain random train_test_split).
# validation_strategy_reviewer should NOT flag this.
import pandas as pd

df = pd.read_csv("data.csv")
df = df.sort_values("event_date")

split_point = int(len(df) * 0.8)
train = df.iloc[:split_point]
test = df.iloc[split_point:]

X_train, y_train = train.drop(columns=["target"]), train["target"]
X_test, y_test = test.drop(columns=["target"]), test["target"]
