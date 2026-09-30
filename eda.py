import pandas as pd

# Load dataset
df = pd.read_csv("data/test.csv")

# Separate features and target
X = df.drop(columns=["timestamp", "label", "event_id"])
y = df["label"]

print("Feature shape:", X.shape)
print("Target shape:", y.shape)

print("\nFeatures used for ML:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())

print("\nNormal samples:", (y == 0).sum())
print("Anomaly samples:", (y == 1).sum())