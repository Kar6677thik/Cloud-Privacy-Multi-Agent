from sklearn.datasets import make_blobs
import pandas as pd
import os

# Generate large synthetic dataset
X, _ = make_blobs(
    n_samples=6000,
    n_features=10,
    centers=6,
    cluster_std=1.2,
    random_state=42
)

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(1, 11)])

# Create output folder
os.makedirs("datasets", exist_ok=True)

# Split into 3 parties
split_size = len(df) // 3
df.iloc[:split_size].to_csv("datasets/party1.csv", index=False)
df.iloc[split_size:2*split_size].to_csv("datasets/party2.csv", index=False)
df.iloc[2*split_size:].to_csv("datasets/party3.csv", index=False)

print("✅ Datasets created:")
print(" - datasets/party1.csv")
print(" - datasets/party2.csv")
print(" - datasets/party3.csv")
