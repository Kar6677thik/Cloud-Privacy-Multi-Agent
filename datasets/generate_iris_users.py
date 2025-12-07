from sklearn.datasets import load_iris
import pandas as pd
import os

# Load iris dataset
iris = load_iris(as_frame=True)
df = iris.frame

# Remove the target labels (unsupervised clustering)
df = df.drop(columns=["target"])

# Create output folder
os.makedirs("datasets", exist_ok=True)

# Split into 3 data owners (50 samples each)
df.iloc[0:50].to_csv("datasets/user1.csv", index=False)
df.iloc[50:100].to_csv("datasets/user2.csv", index=False)
df.iloc[100:150].to_csv("datasets/user3.csv", index=False)

print("✅ Datasets created:")
print(" - datasets/user1.csv")
print(" - datasets/user2.csv")
print(" - datasets/user3.csv")
