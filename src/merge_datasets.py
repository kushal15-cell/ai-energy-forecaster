import pandas as pd
import os

files = [
    "data/mlperf_power_clean.csv",
    "data/codecarbon_clean.csv",
    "data/dataset_clean.csv",
    "data/synthetic_energy_data.csv"
]

dfs = []

for file in files:
    if os.path.exists(file):
        df = pd.read_csv(file)
        df["original_file"] = file
        dfs.append(df)
        print(f"Loaded {file}: {df.shape}")
    else:
        print(f"Missing: {file}")

merged = pd.concat(dfs, ignore_index=True, sort=False)

merged.to_csv("data/merged_dataset.csv", index=False)

print("Merged shape:", merged.shape)
print("Saved data/merged_dataset.csv")
print(merged["source"].value_counts(dropna=False))