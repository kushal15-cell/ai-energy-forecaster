import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load merged dataset
df = pd.read_csv("data/merged_dataset.csv")

print("Merged shape:", df.shape)

df = df.drop_duplicates()

print("After removing duplicates:", df.shape)

# Keep only usable energy rows
df = df.dropna(subset=["energy_kwh"])
df = df[df["energy_kwh"] > 0]

# -----------------------------
# NUMERIC COLUMNS
# -----------------------------

numeric_cols = [
    "n_params",
    "gpu_count",
    "tdp_w",
    "duration_seconds",
    "batch_size",
    "estimated_flops",
    "estimated_tokens",
    "co2_kg"
]

for col in numeric_cols:

    # Create missing columns if absent
    if col not in df.columns:
        df[col] = np.nan

    # Convert object/mixed values to numeric
    df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing with median
    median_val = df[col].median()

    if np.isnan(median_val):
        median_val = 0

    df[col] = df[col].fillna(median_val)

# -----------------------------
# CATEGORICAL COLUMNS
# -----------------------------

categorical_cols = [
    "task",
    "train_type",
    "gpu_model",
    "source",
    "label_quality",
    "precision"
]

for col in categorical_cols:

    if col not in df.columns:
        df[col] = "unknown"

    df[col] = df[col].fillna("unknown")

# -----------------------------
# LOG TRANSFORMS
# -----------------------------

df["log_energy_kwh"] = np.log10(df["energy_kwh"] + 1e-9)

df["log_params"] = np.log10(df["n_params"] + 1)

df["log_duration"] = np.log10(df["duration_seconds"] + 1)

df["log_flops"] = np.log10(df["estimated_flops"] + 1)

df["log_tokens"] = np.log10(df["estimated_tokens"] + 1)

# -----------------------------
# LABEL ENCODING
# -----------------------------

encoders = {}

for col in categorical_cols:

    le = LabelEncoder()

    df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    encoders[col] = le

# -----------------------------
# FINAL FEATURES
# -----------------------------

features = [
    "log_params",
    "gpu_count",
    "tdp_w",
    "batch_size",
    "log_duration",
    "log_flops",
    "log_tokens",
    "task_enc",
    "train_type_enc",
    "gpu_model_enc",
    "source_enc",
    "label_quality_enc",
    "precision_enc"
]

final_df = df[features + ["energy_kwh", "log_energy_kwh"]]

# -----------------------------
# SAVE FINAL DATASET
# -----------------------------

final_df.to_csv("data/final_training_dataset.csv", index=False)

print("\nFinal dataset shape:", final_df.shape)

print("\nFirst 5 rows:")
print(final_df.head())

print("\nSaved:")
print("data/final_training_dataset.csv")