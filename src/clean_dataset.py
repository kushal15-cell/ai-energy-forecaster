import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/hf_raw_models.csv")

print("Raw shape:", df.shape)

# Remove rows without parameter count
df = df.dropna(subset=["n_params"])

# Fill missing GPU info
df["gpu_model"] = df["gpu_model"].fillna("unknown")
df["tdp_w"] = df["tdp_w"].fillna(0)

# Fill task/train type
df["task"] = df["task"].fillna("unknown")
df["train_type"] = df["train_type"].fillna("unknown")

# Estimate GPU count
df["gpu_count"] = 1

# Estimate training tokens
# This is a placeholder estimate because Hugging Face cards often do not provide exact tokens
df["estimated_tokens"] = np.where(
    df["train_type"] == "pre-train",
    50_000_000_000,
    500_000_000
)

# Estimate FLOPs
# Common rough formula: training FLOPs ≈ 6 × parameters × tokens
df["estimated_flops"] = 6 * df["n_params"] * df["estimated_tokens"]

df["log_flops"] = np.log10(df["estimated_flops"])
df["log_params"] = np.log10(df["n_params"])

# Estimate training hours
df["estimated_hours"] = np.where(
    df["train_type"] == "pre-train",
    df["n_params"] / 1_000_000_000 * 8,
    df["n_params"] / 1_000_000_000 * 1
)

df["estimated_hours"] = df["estimated_hours"].clip(lower=0.1, upper=500)

# Estimate energy
# kWh = GPU power in kW × hours × GPU count
df["energy_kwh"] = (df["tdp_w"] / 1000) * df["estimated_hours"] * df["gpu_count"]

# Remove impossible energy values
df = df[df["energy_kwh"] > 0]

df["log_energy_kwh"] = np.log10(df["energy_kwh"])

# Encode categorical columns
task_encoder = LabelEncoder()
train_encoder = LabelEncoder()
gpu_encoder = LabelEncoder()

df["task_enc"] = task_encoder.fit_transform(df["task"])
df["train_type_enc"] = train_encoder.fit_transform(df["train_type"])
df["gpu_model_enc"] = gpu_encoder.fit_transform(df["gpu_model"])

final_cols = [
    "model_id",
    "n_params",
    "log_params",
    "log_flops",
    "gpu_model",
    "gpu_model_enc",
    "gpu_count",
    "tdp_w",
    "train_type",
    "train_type_enc",
    "task",
    "task_enc",
    "downloads",
    "likes",
    "energy_kwh",
    "log_energy_kwh",
    "source"
]

clean_df = df[final_cols]

clean_df.to_csv("data/dataset_clean.csv", index=False)

print("Clean shape:", clean_df.shape)
print(clean_df.head())
print("Saved data/dataset_clean.csv")