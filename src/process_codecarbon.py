import pandas as pd
import os

input_path = "data/emissions.csv"
output_path = "data/codecarbon_clean.csv"

df = pd.read_csv(input_path)

print("Columns found:")
print(df.columns.tolist())
print("Shape:", df.shape)

# CodeCarbon usually gives emissions in kg CO2eq and energy in kWh
clean_df = pd.DataFrame(index=df.index)

clean_df["model_name"] = "RandomForest_digits"
clean_df["model_family"] = "random_forest"
clean_df["task"] = "classification"
clean_df["train_type"] = "training"
clean_df["source"] = "codecarbon"
clean_df["label_quality"] = "measured_codecarbon"

# Safe column extraction
clean_df["energy_kwh"] = df["energy_consumed"] if "energy_consumed" in df.columns else None
clean_df["co2_kg"] = df["emissions"] if "emissions" in df.columns else None
clean_df["duration_seconds"] = df["duration"] if "duration" in df.columns else None

# Hardware fields
clean_df["cpu_model"] = df["cpu_model"] if "cpu_model" in df.columns else "unknown"
clean_df["gpu_model"] = df["gpu_model"] if "gpu_model" in df.columns else "none"
clean_df["gpu_count"] = df["gpu_count"] if "gpu_count" in df.columns else 0

# ML experiment metadata we know from our script
clean_df["n_params"] = None
clean_df["batch_size"] = None
clean_df["precision"] = "fp32"
clean_df["dataset_name"] = "sklearn_digits"

clean_df.to_csv(output_path, index=False)

print("Saved:", output_path)
print(clean_df.head())
