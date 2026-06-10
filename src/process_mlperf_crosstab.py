import pandas as pd
import numpy as np

INPUT_PATH = "data/mlperf_raw.csv"
OUTPUT_PATH = "data/mlperf_power_clean.csv"

df = pd.read_csv(INPUT_PATH, encoding="utf-16", sep="\t")

def clean_number(x):
    if pd.isna(x):
        return np.nan
    x = str(x).replace(",", "").strip()
    try:
        return float(x)
    except:
        return np.nan

def estimate_tdp(accelerator_name):
    text = str(accelerator_name).lower()

    if "gb300" in text:
        return 1200
    if "gb200" in text or "b200" in text:
        return 1000
    if "h200" in text or "h100" in text:
        return 700
    if "a100" in text:
        return 400
    if "v100" in text:
        return 300
    if "t4" in text:
        return 70

    return np.nan

# Exact column positions from your uploaded MLPerf file
public_id_col = "Unnamed: 0"
organization_col = "Unnamed: 2"
system_col = "Unnamed: 3"
accelerator_col = "Unnamed: 4"
total_acc_col = "Unnamed: 15"
metric_col = "Unnamed: 17"

# Latency columns are from column index 18 onwards
latency_cols = list(df.columns[18:])

df_avg = df[df[metric_col].astype(str).str.contains("Avg", case=False, na=False)].copy()

print("Avg rows found:", len(df_avg))
print("Latency columns:", latency_cols)

rows = []

for _, row in df_avg.iterrows():
    latency_values = []

    for col in latency_cols:
        value = clean_number(row[col])
        if not np.isnan(value):
            latency_values.append(value)

    if len(latency_values) == 0:
        continue

    latency_minutes = min(latency_values)

    gpu_model = row[accelerator_col]
    gpu_count = clean_number(row[total_acc_col])
    tdp_w = estimate_tdp(gpu_model)

    duration_hours = latency_minutes / 60

    if np.isnan(gpu_count) or np.isnan(tdp_w):
        energy_kwh = np.nan
    else:
        energy_kwh = (tdp_w * gpu_count * duration_hours) / 1000

    rows.append({
        "public_id": row[public_id_col],
        "model_name": row[system_col],
        "model_family": "mlperf_training_benchmark",
        "task": "training_benchmark",
        "train_type": "benchmark_training",
        "organization": row[organization_col],
        "system_name": row[system_col],
        "gpu_model": gpu_model,
        "gpu_count": gpu_count,
        "tdp_w": tdp_w,
        "duration_minutes": latency_minutes,
        "duration_seconds": latency_minutes * 60,
        "energy_kwh": energy_kwh,
        "co2_kg": np.nan,
        "source": "mlperf",
        "label_quality": "estimated_from_mlperf_latency"
    })

clean_df = pd.DataFrame(rows)

clean_df.to_csv(OUTPUT_PATH, index=False)

print("Saved:", OUTPUT_PATH)
print(clean_df.head())
print(clean_df.info())