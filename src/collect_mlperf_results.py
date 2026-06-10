import pandas as pd

# Later we will replace this with real MLPerf extracted rows
mlperf_rows = [
    {
        "model_name": "resnet",
        "model_family": "cnn",
        "task": "image_classification",
        "train_type": "benchmark_training",
        "gpu_model": "unknown",
        "gpu_count": None,
        "batch_size": None,
        "precision": "unknown",
        "duration_seconds": None,
        "energy_kwh": None,
        "co2_kg": None,
        "source": "mlperf_power",
        "label_quality": "measured_mlperf"
    }
]

df = pd.DataFrame(mlperf_rows)

df.to_csv("data/mlperf_power_clean.csv", index=False)

print("Saved data/mlperf_power_clean.csv")
print(df.head())