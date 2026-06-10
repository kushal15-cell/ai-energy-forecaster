import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

N_ROWS = 12000

gpu_specs = {
    "T4": 70,
    "V100": 300,
    "A100": 400,
    "H100": 700,
    "H200": 700,
    "B200": 1000,
    "GB200": 1000,
    "GB300": 1200
}

model_sizes = [
    82_000_000,
    125_000_000,
    355_000_000,
    760_000_000,
    1_300_000_000,
    2_700_000_000,
    6_700_000_000,
    7_000_000_000,
    13_000_000_000,
    34_000_000_000,
    70_000_000_000
]

tasks = [
    "text-generation",
    "image-classification",
    "translation",
    "summarization",
    "question-answering",
    "training_benchmark"
]

train_types = ["fine-tune", "pre-train", "benchmark_training"]
precisions = ["fp32", "fp16", "bf16"]

rows = []

for _ in range(N_ROWS):
    n_params = random.choice(model_sizes)
    gpu_model = random.choice(list(gpu_specs.keys()))
    tdp_w = gpu_specs[gpu_model]

    gpu_count = random.choice([1, 2, 4, 8, 16, 32, 64, 128, 256])
    batch_size = random.choice([8, 16, 32, 64, 128, 256, 512])
    precision = random.choice(precisions)
    task = random.choice(tasks)
    train_type = random.choice(train_types)

    if train_type == "pre-train":
        tokens = random.choice([10_000_000_000, 30_000_000_000, 50_000_000_000, 100_000_000_000])
    else:
        tokens = random.choice([50_000_000, 100_000_000, 500_000_000, 1_000_000_000])

    estimated_flops = 6 * n_params * tokens

    precision_factor = {
        "fp32": 1.0,
        "fp16": 0.65,
        "bf16": 0.70
    }[precision]

    gpu_speed_factor = {
        "T4": 0.4,
        "V100": 0.7,
        "A100": 1.0,
        "H100": 1.7,
        "H200": 1.8,
        "B200": 2.2,
        "GB200": 2.5,
        "GB300": 3.0
    }[gpu_model]

    # Synthetic runtime equation
    # Bigger models/tokens increase time
    # More GPUs and faster GPUs reduce time
    base_hours = (estimated_flops / 1e21) / (gpu_count * gpu_speed_factor)

    # Precision reduces runtime
    train_hours = base_hours * precision_factor

    # Add realistic noise
    noise = np.random.normal(loc=1.0, scale=0.15)
    train_hours = max(train_hours * noise, 0.01)

    # Energy equation
    energy_kwh = (tdp_w * gpu_count * train_hours) / 1000

    # Add cooling/data-center overhead using PUE
    pue = np.random.uniform(1.1, 1.6)
    energy_kwh = energy_kwh * pue

    # Carbon intensity estimate kg CO2 per kWh
    carbon_intensity = np.random.uniform(0.2, 0.8)
    co2_kg = energy_kwh * carbon_intensity

    rows.append({
        "model_name": "synthetic_model",
        "model_family": "transformer",
        "task": task,
        "train_type": train_type,
        "n_params": n_params,
        "gpu_model": gpu_model,
        "gpu_count": gpu_count,
        "tdp_w": tdp_w,
        "batch_size": batch_size,
        "precision": precision,
        "estimated_tokens": tokens,
        "estimated_flops": estimated_flops,
        "duration_seconds": train_hours * 3600,
        "energy_kwh": energy_kwh,
        "co2_kg": co2_kg,
        "source": "synthetic",
        "label_quality": "simulated_physics_based"
    })

df = pd.DataFrame(rows)

df.to_csv("data/synthetic_energy_data.csv", index=False)

print("Saved data/synthetic_energy_data.csv")
print("Shape:", df.shape)
print(df.head())
print(df["energy_kwh"].describe())