"""Pre-run features only; provenance is retained solely for evaluation."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = ["log_params", "gpu_count", "tdp_w", "batch_size", "log_flops", "log_tokens"]
CATEGORICAL = ["precision", "task", "train_type"]
FEATURES = NUMERIC + CATEGORICAL


def prepare(df):
    df = df.drop_duplicates().copy()
    df["source"] = df["source"].fillna("unknown")
    df.loc[df.original_file.str.contains("codecarbon", na=False), "source"] = (
        "codecarbon"
    )
    df["row_id"] = df.index.astype(str)
    for raw, log in [
        ("n_params", "log_params"),
        ("estimated_flops", "log_flops"),
        ("estimated_tokens", "log_tokens"),
    ]:
        values = pd.to_numeric(df[raw], errors="coerce")
        transformed = np.log10(values.where(values >= 0) + 1)
        # The HF cleaning script saved log10(FLOPs), but omitted raw FLOPs.
        # Preserve this existing pre-run estimate; do not replace it with an imputed value.
        if log == "log_flops" and log in df:
            existing = pd.to_numeric(df[log], errors="coerce")
            transformed = transformed.fillna(np.log10(10.0**existing + 1))
        df[log] = transformed
    for col in ["gpu_count", "tdp_w", "batch_size"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL:
        df[col] = df[col].fillna("unknown").astype(str).str.lower()
    df["energy_kwh"] = pd.to_numeric(df.energy_kwh, errors="coerce")
    df = df[np.isfinite(df.energy_kwh) & (df.energy_kwh > 0)].copy()
    df["log_energy_kwh"] = np.log10(df.energy_kwh)
    return df[
        FEATURES + ["energy_kwh", "log_energy_kwh", "source", "original_file", "row_id"]
    ]


if __name__ == "__main__":
    data = prepare(pd.read_csv(ROOT / "data/merged_dataset.csv"))
    data.to_csv(ROOT / "data/final_training_dataset.csv", index=False)
    print(data.groupby("source").size().to_string())
