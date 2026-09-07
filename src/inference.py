"""Single inference contract shared by the app, scenarios, and batch export."""

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from xgboost import DMatrix

from src.final_feature_engineering import CATEGORICAL, FEATURES, NUMERIC, ROOT

# Values from the local synthetic generator; these are simulation assumptions.
GPU_TDP = {
    "T4": 70,
    "V100": 300,
    "A100": 400,
    "H100": 700,
    "H200": 700,
    "B200": 1000,
    "GB200": 1000,
    "GB300": 1200,
}
DEFAULT = {
    "n_params": 760_000_000,
    "estimated_tokens": 1_000_000_000,
    "gpu_model": "A100",
    "gpu_count": 4,
    "batch_size": 64,
    "precision": "fp16",
    "task": "text-generation",
    "train_type": "fine-tune",
}
LABELS = {
    "log_params": "Model parameters",
    "gpu_count": "GPU count",
    "tdp_w": "GPU power assumption",
    "batch_size": "Batch size",
    "log_flops": "Estimated FLOPs",
    "log_tokens": "Training tokens",
    "precision": "Precision",
    "task": "Task",
    "train_type": "Training type",
}


@lru_cache(maxsize=1)
def artifacts():
    return (
        joblib.load(ROOT / "models/xgboost_energy_forecaster.pkl"),
        json.loads((ROOT / "models/metadata.json").read_text()),
    )


def make_features(config):
    params, tokens = float(config["n_params"]), float(config["estimated_tokens"])
    if not np.isfinite(
        [params, tokens, config["gpu_count"], config["batch_size"]]
    ).all():
        raise ValueError("Inputs must be finite.")
    if min(params, tokens, config["gpu_count"], config["batch_size"]) <= 0:
        raise ValueError(
            "Parameters, tokens, GPU count, and batch size must be positive."
        )
    if config["gpu_model"] not in GPU_TDP:
        raise ValueError("GPU is not supported by this simulator.")
    row = {
        "log_params": np.log10(params + 1),
        "log_tokens": np.log10(tokens + 1),
        "log_flops": np.log10(6.0 * params * tokens + 1),
        "gpu_count": config["gpu_count"],
        "batch_size": config["batch_size"],
        "tdp_w": GPU_TDP[config["gpu_model"]],
    }
    row.update({c: str(config[c]).lower() for c in CATEGORICAL})
    return pd.DataFrame([row], columns=FEATURES)


def forecast(config, explain=True):
    model, meta = artifacts()
    frame = make_features(config)
    pred = float(model.predict(frame)[0])
    radius = meta["synthetic_interval"]["log10_radius"]
    warnings = []
    for c in NUMERIC:
        low, high = meta["training_ranges"][c]
        if not low <= frame.iloc[0][c] <= high:
            warnings.append(f"{LABELS[c]} is outside the synthetic training range.")
    for c in CATEGORICAL:
        if frame.iloc[0][c] not in meta["categories"][c]:
            warnings.append(f"Unseen {LABELS[c].lower()}: {frame.iloc[0][c]}.")
    result = {
        "energy_kwh": 10.0**pred,
        "lower_kwh": 10.0 ** (pred - radius),
        "upper_kwh": 10.0 ** (pred + radius),
        "warnings": warnings,
        "model_version": meta["version"],
    }
    if explain:
        transformed = model["preprocess"].transform(frame)
        contributions = (
            model["regressor"]
            .get_booster()
            .predict(DMatrix(transformed), pred_contribs=True)[0]
        )
        # Exact TreeSHAP contributions, aggregated from one-hot columns to input fields.
        grouped = {c: float(contributions[i]) for i, c in enumerate(NUMERIC)}
        offset = len(NUMERIC)
        for c, categories in zip(
            CATEGORICAL, model["preprocess"].named_transformers_["category"].categories_
        ):
            grouped[c] = float(contributions[offset : offset + len(categories)].sum())
            offset += len(categories)
        result["contributions"] = grouped
        result["base_log10"] = float(contributions[-1])
        top = sorted(grouped, key=lambda c: abs(grouped[c]), reverse=True)[:2]
        result["explanation"] = (
            "; ".join(
                f"{LABELS[c]} {'raises' if grouped[c] >= 0 else 'lowers'} this estimate relative to the model baseline"
                for c in top
            )
            + "."
        )
    return result


def scenarios(config):
    variants = [
        ("Current plan", config),
        (
            "20% fewer tokens",
            {**config, "estimated_tokens": config["estimated_tokens"] * 0.8},
        ),
        ("Half the GPUs", {**config, "gpu_count": max(1, config["gpu_count"] // 2)}),
        ("FP16 precision", {**config, "precision": "fp16"}),
        ("BF16 precision", {**config, "precision": "bf16"}),
    ]
    rows = []
    baseline = forecast(config, False)["energy_kwh"]
    for name, variant in variants:
        result = forecast(variant, False)
        rows.append(
            {
                "scenario": name,
                "energy_kwh": result["energy_kwh"],
                "change_percent": 100 * (result["energy_kwh"] / baseline - 1),
                "upper_kwh": result["upper_kwh"],
                "input_warning": " ".join(result["warnings"]),
            }
        )
    return pd.DataFrame(rows)
