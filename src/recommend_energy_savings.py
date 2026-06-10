import joblib
import pandas as pd
import numpy as np

model = joblib.load("models/xgboost_energy_forecaster.pkl")

FEATURES = [
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

def predict_energy_kwh(input_data):
    X = pd.DataFrame([input_data])[FEATURES]
    pred_log = model.predict(X)[0]
    return 10 ** pred_log


def recommend_savings(base_input):
    base_energy = predict_energy_kwh(base_input)

    recommendations = []

    # 1. Reduce GPU count
    if base_input["gpu_count"] > 1:
        modified = base_input.copy()
        modified["gpu_count"] = max(1, base_input["gpu_count"] // 2)

        new_energy = predict_energy_kwh(modified)

        recommendations.append({
            "recommendation": "Reduce GPU count",
            "change": f"{base_input['gpu_count']} GPUs → {modified['gpu_count']} GPUs",
            "estimated_energy_kwh": new_energy,
            "estimated_saving_kwh": base_energy - new_energy,
            "saving_percent": ((base_energy - new_energy) / base_energy) * 100
        })

    # 2. Use lower TDP GPU
    if base_input["tdp_w"] > 400:
        modified = base_input.copy()
        modified["tdp_w"] = 400

        new_energy = predict_energy_kwh(modified)

        recommendations.append({
            "recommendation": "Use lower-power GPU",
            "change": f"{base_input['tdp_w']}W GPU → 400W GPU",
            "estimated_energy_kwh": new_energy,
            "estimated_saving_kwh": base_energy - new_energy,
            "saving_percent": ((base_energy - new_energy) / base_energy) * 100
        })

    # 3. Reduce FLOPs
    modified = base_input.copy()
    modified["log_flops"] = base_input["log_flops"] - 0.2

    new_energy = predict_energy_kwh(modified)

    recommendations.append({
        "recommendation": "Reduce compute/FLOPs",
        "change": "Reduce sequence length, tokens, or model size",
        "estimated_energy_kwh": new_energy,
        "estimated_saving_kwh": base_energy - new_energy,
        "saving_percent": ((base_energy - new_energy) / base_energy) * 100
    })

    # 4. Reduce duration
    modified = base_input.copy()
    modified["log_duration"] = base_input["log_duration"] - 0.2

    new_energy = predict_energy_kwh(modified)

    recommendations.append({
        "recommendation": "Optimize training runtime",
        "change": "Use better batching, mixed precision, or optimized dataloader",
        "estimated_energy_kwh": new_energy,
        "estimated_saving_kwh": base_energy - new_energy,
        "saving_percent": ((base_energy - new_energy) / base_energy) * 100
    })

    recommendations = sorted(
        recommendations,
        key=lambda x: x["estimated_saving_kwh"],
        reverse=True
    )

    return base_energy, recommendations[:3]


# Example input
base_input = {
    "log_params": np.log10(70_000_000_000 + 1),
    "gpu_count": 128,
    "tdp_w": 700,
    "batch_size": 256,
    "log_duration": np.log10(72 * 3600 + 1),
    "log_flops": np.log10(float(6) * float(70_000_000_000) * float(50_000_000_000) + 1),
    "log_tokens": np.log10(float(50_000_000_000) + 1),
    "task_enc": 0,
    "train_type_enc": 0,
    "gpu_model_enc": 0,
    "source_enc": 0,
    "label_quality_enc": 0,
    "precision_enc": 0
}

base_energy, recommendations = recommend_savings(base_input)

print("\nBASELINE ENERGY PREDICTION")
print("---------------------------")
print(f"Predicted energy: {base_energy:.2f} kWh")

print("\nTOP ENERGY SAVING RECOMMENDATIONS")
print("----------------------------------")

for i, rec in enumerate(recommendations, 1):
    print(f"\n{i}. {rec['recommendation']}")
    print(f"Change: {rec['change']}")
    print(f"New estimated energy: {rec['estimated_energy_kwh']:.2f} kWh")
    print(f"Estimated saving: {rec['estimated_saving_kwh']:.2f} kWh")
    print(f"Saving percent: {rec['saving_percent']:.2f}%")