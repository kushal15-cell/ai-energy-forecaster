import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="Budget Mode",
    page_icon="💰",
    layout="wide"
)

model = joblib.load("models/xgboost_energy_forecaster.pkl")

GPU_TDP = {
    "T4": 70,
    "V100": 300,
    "A100": 400,
    "H100": 700,
    "H200": 700,
    "B200": 1000,
    "GB200": 1000,
    "GB300": 1200
}

candidate_models = [
    66_000_000,
    110_000_000,
    124_000_000,
    355_000_000,
    760_000_000,
    1_300_000_000,
    2_700_000_000,
    7_000_000_000,
    13_000_000_000,
    34_000_000_000,
    70_000_000_000
]

def get_model_scale_config(params):
    if params <= 500_000_000:
        return {
            "tokens": 1_000_000_000,
            "gpu_count": 1,
            "hours": 3,
            "batch_size": 32,
            "recommended_gpu": "T4"
        }

    elif params <= 2_000_000_000:
        return {
            "tokens": 10_000_000_000,
            "gpu_count": 4,
            "hours": 12,
            "batch_size": 64,
            "recommended_gpu": "V100"
        }

    elif params <= 7_000_000_000:
        return {
            "tokens": 100_000_000_000,
            "gpu_count": 8,
            "hours": 48,
            "batch_size": 128,
            "recommended_gpu": "A100"
        }

    elif params <= 13_000_000_000:
        return {
            "tokens": 300_000_000_000,
            "gpu_count": 16,
            "hours": 96,
            "batch_size": 256,
            "recommended_gpu": "A100"
        }

    elif params <= 34_000_000_000:
        return {
            "tokens": 800_000_000_000,
            "gpu_count": 64,
            "hours": 168,
            "batch_size": 512,
            "recommended_gpu": "H100"
        }

    else:
        return {
            "tokens": 2_000_000_000_000,
            "gpu_count": 256,
            "hours": 240,
            "batch_size": 512,
            "recommended_gpu": "H100"
        }

def make_features(params, tokens, gpu, gpu_count, batch_size, hours, precision_enc=0):
    flops = float(6) * float(params) * float(tokens)

    return pd.DataFrame([{
        "log_params": np.log10(float(params) + 1),
        "gpu_count": gpu_count,
        "tdp_w": GPU_TDP[gpu],
        "batch_size": batch_size,
        "log_duration": np.log10(float(hours) * 3600 + 1),
        "log_flops": np.log10(flops + 1),
        "log_tokens": np.log10(float(tokens) + 1),
        "task_enc": 0,
        "train_type_enc": 0,
        "gpu_model_enc": 0,
        "source_enc": 0,
        "label_quality_enc": 0,
        "precision_enc": precision_enc
    }])

def predict_energy(params, tokens, gpu, gpu_count, batch_size, hours, precision_enc=0):
    X = make_features(params, tokens, gpu, gpu_count, batch_size, hours, precision_enc)
    pred_log = model.predict(X)[0]
    return 10 ** pred_log

def estimate_cost(energy_kwh, rate):
    return energy_kwh * rate

def estimate_co2(energy_kwh, carbon_factor=0.45):
    return energy_kwh * carbon_factor

st.title("💰 Training Budget Mode")

st.write(
    "Enter your energy or financial budget and find the largest model size that fits realistic training assumptions."
)

with st.form("budget_form"):

    budget_type = st.selectbox(
        "Budget Type",
        ["Energy Budget (kWh)", "Financial Budget ($)"]
    )

    electricity_rate = st.number_input(
        "Electricity / Cloud Energy Rate ($ per kWh)",
        min_value=0.01,
        max_value=5.0,
        value=0.15
    )

    if budget_type == "Energy Budget (kWh)":
        budget_limit = st.number_input(
            "Maximum Energy Budget (kWh)",
            min_value=1.0,
            max_value=1_000_000.0,
            value=100.0
        )

        effective_kwh_budget = budget_limit

    else:
        dollar_budget = st.number_input(
            "Maximum Financial Budget ($)",
            min_value=1.0,
            max_value=1_000_000.0,
            value=50.0
        )

        effective_kwh_budget = dollar_budget / electricity_rate

    st.subheader("Optimization Preference")

    precision = st.selectbox(
        "Precision",
        ["FP32", "FP16", "BF16"],
        index=1
    )

    submitted = st.form_submit_button("Find Largest Trainable Model")

if submitted:

    precision_enc = {
        "FP32": 0,
        "FP16": 1,
        "BF16": 2
    }[precision]

    valid_configs = []
    all_results = []

    for params_candidate in candidate_models:

        scale_config = get_model_scale_config(params_candidate)

        gpu = scale_config["recommended_gpu"]
        gpu_count = scale_config["gpu_count"]
        batch_size = scale_config["batch_size"]
        hours = scale_config["hours"]
        tokens = scale_config["tokens"]

        predicted_energy = predict_energy(
            params_candidate,
            tokens,
            gpu,
            gpu_count,
            batch_size,
            hours,
            precision_enc
        )

        predicted_cost = estimate_cost(predicted_energy, electricity_rate)
        predicted_co2 = estimate_co2(predicted_energy)

        row = {
            "Model Size": f"{params_candidate / 1e9:.2f}B",
            "Parameters": params_candidate,
            "Recommended GPU": gpu,
            "GPU Count": gpu_count,
            "Training Tokens": tokens,
            "Training Hours": hours,
            "Batch Size": batch_size,
            "Precision": precision,
            "Energy kWh": predicted_energy,
            "Cost $": predicted_cost,
            "CO2 kg": predicted_co2,
            "Fits Budget": predicted_energy <= effective_kwh_budget
        }

        all_results.append(row)

        if predicted_energy <= effective_kwh_budget:
            valid_configs.append(row)

    result_df = pd.DataFrame(all_results)

    if valid_configs:

        valid_df = pd.DataFrame(valid_configs)
        best = valid_df.sort_values("Parameters", ascending=False).iloc[0]

        st.success(
            f"Largest viable model: **{best['Model Size']} parameters**"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Estimated Energy",
            f"{best['Energy kWh']:,.2f} kWh"
        )

        col2.metric(
            "Estimated Cost",
            f"${best['Cost $']:,.2f}"
        )

        col3.metric(
            "Estimated CO₂",
            f"{best['CO2 kg']:,.2f} kg"
        )

        st.subheader("Recommended Configuration")

        config_df = pd.DataFrame([{
            "Model Size": best["Model Size"],
            "GPU": best["Recommended GPU"],
            "GPU Count": best["GPU Count"],
            "Training Tokens": best["Training Tokens"],
            "Training Hours": best["Training Hours"],
            "Batch Size": best["Batch Size"],
            "Precision": best["Precision"]
        }])

        st.dataframe(config_df, use_container_width=True)

    else:

        st.error(
            "No model configuration fits within this budget. Try increasing your budget or using mixed precision."
        )

    st.subheader("All Candidate Models")

    display_df = result_df.copy()

    display_df["Energy kWh"] = display_df["Energy kWh"].map(lambda x: f"{x:,.2f}")
    display_df["Cost $"] = display_df["Cost $"].map(lambda x: f"{x:,.2f}")
    display_df["CO2 kg"] = display_df["CO2 kg"].map(lambda x: f"{x:,.2f}")

    st.dataframe(display_df, use_container_width=True)

    st.download_button(
        label="Download Budget Analysis",
        data=result_df.to_csv(index=False),
        file_name="budget_mode_results.csv",
        mime="text/csv"
    )

else:

    st.info(
        "Enter your budget and click 'Find Largest Trainable Model'."
    )