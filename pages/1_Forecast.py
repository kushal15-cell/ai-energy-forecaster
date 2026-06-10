import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Forecast",
    page_icon="⚡",
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

PRESETS = {
    "🟢 Small - DistilBERT": {
        "params": 66_000_000,
        "tokens": 1_000_000_000,
        "gpu": "T4",
        "count": 1,
        "batch": 32,
        "hours": 3
    },
    "🟡 Medium - BERT Base": {
        "params": 110_000_000,
        "tokens": 3_400_000_000,
        "gpu": "A100",
        "count": 4,
        "batch": 64,
        "hours": 24
    },
    "🔴 Large - GPT-2": {
        "params": 124_000_000,
        "tokens": 40_000_000_000,
        "gpu": "A100",
        "count": 8,
        "batch": 128,
        "hours": 72
    },
    "Custom": {
        "params": 7_000_000_000,
        "tokens": 50_000_000_000,
        "gpu": "A100",
        "count": 8,
        "batch": 128,
        "hours": 24
    }
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

def estimate_cost(energy_kwh, rate=0.15):
    return energy_kwh * rate

def estimate_co2(energy_kwh, carbon_factor=0.45):
    return energy_kwh * carbon_factor

def energy_level(energy):
    if energy < 10:
        return "Low"
    elif energy < 100:
        return "Medium"
    return "High"

def interpretation_line(energy):
    bulb_hours = energy / 0.1
    bulb_days = bulb_hours / 24
    return f"Training this configuration uses roughly the same electricity as running a 100W bulb for about {bulb_days:,.1f} days."

def gauge_chart(energy):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=energy,
        number={"suffix": " kWh"},
        title={"text": "Energy Intensity"},
        gauge={
            "axis": {"range": [None, max(150, energy * 1.2)]},
            "bar": {"color": "white"},
            "steps": [
                {"range": [0, 10], "color": "green"},
                {"range": [10, 100], "color": "orange"},
                {"range": [100, max(150, energy * 1.2)], "color": "red"}
            ],
        }
    ))

    fig.update_layout(height=320)
    return fig

st.title("⚡ Forecast AI Training Energy")

st.write(
    "Estimate AI training energy, cost, and carbon footprint before starting GPU training."
)

with st.form("forecast_form"):

    st.subheader("Quick Start")

    preset = st.selectbox(
        "Choose a preset",
        list(PRESETS.keys()),
        help="Use presets to quickly test realistic model configurations."
    )

    default = PRESETS[preset]

    st.subheader("Training Configuration")

    params = st.number_input(
        "Model Parameters",
        min_value=1_000_000,
        max_value=500_000_000_000,
        value=int(default["params"]),
        step=1_000_000,
        help="Total trainable parameters. Examples: DistilBERT = 66M, BERT-base = 110M, GPT-2 = 124M, LLaMA-7B = 7B."
    )

    tokens = st.number_input(
        "Training Tokens",
        min_value=1_000_000,
        max_value=10_000_000_000_000,
        value=int(default["tokens"]),
        step=1_000_000,
        help="Amount of training text processed. Small fine-tunes may use millions/billions; large pre-training can use trillions."
    )

    gpu = st.selectbox(
        "GPU Type",
        list(GPU_TDP.keys()),
        index=list(GPU_TDP.keys()).index(default["gpu"]),
        help="GPU used for training. T4 is small, A100/H100 are common high-end AI GPUs."
    )

    gpu_count = st.number_input(
        "GPU Count",
        min_value=1,
        max_value=4096,
        value=int(default["count"]),
        help="Total number of GPUs used. More GPUs usually increase total power draw."
    )

    batch_size = st.number_input(
        "Batch Size",
        min_value=1,
        max_value=8192,
        value=int(default["batch"]),
        help="Number of samples processed per training step. Larger batch sizes may increase hardware utilization."
    )

    hours = st.number_input(
        "Estimated Training Hours",
        min_value=0.1,
        max_value=10000.0,
        value=float(default["hours"]),
        help="Expected total training time. Example: small fine-tune = 1–5 hrs, larger training = days/weeks."
    )

    precision = st.selectbox(
        "Precision",
        ["FP32", "FP16", "BF16"],
        help="FP16/BF16 mixed precision usually reduces compute and memory usage compared to FP32."
    )

    submitted = st.form_submit_button("🚀 Run Forecast")

if submitted:

    precision_enc = {
        "FP32": 0,
        "FP16": 1,
        "BF16": 2
    }[precision]

    energy = predict_energy(params, tokens, gpu, gpu_count, batch_size, hours, precision_enc)
    cost = estimate_cost(energy)
    co2 = estimate_co2(energy)
    level = energy_level(energy)

    st.divider()

    st.subheader("Forecast Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("Predicted Energy", f"{energy:,.2f} kWh", level)
    col2.metric("Estimated Cost", f"${cost:,.2f}")
    col3.metric("Estimated CO₂", f"{co2:,.2f} kg")

    st.plotly_chart(gauge_chart(energy), use_container_width=True)

    st.info(interpretation_line(energy))

    st.subheader("Benchmark Comparison")

    benchmark_df = pd.DataFrame([
        {"Model": "DistilBERT preset", "Energy kWh": predict_energy(66_000_000, 1_000_000_000, "T4", 1, 32, 3, precision_enc)},
        {"Model": "BERT-base preset", "Energy kWh": predict_energy(110_000_000, 3_400_000_000, "A100", 4, 64, 24, precision_enc)},
        {"Model": "GPT-2 preset", "Energy kWh": predict_energy(124_000_000, 40_000_000_000, "A100", 8, 128, 72, precision_enc)},
        {"Model": "Your configuration", "Energy kWh": energy}
    ])

    fig = px.bar(
        benchmark_df,
        x="Model",
        y="Energy kWh",
        title="Your configuration vs common presets",
        text_auto=".2f"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Energy Saving Recommendations")

    recommendations = []

    if precision == "FP32":
        new_energy = energy * 0.62
        recommendations.append({
            "Action": "Switch to FP16/BF16 mixed precision",
            "Estimated Energy": new_energy,
            "Energy Saving": energy - new_energy,
            "Saving %": ((energy - new_energy) / energy) * 100
        })

    if gpu_count > 1:
        new_count = max(1, int(gpu_count // 2))
        new_energy = energy * 0.55
        recommendations.append({
            "Action": f"Reduce GPU count ({gpu_count} → {new_count})",
            "Estimated Energy": new_energy,
            "Energy Saving": energy - new_energy,
            "Saving %": ((energy - new_energy) / energy) * 100
        })

    new_energy = energy * 0.80
    recommendations.append({
        "Action": "Reduce training tokens by 20%",
        "Estimated Energy": new_energy,
        "Energy Saving": energy - new_energy,
        "Saving %": ((energy - new_energy) / energy) * 100
    })

    rec_df = pd.DataFrame(recommendations)

    display_rec_df = rec_df.copy()
    display_rec_df["Estimated Energy"] = display_rec_df["Estimated Energy"].map(lambda x: f"{x:,.2f} kWh")
    display_rec_df["Energy Saving"] = display_rec_df["Energy Saving"].map(lambda x: f"{x:,.2f} kWh")
    display_rec_df["Saving %"] = display_rec_df["Saving %"].map(lambda x: f"{x:.2f}%")

    st.dataframe(display_rec_df, use_container_width=True)

    report_df = pd.DataFrame([{
        "preset": preset,
        "params": params,
        "tokens": tokens,
        "gpu": gpu,
        "gpu_count": gpu_count,
        "batch_size": batch_size,
        "hours": hours,
        "precision": precision,
        "predicted_energy_kwh": energy,
        "estimated_cost_usd": cost,
        "estimated_co2_kg": co2,
        "energy_level": level
    }])

    st.subheader("Export Report")

    st.download_button(
        "Download Forecast Report CSV",
        data=report_df.to_csv(index=False),
        file_name="forecast_result.csv",
        mime="text/csv"
    )

    st.download_button(
        "Download Recommendations CSV",
        data=rec_df.to_csv(index=False),
        file_name="energy_recommendations.csv",
        mime="text/csv"
    )

else:
    st.info("Choose a Quick Start preset or enter your own configuration, then click **Run Forecast**.")