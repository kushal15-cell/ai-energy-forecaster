import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Model Info",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Info & Methodology")

st.write(
    "This page explains how the AI Training Energy Forecaster was built, "
    "what data sources it uses, and which features influence predictions."
)

st.divider()

st.subheader("Dataset Sources")

source_df = pd.DataFrame([
    {
        "Source": "MLPerf benchmark-derived rows",
        "Purpose": "Adds industry benchmark hardware/runtime patterns",
        "Label Type": "Estimated from latency + accelerator power"
    },
    {
        "Source": "CodeCarbon measured experiments",
        "Purpose": "Adds real measured local training energy/emissions",
        "Label Type": "Measured"
    },
    {
        "Source": "Hugging Face metadata",
        "Purpose": "Adds model architecture and task variety",
        "Label Type": "Estimated / metadata-derived"
    },
    {
        "Source": "Synthetic physics-based simulations",
        "Purpose": "Adds scale and coverage for many training configurations",
        "Label Type": "Simulated"
    }
])

st.dataframe(source_df, use_container_width=True)

st.divider()

st.subheader("Model Performance")

metrics_df = pd.DataFrame([{
    "Model": "XGBoost Regressor",
    "Target": "log_energy_kwh",
    "MAE": 0.0462,
    "RMSE": 0.0645,
    "R² Score": 0.9948
}])

st.dataframe(metrics_df, use_container_width=True)

st.info(
    "The high R² score is expected because part of the dataset is physics-based synthetic data. "
    "The model should be interpreted as a planning estimator, not an exact real-world power meter."
)

st.divider()

st.subheader("Feature Importance")

if os.path.exists("models/shap_bar.png"):
    st.image(
        "models/shap_bar.png",
        caption="SHAP feature importance from the trained XGBoost model"
    )
else:
    st.warning("SHAP image not found. Run src/shap_analysis.py first.")

st.divider()

st.subheader("Most Important Features")

feature_df = pd.DataFrame([
    {
        "Feature": "gpu_count",
        "Meaning": "Number of GPUs used",
        "Why it matters": "More GPUs usually increase total electricity consumption."
    },
    {
        "Feature": "tdp_w",
        "Meaning": "GPU power rating in watts",
        "Why it matters": "Higher-TDP GPUs consume more power."
    },
    {
        "Feature": "log_flops",
        "Meaning": "Log of estimated training compute",
        "Why it matters": "More compute operations require more energy."
    },
    {
        "Feature": "log_duration",
        "Meaning": "Log of training runtime",
        "Why it matters": "Longer training consumes more electricity."
    }
])

st.dataframe(feature_df, use_container_width=True)

st.divider()

st.subheader("Project Methodology")

st.write("""
The system follows this pipeline:

1. Collect data from MLPerf-style benchmark exports, CodeCarbon runs, Hugging Face metadata, and synthetic simulation.
2. Clean and standardize all sources into one unified dataset.
3. Create log-transformed features such as model size, FLOPs, tokens, and duration.
4. Train an XGBoost regression model to predict `log_energy_kwh`.
5. Convert the prediction back to kWh.
6. Estimate cost and CO₂ emissions.
7. Use SHAP to explain which features drive the prediction.
""")

st.subheader("Core Innovation")

st.success(
    "Most tools measure energy after training starts. "
    "This system predicts energy before GPU execution, helping users plan cost and sustainability impact proactively."
)