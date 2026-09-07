import json

import pandas as pd
import streamlit as st

from src.inference import DEFAULT, GPU_TDP, LABELS, artifacts, forecast, scenarios
from src.ui import disclosure, setup

setup("Forecast")
st.caption("01 / PRE-RUN PLANNING")
st.title("Forecast a training plan")
disclosure()
presets = {
    "Small fine-tune": DEFAULT,
    "7B token experiment": {
        **DEFAULT,
        "n_params": 7_000_000_000,
        "estimated_tokens": 500_000_000,
        "gpu_count": 8,
    },
    "Pre-training simulation": {
        **DEFAULT,
        "n_params": 13_000_000_000,
        "estimated_tokens": 50_000_000_000,
        "gpu_count": 64,
        "train_type": "pre-train",
    },
}
preset = st.selectbox("Simulation preset", list(presets))
default = presets[preset]
_, meta = artifacts()
with st.form("forecast_form"):
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Workload")
        params = st.number_input(
            "Model parameters",
            min_value=1_000_000,
            value=default["n_params"],
            step=1_000_000,
        )
        tokens = st.number_input(
            "Training tokens",
            min_value=1_000_000,
            value=default["estimated_tokens"],
            step=1_000_000,
        )
        task = st.selectbox(
            "Task",
            meta["categories"]["task"],
            index=meta["categories"]["task"].index(default["task"]),
        )
        train_type = st.selectbox(
            "Training type",
            meta["categories"]["train_type"],
            index=meta["categories"]["train_type"].index(default["train_type"]),
        )
    with right:
        st.subheader("Compute & accounting assumptions")
        gpu = st.selectbox(
            "GPU simulation profile",
            list(GPU_TDP),
            index=2,
            help="Power assumptions come from this project’s generator, not measured hardware specifications.",
        )
        count = st.number_input(
            "GPU count", min_value=1, max_value=4096, value=default["gpu_count"]
        )
        batch = st.number_input(
            "Batch size", min_value=1, max_value=8192, value=default["batch_size"]
        )
        precision = st.selectbox(
            "Precision",
            meta["categories"]["precision"],
            index=meta["categories"]["precision"].index(default["precision"]),
        )
        rate = st.number_input(
            "Electricity rate ($/kWh)", min_value=0.0, value=0.15, step=0.01
        )
        carbon = st.number_input(
            "Carbon intensity (kg CO₂e/kWh)", min_value=0.0, value=0.45, step=0.01
        )
    submitted = st.form_submit_button("Estimate energy", type="primary")
if submitted:
    config = {
        "n_params": params,
        "estimated_tokens": tokens,
        "gpu_model": gpu,
        "gpu_count": count,
        "batch_size": batch,
        "precision": precision,
        "task": task,
        "train_type": train_type,
    }
    st.session_state["forecast_result"] = (
        config,
        rate,
        carbon,
        forecast(config),
        scenarios(config),
    )
if "forecast_result" in st.session_state:
    config, rate, carbon, result, variants = st.session_state["forecast_result"]
    st.divider()
    st.subheader("Your simulated estimate")
    st.caption(
        "Results reflect the last submitted configuration. Submit again after changing inputs."
    )
    a, b, c = st.columns(3)
    a.metric("Energy", f"{result['energy_kwh']:,.4f} kWh")
    b.metric("Electricity cost", f"${result['energy_kwh'] * rate:,.4f}")
    c.metric("Operational emissions", f"{result['energy_kwh'] * carbon:,.4f} kg CO₂e")
    st.caption(
        "Electricity cost excludes cloud rental, hardware, and other fees. Carbon intensity is your assumption; embodied emissions are excluded."
    )
    st.write(
        f"Synthetic 90% nominal prediction interval: **{result['lower_kwh']:,.4f}–{result['upper_kwh']:,.4f} kWh**."
    )
    st.caption(
        "Calibrated on separate synthetic rows. Observed coverage is reported in the model card; this interval does not guarantee real-world coverage."
    )
    for warning in result["warnings"]:
        st.warning(warning)
    st.subheader("Why this estimate?")
    st.write(result["explanation"])
    contributions = pd.DataFrame(
        {
            "Feature": [LABELS[c] for c in result["contributions"]],
            "Contribution (log₁₀ kWh)": list(result["contributions"].values()),
        }
    )
    st.bar_chart(contributions.set_index("Feature"), horizontal=True)
    st.caption(
        "Exact TreeSHAP, grouped by input field. Positive values raise the log-energy prediction relative to the model baseline. Correlated FLOPs, parameters, and tokens share attribution; these are not causal effects."
    )
    st.subheader("What changes with a different plan?")
    st.dataframe(variants, hide_index=True, width="stretch")
    st.caption(
        "Each row reruns the model. Fewer tokens change the workload; fewer GPUs may increase runtime. Quality, runtime, memory fit, and causal savings are not evaluated."
    )
    report = dict(
        configuration=config,
        electricity_rate=rate,
        carbon_intensity=carbon,
        **result,
        scenarios=variants.to_dict(orient="records"),
        limitation="Synthetic planning prototype; external validation failed. Intervals calibrated on synthetic data only.",
    )
    st.download_button(
        "Download evidence report",
        json.dumps(report, indent=2),
        file_name="energy_forecast.json",
        mime="application/json",
    )
    st.download_button(
        "Download scenario CSV",
        variants.to_csv(index=False),
        file_name="energy_scenarios.csv",
        mime="text/csv",
    )
