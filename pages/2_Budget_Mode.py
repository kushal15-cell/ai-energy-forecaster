import pandas as pd
import streamlit as st

from src.inference import DEFAULT, GPU_TDP, forecast
from src.ui import disclosure, setup

setup("Budget Explorer", "🧭")
st.caption("02 / SCENARIO EXPLORATION")
st.title("Explore an energy budget")
disclosure()
st.write(
    "Compare model sizes while keeping tokens and hardware fixed. This is a simulation screen, not a claim that a model will fit GPU memory or achieve a target quality."
)
with st.form("budget"):
    a, b = st.columns(2)
    with a:
        budget = st.number_input(
            "Energy budget (kWh)", min_value=0.001, value=1.0, format="%.3f"
        )
        tokens = st.number_input(
            "Fixed training tokens",
            min_value=1_000_000,
            value=1_000_000_000,
            step=1_000_000,
        )
        gpu = st.selectbox("GPU simulation profile", list(GPU_TDP), index=2)
    with b:
        count = st.number_input("GPU count", min_value=1, max_value=256, value=4)
        precision = st.selectbox("Precision", ["fp16", "bf16", "fp32"])
        criterion = st.selectbox(
            "Budget screening rule",
            ["Synthetic interval upper bound", "Point estimate"],
        )
    submitted = st.form_submit_button("Compare candidate plans", type="primary")
if submitted:
    rows = []
    for params in [
        82_000_000,
        125_000_000,
        355_000_000,
        760_000_000,
        1_300_000_000,
        2_700_000_000,
        7_000_000_000,
        13_000_000_000,
        34_000_000_000,
        70_000_000_000,
    ]:
        result = forecast(
            {
                **DEFAULT,
                "n_params": params,
                "estimated_tokens": tokens,
                "gpu_model": gpu,
                "gpu_count": count,
                "precision": precision,
            },
            False,
        )
        check = (
            result["upper_kwh"]
            if criterion == "Synthetic interval upper bound"
            else result["energy_kwh"]
        )
        rows.append(
            {
                "parameters_billions": params / 1e9,
                "energy_kwh": result["energy_kwh"],
                "upper_kwh": result["upper_kwh"],
                "within_simulated_budget": check <= budget,
                "input_warning": " ".join(result["warnings"]),
            }
        )
    frame = pd.DataFrame(rows)
    fitting = frame[frame.within_simulated_budget & frame.input_warning.eq("")]
    if len(fitting):
        st.success(
            f"Largest screened candidate within training ranges: {fitting.parameters_billions.max():g}B parameters. Validate memory, runtime, quality, and measured energy separately."
        )
    else:
        st.warning(
            "No candidate within training ranges passes this simulated budget rule."
        )
    st.bar_chart(frame.set_index("parameters_billions")[["energy_kwh", "upper_kwh"]])
    st.dataframe(frame, hide_index=True, width="stretch")
    st.caption(
        "Fixed task: text-generation; training type: fine-tune; batch size: 64. Upper bounds are calibrated only on synthetic data."
    )
    export = frame.assign(
        tokens=tokens,
        gpu_model=gpu,
        gpu_count=count,
        precision=precision,
        budget_kwh=budget,
        screening_rule=criterion,
        limitation="Synthetic only; not a validated real-world budget",
    )
    st.download_button(
        "Download budget comparison",
        export.to_csv(index=False),
        "budget_comparison.csv",
        "text/csv",
    )
