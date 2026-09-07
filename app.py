import streamlit as st

from src.ui import comparison, disclosure, setup

setup("Plan compute. Understand energy.")
st.caption("RESPONSIBLE AI / COMPUTE PLANNING")
st.title("Understand energy before the run.")
st.markdown(
    "Explore training configurations, compare energy tradeoffs, and inspect the evidence behind every estimate."
)
disclosure()
results = comparison()
corrected = results[results.version == "v2_no_duration"].set_index("evaluation")
a, b, c = st.columns(3)
a.metric(
    "Synthetic holdout R² · log₁₀",
    f"{corrected.loc['synthetic_holdout', 'r2_log10']:.4f}",
)
b.metric(
    "External holdout R² · log₁₀", f"{corrected.loc['external_all', 'r2_log10']:.4f}"
)
c.metric("External labeled rows", int(corrected.loc["external_all", "n"]))
st.caption(
    "Both scores are shown deliberately: learning a simulator does not establish real-world accuracy."
)
st.divider()
left, right = st.columns(2, gap="large")
with left:
    st.subheader("01 / Forecast a training plan")
    st.write(
        "Enter model size, tokens, and hardware. See a model-derived energy estimate, a synthetic calibration interval, and exact TreeSHAP drivers."
    )
    st.page_link("pages/1_Forecast.py", label="Open Forecast →", icon="⚡")
    st.subheader("02 / Explore a budget")
    st.write(
        "Compare candidate model sizes at a fixed token budget. Inspect which simulated plans fit, with uncertainty kept visible."
    )
    st.page_link("pages/2_Budget_Mode.py", label="Open Budget Explorer →", icon="🧭")
with right:
    st.subheader("03 / Inspect the audit")
    st.write(
        "Follow the duration-leakage correction, source-level evaluation, archived model comparison, and reproducible split manifest."
    )
    st.page_link(
        "pages/3_Model_Info.py", label="Open Evidence & Model Card →", icon="🔬"
    )
    st.markdown("**What makes this useful**")
    st.write(
        "The interface distinguishes simulated behavior from measured evidence. Scenario changes are recomputed by the model; explanations use local feature contributions; exported reports retain assumptions and limitations."
    )
