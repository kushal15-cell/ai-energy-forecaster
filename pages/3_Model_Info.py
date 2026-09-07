import json

import pandas as pd
import streamlit as st

from src.final_feature_engineering import ROOT
from src.inference import artifacts
from src.ui import comparison, disclosure, setup

setup("Evidence & Model Card", "🔬")
st.caption("03 / MODEL AUDIT")
st.title("Evidence before accuracy claims")
disclosure()
_, meta = artifacts()
st.subheader("Where the data comes from")
st.dataframe(
    pd.DataFrame(
        [
            [
                "Synthetic",
                12000,
                12000,
                "Physics-inspired simulation; heuristic runtime, noise, and overhead",
            ],
            [
                "Hugging Face",
                168,
                168,
                "Model metadata with heuristic energy labels; not measured training energy",
            ],
            [
                "MLPerf",
                91,
                66,
                "Minimum reported latency × assumed accelerator TDP; incomplete workload metadata",
            ],
            [
                "CodeCarbon",
                1,
                1,
                "One instrumented CPU RandomForest run; not a GPU training benchmark",
            ],
        ],
        columns=["Source", "Raw rows", "Usable labels", "What the label means"],
    ),
    hide_index=True,
    width="stretch",
)
st.write(
    "12,260 raw rows; 12,235 usable positive energy labels. Synthetic share: 97.88% raw, 98.08% usable. All 235 usable external rows are reserved for evaluation. The 25 other MLPerf rows have no usable energy label."
)
st.subheader("Before / after / baseline")
results = comparison()
st.dataframe(results, hide_index=True, width="stretch")
st.caption(
    "R² and RMSE with suffix log10 use log₁₀(kWh). MAE is in kWh; r2_kwh uses the original scale. Single-row R² is undefined. The archived score is replayed on the original blended split; it is not directly comparable to the new external holdout."
)
st.write(
    "The corrected model fits the synthetic generator well but fails external validation. External log-energy performance is worse than the synthetic-training median baseline. Missing workload fields, estimated labels, and domain mismatch prevent a real-world accuracy claim."
)
st.subheader("What changed")
st.markdown(
    "- Removed duration and outcome-derived inputs. Source and label quality are evaluation metadata only.\n- Fit imputation and categorical encoding on synthetic training rows only; inference uses the saved pipeline.\n- Reserved 8,400 synthetic rows for training, 1,200 for calibration, and 2,400 for testing.\n- Archived the old model and saved split IDs, source metrics, predictions, hashes, and package versions.\n- Replaced hard-coded results, confidence bands, and savings with actual model outputs."
)
st.subheader("Uncertainty under distribution shift")
coverage = results[
    (results.version == "v2_no_duration")
    & results.evaluation.isin(["synthetic_holdout", "external_all"])
][["evaluation", "interval_coverage"]]
st.dataframe(coverage, hide_index=True)
st.write(
    "Split conformal intervals target 90% coverage under exchangeability with synthetic calibration data. Their external coverage is poor. They quantify simulator residuals, not missing real-world uncertainty."
)
st.subheader("Global feature attribution")
for name in ["shap_bar.png", "shap_summary.png"]:
    path = ROOT / "models" / name
    if path.exists():
        st.image(str(path), width="stretch")
st.caption(
    "Generated from corrected-model synthetic holdout rows. SHAP explains predictions, not causation or validation. Plain-language notes are deterministic and require no API key."
)
with st.expander("Reproducibility manifest"):
    st.json(meta)
st.download_button(
    "Download model manifest",
    json.dumps(meta, indent=2),
    "model_manifest.json",
    "application/json",
)
st.download_button(
    "Download evaluation table",
    results.to_csv(index=False),
    "model_comparison.csv",
    "text/csv",
)
