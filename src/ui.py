"""Shared presentation and evidence disclosure."""

import pandas as pd
import streamlit as st

from src.final_feature_engineering import ROOT
from src.inference import artifacts


def setup(title, icon="⚡"):
    st.set_page_config(
        page_title=title + " · AI Energy Forecaster", page_icon=icon, layout="wide"
    )
    st.markdown(
        """<style>
    .stApp {background:radial-gradient(ellipse at 95% 0%,#163746 0%,#0b1220 48%);}
    .block-container {max-width:1200px;padding-top:2.5rem;}
    h1,h2,h3 {letter-spacing:-.035em;}
    [data-testid="stMetric"] {background:#142333;border:1px solid #294152;border-radius:14px;padding:18px;}
    [data-testid="stSidebar"] {background:#0d1928;}
    </style>""",
        unsafe_allow_html=True,
    )
    st.sidebar.caption("AI ENERGY FORECASTER / V2")
    st.sidebar.markdown("**Plan compute.\nUnderstand the evidence.**")
    st.sidebar.caption("CPU-trained XGBoost · Pre-run inputs · Local TreeSHAP")


def disclosure():
    _, meta = artifacts()
    n = sum(meta["source_counts"].values())
    share = 100 * meta["source_counts"]["synthetic"] / n
    st.info(
        f"Simulation-based planning prototype · {share:.2f}% of usable data is physics-based synthetic data. External validation failed; estimates are not validated training budgets."
    )


def comparison():
    return pd.read_csv(ROOT / "reports/model_comparison.csv")
