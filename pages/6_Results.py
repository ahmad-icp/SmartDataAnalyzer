from __future__ import annotations

import streamlit as st

from core.intelligence_engine import detect_high_correlation
from utils.session import initialize_state, reset_app

st.set_page_config(page_title="Results", layout="wide")
initialize_state()

with st.sidebar:
    st.header("📂 Control Panel")
    if st.button("Reset App"):
        reset_app()
        st.success("Application state cleared.")
        st.stop()

st.header("📈 Results")
st.caption("Model outcome and workflow summary with interpretation.")

if st.session_state.get("data") is None:
    st.warning("No dataset found. Complete Data Upload first.")
    st.stop()

payload = st.session_state.get("model_payload")
if payload is None:
    st.warning("Key Insight: No model has been trained yet.")
else:
    st.success(f"Key Insight: Model {payload['metric'].upper()} = {payload['score']:.4f}")

st.info("Performance depends on data quality and preprocessing quality.")

summary_lines = []
if st.session_state.get("cleaned_data") is not None:
    summary_lines.append("✅ Data Cleaning applied")
if st.session_state.get("engineered_data") is not None:
    summary_lines.append("✅ Feature Engineering applied")
if payload is not None:
    summary_lines.append(f"✅ Modeling completed ({payload['problem']})")

if not summary_lines:
    summary_lines.append("No processing steps completed yet.")

for line in summary_lines:
    st.write(line)

with st.expander("Additional Intelligent Warnings", expanded=False):
    df = st.session_state.get("engineered_data") or st.session_state["data"]
    missing = int(df.isna().sum().sum())
    if missing > 0:
        st.warning(f"Missing values remain: {missing}")
    high_corr = detect_high_correlation(df)
    if high_corr:
        st.warning("High correlation still present. Consider feature reduction.")
    if df.shape[1] > 50:
        st.warning("Too many features detected. Consider feature selection.")
    if missing == 0 and not high_corr and df.shape[1] <= 50:
        st.success("No major data quality warnings detected.")
