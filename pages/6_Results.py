from __future__ import annotations

import streamlit as st

from app.ui_components import render_insight_banners, render_page_header, render_sidebar
from core.intelligence_engine import detect_high_correlation
from utils.session import initialize_state, reset_app

st.set_page_config(page_title="Results", layout="wide")
initialize_state()

with st.sidebar:
    render_sidebar("results")
    if st.button("Reset App", use_container_width=True):
        reset_app()
        st.success("Session reset complete.")
        st.stop()

render_page_header("📈", "Results", "Review model outcomes and processing impact.")

if st.session_state.get("data") is None:
    st.warning("No dataset found. Complete Data Upload first.")
    st.stop()

payload = st.session_state.get("model_payload")
if payload:
    render_insight_banners([f"Model {payload['metric'].upper()} score: {payload['score']:.4f}"])
else:
    render_insight_banners(["No model result available yet. Complete Modeling page."])

st.info("Model performance depends on preprocessing and feature selection quality.")

done = []
if st.session_state.get("cleaned_data") is not None:
    done.append("✅ Data Cleaning completed")
if st.session_state.get("engineered_data") is not None:
    done.append("✅ Feature Engineering completed")
if payload is not None:
    done.append(f"✅ Modeling completed ({payload['problem']})")
for line in done or ["No completed actions recorded."]:
    st.write(line)

with st.expander("Additional warnings", expanded=False):
    df = st.session_state.get("engineered_data") or st.session_state["data"]
    missing = int(df.isna().sum().sum())
    high_corr = detect_high_correlation(df)
    if missing > 0:
        st.warning(f"Missing values remaining: {missing}")
    if high_corr:
        st.warning("Strong feature correlation remains. Consider feature reduction.")
    if df.shape[1] > 50:
        st.warning("High dimensionality remains. Consider feature selection.")
    if missing == 0 and not high_corr and df.shape[1] <= 50:
        st.success("No major quality risks detected in final dataset.")
