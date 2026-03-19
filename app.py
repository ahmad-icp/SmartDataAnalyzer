from __future__ import annotations

import streamlit as st

from utils.session import initialize_state, reset_app

st.set_page_config(page_title="Smart Data Analyzer", layout="wide")
initialize_state()

with st.sidebar:
    st.header("📂 Control Panel")
    if st.button("Reset App"):
        reset_app()
        st.success("Application state cleared.")
        st.stop()
    st.info("Navigate pages 1→6 for the full workflow.")

st.header("📊 Smart Data Analyzer")
st.caption("Professional multi-page intelligent data analysis app (API-free).")

if st.session_state.get("data") is None:
    st.warning("No dataset loaded. Start from **1_Data_Upload**.")
else:
    df = st.session_state["data"]
    st.success(f"Active dataset: {df.shape[0]} rows × {df.shape[1]} columns")

with st.expander("Workflow", expanded=False):
    st.markdown(
        """
1. Data Upload
2. EDA
3. Data Cleaning
4. Feature Engineering
5. Modeling
6. Results
"""
    )
