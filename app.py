"""Smart Data Analyzer multi-page landing page."""

from __future__ import annotations

import streamlit as st

from utils.session import initialize_state, reset_app

st.set_page_config(page_title="Smart Data Analyzer", layout="wide")
initialize_state()

with st.sidebar:
    st.header("Control Panel")
    st.caption("Use pages in the left navigation to move through the workflow.")
    if st.button("Reset App"):
        reset_app()
        st.success("Application state cleared.")
        st.stop()
    st.progress(0.1)
    st.info("Workflow: Upload → EDA → Cleaning → Feature Engineering → Modeling → Results")

st.title("📊 Smart Data Analyzer")
st.subheader("Production-ready, user-controlled, AI-free intelligence platform")
st.divider()

st.markdown(
    """
Welcome to **Smart Data Analyzer**.

This multi-page app provides:
- Dataset upload and validation
- Rule-based intelligence insights
- Guided EDA and cleaning
- Feature engineering
- Manual/automatic modeling controls
- Results and downloadable outputs

Use the left sidebar page navigation to begin with **Data Upload**.
"""
)

with st.expander("How to use", expanded=False):
    st.markdown(
        """
1. Open **Data Upload** and upload a CSV.
2. Review diagnostics in **EDA**.
3. Apply transformations in **Data Cleaning** and **Feature Engineering**.
4. Train models in **Modeling**.
5. Review metrics and download output in **Results**.
"""
    )
