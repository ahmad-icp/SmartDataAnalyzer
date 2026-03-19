from __future__ import annotations

import streamlit as st

from app.ui_components import render_page_header, render_sidebar
from utils.session import initialize_state, reset_app

st.set_page_config(page_title="Smart Data Analyzer", layout="wide")
initialize_state()

with st.sidebar:
    render_sidebar("upload")
    if st.button("Reset App", use_container_width=True):
        reset_app()
        st.success("Session reset complete.")
        st.stop()

render_page_header(
    icon="📊",
    title="Smart Data Analyzer",
    description="Modern multi-page analytics workspace for intelligent dataset diagnostics and modeling.",
)

if st.session_state.get("data") is None:
    st.info("No active dataset. Start from **📂 Data Upload** in the pages menu.")
else:
    df = st.session_state["data"]
    st.success(f"Active dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

with st.expander("Navigation Guide", expanded=False):
    st.markdown(
        """
- 📂 Data Upload
- 📊 EDA
- 🧹 Data Cleaning
- ⚙️ Feature Engineering
- 🤖 Modeling
- 📈 Results
"""
    )
