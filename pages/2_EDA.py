from __future__ import annotations

import streamlit as st

from core.eda import correlation_matrix, dataset_overview, missing_values_summary
from core.intelligence_engine import detect_high_correlation
from utils.session import initialize_state, reset_app
from utils.visualization import correlation_heatmap

st.set_page_config(page_title="EDA", layout="wide")
initialize_state()

with st.sidebar:
    st.header("📂 Control Panel")
    if st.button("Reset App"):
        reset_app()
        st.success("Application state cleared.")
        st.stop()

st.header("📊 Exploratory Data Analysis")
st.caption("Smart diagnostics for structure, quality, and correlation patterns.")

if st.session_state.get("data") is None:
    st.warning("No dataset found. Complete Data Upload first.")
    st.stop()

df = st.session_state["data"]
overview = dataset_overview(df)
missing_tbl = missing_values_summary(df)
corr = correlation_matrix(df)
high_corr = detect_high_correlation(df)

if high_corr:
    st.warning("Key Insight: Strong correlation detected between features.")
else:
    st.info("Key Insight: No strong correlation issues detected.")

c1, c2, c3 = st.columns(3)
c1.metric("Rows", overview["shape"][0])
c2.metric("Columns", overview["shape"][1])
c3.metric("Categorical Columns", len(df.select_dtypes(exclude="number").columns))

with st.expander("Summary Statistics", expanded=False):
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

with st.expander("Missing Values", expanded=False):
    st.dataframe(missing_tbl, use_container_width=True)

with st.expander("Correlation Matrix", expanded=False):
    if corr.empty:
        st.info("No numeric columns available for correlation matrix.")
    else:
        st.plotly_chart(correlation_heatmap(corr), use_container_width=True)
        st.write("High correlation pairs:", high_corr)
