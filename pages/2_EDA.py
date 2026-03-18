"""EDA page."""

from __future__ import annotations

import streamlit as st

from core.eda import correlation_matrix, dataset_overview, missing_values_summary
from core.intelligence_engine import generate_intelligence_report
from utils.session import initialize_state, reset_app
from utils.visualization import correlation_heatmap

st.set_page_config(page_title="EDA", layout="wide")
initialize_state()

with st.sidebar:
    st.header("Control Panel")
    if st.button("Reset App"):
        reset_app()
        st.success("Application state cleared.")
        st.stop()
    st.progress(0.3)
    st.caption("Step 2/6")

st.header("2) EDA")
st.subheader("Exploratory Data Analysis with progressive disclosure")
st.divider()

if st.session_state.get("data") is None:
    st.warning("No dataset found. Complete Data Upload first.")
    st.stop()

df = st.session_state["data"]
overview = dataset_overview(df)
missing = missing_values_summary(df)
corr = correlation_matrix(df)

col1, col2, col3 = st.columns(3)
col1.metric("Rows", overview["shape"][0])
col2.metric("Columns", overview["shape"][1])
col3.metric("Missing Cells", int(df.isna().sum().sum()))

with st.expander("Column types", expanded=False):
    st.json(overview["dtypes"])

with st.expander("Summary statistics", expanded=False):
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

with st.expander("Missing values table", expanded=False):
    st.dataframe(missing, use_container_width=True)

with st.expander("Correlation matrix", expanded=False):
    if corr.empty:
        st.info("No numeric columns available for correlation matrix.")
    else:
        st.plotly_chart(correlation_heatmap(corr), use_container_width=True)

st.subheader("Intelligence Engine")
insights = generate_intelligence_report(df)
st.session_state["insights"] = insights
for item in insights:
    st.write(f"- {item}")
