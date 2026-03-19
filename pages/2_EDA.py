from __future__ import annotations

import streamlit as st

from app.ui_components import render_insight_banners, render_page_header, render_sidebar, render_stat_cards
from core.eda import correlation_matrix, dataset_overview, missing_values_summary
from core.intelligence_engine import detect_high_correlation
from utils.session import initialize_state, reset_app
from utils.visualization import correlation_heatmap

st.set_page_config(page_title="EDA", layout="wide")
initialize_state()

with st.sidebar:
    render_sidebar("eda")
    if st.button("Reset App", use_container_width=True):
        reset_app()
        st.success("Session reset complete.")
        st.stop()

render_page_header("📊", "Exploratory Data Analysis", "Understand dataset structure and detect quality risks.")

if st.session_state.get("data") is None:
    st.warning("No dataset found. Go to Data Upload first.")
    st.stop()

df = st.session_state["data"]
overview = dataset_overview(df)
missing_tbl = missing_values_summary(df)
corr = correlation_matrix(df)
high_corr = detect_high_correlation(df)
cat_count = len(df.select_dtypes(exclude="number").columns)

insights: list[str] = []
if high_corr:
    insights.append("🔥 Strong correlation detected between one or more feature pairs.")
if cat_count > 0:
    insights.append(f"ℹ️ Categorical columns detected: {cat_count}. Encoding will be required for modeling.")
if df.shape[1] > 50:
    insights.append("⚠ Dataset has many features. Consider feature selection.")
if not insights:
    insights.append("✅ No major structural warnings detected in EDA.")
render_insight_banners(insights)

render_stat_cards(
    [
        ("📏", "Rows", str(overview["shape"][0])),
        ("🧱", "Columns", str(overview["shape"][1])),
        ("🏷️", "Categorical", str(cat_count)),
    ]
)

with st.expander("Summary statistics", expanded=False):
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

with st.expander("Missing values", expanded=False):
    st.dataframe(missing_tbl, use_container_width=True)

with st.expander("Correlation matrix", expanded=False):
    if corr.empty:
        st.info("No numeric columns available for correlation matrix.")
    else:
        st.plotly_chart(correlation_heatmap(corr), use_container_width=True)
        st.write("High-correlation pairs:", high_corr)
