from __future__ import annotations

import streamlit as st

from app.ui_components import render_insight_banners, render_page_header, render_sidebar
from core.feature_engineering import encode_features, scale_numeric_features, suggest_highly_correlated_features
from utils.session import initialize_state, reset_app

st.set_page_config(page_title="Feature Engineering", layout="wide")
initialize_state()

with st.sidebar:
    render_sidebar("feature")
    if st.button("Reset App", use_container_width=True):
        reset_app()
        st.success("Session reset complete.")
        st.stop()

render_page_header("⚙️", "Feature Engineering", "Create model-ready features using encoding and scaling.")

if st.session_state.get("cleaned_data") is None:
    st.warning("No cleaned data found. Complete Data Cleaning first.")
    st.stop()

df = st.session_state["cleaned_data"].copy()
if df.shape[1] > 50:
    render_insight_banners(["⚠ High dimensionality detected. Consider feature selection due to many columns."])
else:
    render_insight_banners(["✅ Feature dimensionality is within a manageable range."])

apply_ohe = st.checkbox("Apply one-hot encoding", value=True)
apply_label = st.checkbox("Apply label encoding", value=False)
scaler = st.selectbox("Scaling", ["None", "StandardScaler", "MinMax"])

if st.button("Apply transformations"):
    out = df.copy()
    if apply_ohe:
        out = encode_features(out)
    if apply_label:
        for col in out.select_dtypes(include="object").columns:
            out[col] = out[col].factorize()[0]
    if scaler == "StandardScaler":
        out = scale_numeric_features(out)
    elif scaler == "MinMax":
        for col in out.select_dtypes(include="number").columns:
            mn, mx = out[col].min(), out[col].max()
            out[col] = 0.0 if mn == mx else (out[col] - mn) / (mx - mn)

    st.session_state["engineered_data"] = out
    st.success("Feature engineering complete.")

with st.expander("Feature diagnostics", expanded=False):
    st.write("High-correlation pairs:", suggest_highly_correlated_features(df))
    st.dataframe(st.session_state.get("engineered_data", df).head(100), use_container_width=True)
