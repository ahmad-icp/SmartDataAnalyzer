from __future__ import annotations

import streamlit as st

from core.feature_engineering import encode_features, scale_numeric_features, suggest_highly_correlated_features
from utils.session import initialize_state, reset_app

st.set_page_config(page_title="Feature Engineering", layout="wide")
initialize_state()

with st.sidebar:
    st.header("📂 Control Panel")
    if st.button("Reset App"):
        reset_app()
        st.success("Application state cleared.")
        st.stop()

st.header("⚙️ Feature Engineering")
st.caption("Transform features for robust model training.")

if st.session_state.get("cleaned_data") is None:
    st.warning("No cleaned data found. Complete Data Cleaning first.")
    st.stop()

df = st.session_state["cleaned_data"].copy()

if df.shape[1] > 50:
    st.warning("Key Insight: High dimensionality detected. Consider feature selection.")
else:
    st.info("Key Insight: Feature count is manageable.")

apply_onehot = st.checkbox("Apply One-Hot Encoding", value=True)
apply_label = st.checkbox("Apply Label Encoding", value=False)
scaler = st.selectbox("Scaling", ["None", "StandardScaler", "MinMax"])

if st.button("Apply Feature Engineering"):
    out = df.copy()
    if apply_onehot:
        out = encode_features(out)
    if apply_label:
        for c in out.select_dtypes(include="object").columns:
            out[c] = out[c].factorize()[0]
    if scaler == "StandardScaler":
        out = scale_numeric_features(out)
    elif scaler == "MinMax":
        num_cols = out.select_dtypes(include="number").columns
        for c in num_cols:
            cmin, cmax = out[c].min(), out[c].max()
            out[c] = 0.0 if cmax == cmin else (out[c] - cmin) / (cmax - cmin)

    st.session_state["engineered_data"] = out
    st.success("Feature engineering applied.")

with st.expander("Feature Diagnostics", expanded=False):
    st.write("High correlation pairs:", suggest_highly_correlated_features(df))
    st.dataframe(st.session_state.get("engineered_data", df).head(100), use_container_width=True)
