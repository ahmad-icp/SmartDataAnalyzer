"""Feature engineering page."""

from __future__ import annotations

import streamlit as st

from core.feature_engineering import encode_features, scale_numeric_features, suggest_highly_correlated_features
from utils.session import initialize_state, reset_app

st.set_page_config(page_title="Feature Engineering", layout="wide")
initialize_state()

with st.sidebar:
    st.header("Control Panel")
    if st.button("Reset App"):
        reset_app()
        st.success("Application state cleared.")
        st.stop()
    st.progress(0.6)
    st.caption("Step 4/6")

st.header("4) Feature Engineering")
st.subheader("Apply encoding and scaling with user controls")
st.divider()

if st.session_state.get("cleaned_data") is None:
    st.warning("Complete Data Cleaning first.")
    st.stop()

df = st.session_state["cleaned_data"].copy()

corr_pairs = suggest_highly_correlated_features(df)
with st.expander("Feature diagnostics", expanded=False):
    st.write("Highly correlated feature pairs:", corr_pairs)

apply_ohe = st.checkbox("Apply One-Hot Encoding", value=True)
apply_label = st.checkbox("Apply Label Encoding (factorize object columns)", value=False)
scaler = st.selectbox("Scaling method", ["None", "StandardScaler", "MinMaxScaler"])

def _apply_label_encoding(in_df):
    out = in_df.copy()
    for col in out.select_dtypes(include="object").columns:
        out[col] = out[col].factorize()[0]
    return out


def _apply_minmax(in_df):
    out = in_df.copy()
    num_cols = out.select_dtypes(include="number").columns
    for col in num_cols:
        min_val = out[col].min()
        max_val = out[col].max()
        if max_val == min_val:
            out[col] = 0.0
        else:
            out[col] = (out[col] - min_val) / (max_val - min_val)
    return out

if st.button("Apply Feature Engineering"):
    engineered = df.copy()
    if apply_ohe:
        engineered = encode_features(engineered)
    if apply_label:
        engineered = _apply_label_encoding(engineered)
    if scaler == "StandardScaler":
        engineered = scale_numeric_features(engineered)
    elif scaler == "MinMaxScaler":
        engineered = _apply_minmax(engineered)

    st.session_state["engineered_data"] = engineered
    st.success("Feature engineering complete.")

with st.expander("Engineered dataset preview", expanded=False):
    st.dataframe(st.session_state.get("engineered_data", df).head(50), use_container_width=True)
