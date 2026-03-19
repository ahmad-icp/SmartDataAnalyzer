from __future__ import annotations

import streamlit as st

from core.cleaning import correct_data_types
from utils.session import initialize_state, reset_app

st.set_page_config(page_title="Data Cleaning", layout="wide")
initialize_state()

with st.sidebar:
    st.header("📂 Control Panel")
    if st.button("Reset App"):
        reset_app()
        st.success("Application state cleared.")
        st.stop()

st.header("🧹 Data Cleaning")
st.caption("Apply cleaning actions with intelligent method recommendations.")

if st.session_state.get("data") is None:
    st.warning("No dataset found. Complete Data Upload first.")
    st.stop()

df = st.session_state.get("cleaned_data")
if df is None:
    df = st.session_state["data"].copy()

missing_ratio = df.isna().mean().sort_values(ascending=False)
if missing_ratio.max() > 0:
    rec = "median" if any(df[c].dtype != "object" for c in df.columns) else "mode"
    st.warning(f"Key Insight: Missing values detected. Recommended default strategy: {rec} imputation.")
else:
    st.success("Key Insight: Dataset currently has no missing values.")

strategy = st.selectbox("Missing value strategy", ["mean", "median", "mode", "drop_rows"])
selected_columns = st.multiselect("Columns to apply strategy", list(df.columns), default=list(df.columns))

col1, col2 = st.columns(2)
with col1:
    if st.button("Fill Missing Values"):
        out = df.copy()
        for col in selected_columns:
            if col not in out.columns or out[col].isna().sum() == 0:
                continue
            if strategy == "drop_rows":
                out = out.dropna(subset=[col])
            elif strategy == "mean" and out[col].dtype != "object":
                out[col] = out[col].fillna(out[col].mean())
            elif strategy == "median" and out[col].dtype != "object":
                out[col] = out[col].fillna(out[col].median())
            else:
                mode = out[col].mode(dropna=True)
                if not mode.empty:
                    out[col] = out[col].fillna(mode.iloc[0])
        out = correct_data_types(out)
        st.session_state["cleaned_data"] = out
        st.session_state["engineered_data"] = out.copy()
        st.success("Missing value action completed.")
with col2:
    if st.button("Drop Duplicate Rows"):
        out = df.drop_duplicates().reset_index(drop=True)
        st.session_state["cleaned_data"] = out
        st.session_state["engineered_data"] = out.copy()
        st.success("Duplicates removed.")

with st.expander("Cleaned Data Preview", expanded=False):
    st.dataframe(st.session_state.get("cleaned_data", df).head(100), use_container_width=True)
