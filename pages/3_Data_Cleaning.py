from __future__ import annotations

import streamlit as st

from app.ui_components import render_insight_banners, render_page_header, render_sidebar
from core.cleaning import correct_data_types
from utils.session import initialize_state, reset_app

st.set_page_config(page_title="Data Cleaning", layout="wide")
initialize_state()

with st.sidebar:
    render_sidebar("cleaning")
    if st.button("Reset App", use_container_width=True):
        reset_app()
        st.success("Session reset complete.")
        st.stop()

render_page_header("🧹", "Data Cleaning", "Clean missing values and duplicate rows with guided actions.")

if st.session_state.get("data") is None:
    st.warning("No dataset found. Go to Data Upload first.")
    st.stop()

df = st.session_state.get("cleaned_data")
if df is None:
    df = st.session_state["data"].copy()

num_cols = df.select_dtypes(include="number").columns.tolist()
if int(df.isna().sum().sum()) > 0:
    rec = "median" if num_cols else "mode"
    render_insight_banners([f"💡 Use {rec} imputation for better robustness on missing values."])
else:
    render_insight_banners(["✅ No missing values detected. Cleaning is optional."])

strategy = st.selectbox("Missing value strategy", ["mean", "median", "mode", "drop_rows"])
action_cols = st.multiselect("Columns to process", list(df.columns), default=list(df.columns))

c1, c2 = st.columns(2)
with c1:
    if st.button("Fill missing values", use_container_width=True):
        out = df.copy()
        for col in action_cols:
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
        st.success("Missing value action applied.")
with c2:
    if st.button("Drop duplicate rows", use_container_width=True):
        out = df.drop_duplicates().reset_index(drop=True)
        st.session_state["cleaned_data"] = out
        st.session_state["engineered_data"] = out.copy()
        st.success("Duplicate rows removed.")

with st.expander("Cleaned data preview", expanded=False):
    st.dataframe(st.session_state.get("cleaned_data", df).head(100), use_container_width=True)
