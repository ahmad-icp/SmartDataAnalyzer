"""Data cleaning page."""

from __future__ import annotations

import streamlit as st

from core.cleaning import correct_data_types
from core.correction import suggest_corrections
from utils.session import initialize_state, reset_app

st.set_page_config(page_title="Data Cleaning", layout="wide")
initialize_state()

with st.sidebar:
    st.header("Control Panel")
    if st.button("Reset App"):
        reset_app()
        st.success("Application state cleared.")
        st.stop()
    st.progress(0.45)
    st.caption("Step 3/6")

st.header("3) Data Cleaning")
st.subheader("User-controlled missing value and duplicate handling")
st.divider()

if st.session_state.get("data") is None:
    st.warning("No dataset found. Complete Data Upload first.")
    st.stop()

df = st.session_state.get("cleaned_data")
if df is None:
    df = st.session_state["data"].copy()

st.write("Select columns and cleaning actions.")
selected_cols = st.multiselect("Columns", options=list(df.columns), default=list(df.columns))
method = st.selectbox("Missing value strategy", ["drop", "mean", "median", "mode"])
remove_dupes = st.checkbox("Remove duplicates", value=True)

if st.button("Apply Cleaning"):
    out = df.copy()
    cols = [c for c in selected_cols if c in out.columns]
    if cols:
        if method == "drop":
            out = out.dropna(subset=cols)
        else:
            for col in cols:
                if out[col].isna().sum() == 0:
                    continue
                if method == "mean" and out[col].dtype != "object":
                    out[col] = out[col].fillna(out[col].mean())
                elif method == "median" and out[col].dtype != "object":
                    out[col] = out[col].fillna(out[col].median())
                else:
                    mode = out[col].mode(dropna=True)
                    if not mode.empty:
                        out[col] = out[col].fillna(mode.iloc[0])

    if remove_dupes:
        out = out.drop_duplicates().reset_index(drop=True)

    out = correct_data_types(out)
    st.session_state["cleaned_data"] = out
    st.session_state["engineered_data"] = out.copy()
    st.success("Cleaning applied successfully.")

with st.expander("Current cleaned dataset preview", expanded=False):
    st.dataframe(st.session_state.get("cleaned_data", df).head(50), use_container_width=True)

st.divider()
st.subheader("Categorical Correction (Fuzzy Matching)")
working = st.session_state.get("cleaned_data", df).copy()
cat_cols = working.select_dtypes(include="object").columns.tolist()
if not cat_cols:
    st.info("No categorical columns available for correction.")
else:
    corr_col = st.selectbox("Column for correction", cat_cols)
    groups = suggest_corrections(working[corr_col])
    if not groups:
        st.info("No fuzzy correction groups detected.")
    else:
        edited: list[dict[str, object]] = []
        for idx, group in enumerate(groups):
            st.write(f"Detected: {group['original']}")
            canonical = st.text_input(f"Canonical value group {idx + 1}", value=str(group["suggested"]))
            edited.append({"original": group["original"], "suggested": canonical})
        if st.button("Apply Corrections"):
            corrected = working.copy()
            for item in edited:
                corrected[corr_col] = corrected[corr_col].replace(item["original"], item["suggested"])
            st.session_state["cleaned_data"] = corrected
            st.session_state["engineered_data"] = corrected.copy()
            st.success("Corrections applied.")
