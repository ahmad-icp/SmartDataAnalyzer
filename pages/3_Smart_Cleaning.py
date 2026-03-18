from __future__ import annotations

import streamlit as st

from core.cleaning import correct_data_types
from core.cognitive.recommendations import adaptive_pipeline, error_simulation
from core.correction import suggest_corrections
from utils.session import initialize_state, reset_app

st.set_page_config(page_title="CDAS • Smart Cleaning", layout="wide")
initialize_state()

with st.sidebar:
    st.header("CDAS Control")
    st.session_state["ui_mode"] = st.radio("Mode", ["Learning", "Expert"], horizontal=True)
    if st.button("Reset App"):
        reset_app()
        st.success("State cleared.")
        st.stop()

st.title("⚠️ CDAS — 3) Smart Cleaning")
st.caption("Adaptive preprocessing actions with one-click pipeline execution.")

if st.session_state.get("data") is None:
    st.warning("Upload data first.")
    st.stop()

base_df = st.session_state.get("cleaned_data") or st.session_state["data"]
issues = st.session_state.get("issues", [])
pipeline = adaptive_pipeline(issues)
st.info("Key Insight: " + (pipeline[0] if pipeline else "No preprocessing actions required."))
st.divider()

if st.button("Run Adaptive Pipeline"):
    out = base_df.copy()
    if "Impute missing values" in pipeline:
        for col in out.columns:
            if out[col].isna().sum() == 0:
                continue
            if out[col].dtype == "object":
                mode = out[col].mode(dropna=True)
                if not mode.empty:
                    out[col] = out[col].fillna(mode.iloc[0])
            else:
                out[col] = out[col].fillna(out[col].median())
    if "Remove duplicate rows" in pipeline:
        out = out.drop_duplicates().reset_index(drop=True)

    out = correct_data_types(out)
    st.session_state["cleaned_data"] = out
    st.session_state["engineered_data"] = out.copy()
    st.success("Adaptive cleaning pipeline applied.")

st.subheader("Editable Fuzzy Correction")
working = st.session_state.get("cleaned_data", base_df).copy()
cat_cols = working.select_dtypes(include="object").columns.tolist()
if not cat_cols:
    st.info("No categorical columns found.")
else:
    col = st.selectbox("Categorical column", cat_cols)
    groups = suggest_corrections(working[col])
    if groups:
        edits: list[dict[str, object]] = []
        for i, group in enumerate(groups):
            canonical = st.text_input(f"Canonical value #{i+1}", value=str(group["suggested"]))
            edits.append({"original": group["original"], "suggested": canonical})
        if st.button("Apply Fuzzy Corrections"):
            for item in edits:
                working[col] = working[col].replace(item["original"], item["suggested"])
            st.session_state["cleaned_data"] = working
            st.session_state["engineered_data"] = working.copy()
            st.success("Fuzzy corrections applied.")

with st.expander("Error Simulation (if recommendations ignored)", expanded=st.session_state.get("ui_mode") == "Learning"):
    for line in error_simulation(issues):
        st.write(f"- {line}")
