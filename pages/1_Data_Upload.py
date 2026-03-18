"""Data upload page."""

from __future__ import annotations

import streamlit as st

from utils.session import compute_file_hash, initialize_state, reset_app, set_dataset
from utils.validators import load_csv_bytes, validate_uploaded_file

st.set_page_config(page_title="Data Upload", layout="wide")
initialize_state()

with st.sidebar:
    st.header("Control Panel")
    if st.button("Reset App"):
        reset_app()
        st.success("Application state cleared.")
        st.stop()
    st.progress(0.15)
    st.caption("Step 1/6")

st.header("1) Upload Data")
st.subheader("Upload your CSV dataset")
st.divider()

uploaded = st.file_uploader("Upload dataset", type=["csv"])
if uploaded is None:
    st.info("Please upload a CSV file to continue.")
    st.stop()

file_bytes = uploaded.getvalue()
validation = validate_uploaded_file(uploaded.name, file_bytes)
if not validation.is_valid:
    st.error(validation.message)
    st.stop()

try:
    df = load_csv_bytes(file_bytes)
    set_dataset(df, compute_file_hash(file_bytes))
    st.success("Dataset uploaded and stored in session state.")
except Exception as exc:
    st.error(f"Failed to load dataset: {exc}")
    st.stop()

st.write(f"Rows: {len(df)} | Columns: {df.shape[1]}")
with st.expander("Dataset preview", expanded=False):
    st.dataframe(df.head(50), use_container_width=True)
