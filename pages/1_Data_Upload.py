from __future__ import annotations

import streamlit as st

from utils.session import compute_file_hash, initialize_state, reset_app, set_dataset
from utils.validators import load_csv_bytes, validate_uploaded_file

st.set_page_config(page_title="Data Upload", layout="wide")
initialize_state()

with st.sidebar:
    st.header("📂 Control Panel")
    if st.button("Reset App"):
        reset_app()
        st.success("Application state cleared.")
        st.stop()

st.header("📂 Data Upload")
st.caption("Upload a CSV dataset to start the analysis pipeline.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is None:
    st.info("Key Insight: Upload a file to unlock EDA and modeling pages.")
    st.stop()

raw_bytes = uploaded.getvalue()
validation = validate_uploaded_file(uploaded.name, raw_bytes)
if not validation.is_valid:
    st.error(validation.message)
    st.stop()

df = load_csv_bytes(raw_bytes)
set_dataset(df, compute_file_hash(raw_bytes))

missing = int(df.isna().sum().sum())
if missing > 0:
    st.warning(f"Key Insight: Dataset has {missing} missing values.")
else:
    st.success("Key Insight: No missing values detected.")

st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

with st.expander("Dataset Preview", expanded=False):
    st.dataframe(df.head(100), use_container_width=True)
