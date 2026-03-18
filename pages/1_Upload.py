from __future__ import annotations

import streamlit as st

from utils.session import compute_file_hash, initialize_state, reset_app, set_dataset
from utils.validators import load_csv_bytes, validate_uploaded_file

st.set_page_config(page_title="CDAS • Upload", layout="wide")
initialize_state()

with st.sidebar:
    st.header("CDAS Control")
    mode = st.radio("Mode", ["Learning", "Expert"], horizontal=True)
    st.session_state["ui_mode"] = mode
    if st.button("Reset App"):
        reset_app()
        st.success("State cleared.")
        st.stop()

st.title("🧠 CDAS — 1) Upload")
st.caption("Upload dataset and initialize cognitive analysis context.")
key_insight = "No dataset loaded yet." if st.session_state.get("data") is None else "Dataset active in memory."
st.info(f"Key Insight: {key_insight}")
st.divider()

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    file_bytes = uploaded.getvalue()
    validation = validate_uploaded_file(uploaded.name, file_bytes)
    if not validation.is_valid:
        st.error(validation.message)
        st.stop()

    df = load_csv_bytes(file_bytes)
    set_dataset(df, compute_file_hash(file_bytes))
    st.success("Dataset uploaded successfully.")

if st.session_state.get("data") is not None:
    df = st.session_state["data"]
    st.write(f"Rows: {len(df)} | Columns: {df.shape[1]}")
    with st.expander("Dataset Preview", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)
