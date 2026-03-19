from __future__ import annotations

import streamlit as st

from app.ui_components import render_insight_banners, render_page_header, render_sidebar, render_stat_cards
from utils.session import compute_file_hash, initialize_state, reset_app, set_dataset
from utils.validators import load_csv_bytes, validate_uploaded_file

st.set_page_config(page_title="Data Upload", layout="wide")
initialize_state()

with st.sidebar:
    render_sidebar("upload")
    if st.button("Reset App", use_container_width=True):
        reset_app()
        st.success("Session reset complete.")
        st.stop()

render_page_header("📂", "Data Upload", "Upload and validate your dataset before analysis.")
st.info("Accepted format: CSV files only. Max size: 200MB.")

upload = st.file_uploader("Drag and drop CSV file", type=["csv"])
if upload is None:
    render_insight_banners(["Upload a dataset to unlock EDA, cleaning, feature engineering, and modeling."])
    st.stop()

raw = upload.getvalue()
validation = validate_uploaded_file(upload.name, raw)
if not validation.is_valid:
    st.error(validation.message)
    st.stop()

df = load_csv_bytes(raw)
set_dataset(df, compute_file_hash(raw))

missing_cells = int(df.isna().sum().sum())
missing_cols = int((df.isna().sum() > 0).sum())
file_mb = len(raw) / (1024 * 1024)
render_stat_cards(
    [
        ("📏", "Rows", str(df.shape[0])),
        ("🧱", "Columns", str(df.shape[1])),
        ("⚠️", "Missing", str(missing_cells)),
        ("💾", "Size (MB)", f"{file_mb:.2f}"),
    ]
)

if missing_cols > 0:
    st.warning(f"Missing values detected in {missing_cols} columns → Proceed to Data Cleaning.")
else:
    st.success("No missing value warnings detected.")

with st.expander("Dataset preview", expanded=False):
    st.dataframe(df.head(100), use_container_width=True)
