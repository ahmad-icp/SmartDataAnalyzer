"""Results page."""

from __future__ import annotations

import io

import streamlit as st

from core.data_quality import compute_data_quality_score
from core.intelligence_engine import generate_intelligence_report
from utils.session import initialize_state, reset_app

st.set_page_config(page_title="Results", layout="wide")
initialize_state()

with st.sidebar:
    st.header("Control Panel")
    if st.button("Reset App"):
        reset_app()
        st.success("Application state cleared.")
        st.stop()
    st.progress(1.0)
    st.caption("Step 6/6")

st.header("6) Results")
st.subheader("Model performance, intelligence summary, and export")
st.divider()

if st.session_state.get("engineered_data") is None:
    st.warning("No processed dataset found. Complete earlier steps first.")
    st.stop()

df = st.session_state["engineered_data"]
quality = compute_data_quality_score(st.session_state.get("cleaned_data", df))

col1, col2 = st.columns(2)
with col1:
    st.metric("Data Quality Score", f"{quality['score']}/100")
with col2:
    payload = st.session_state.get("model_payload")
    if payload:
        st.metric(payload["metric_name"].upper(), f"{payload['score']:.4f}")
    else:
        st.info("No trained model yet.")

st.subheader("Insights Summary")
target = st.session_state.get("target_column")
insights = generate_intelligence_report(st.session_state.get("cleaned_data", df), target_column=target)
for line in insights:
    st.write(f"- {line}")

with st.expander("Detailed quality penalties", expanded=False):
    st.json(quality["penalties"])

st.divider()
st.subheader("Download Processed Dataset")

csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv_bytes, file_name="smart_data_analyzer_processed.csv", mime="text/csv")

excel_data = None
try:
    import openpyxl  # noqa: F401

    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    excel_data = buffer.getvalue()
except Exception:
    excel_data = None

if excel_data:
    st.download_button(
        "Download Excel",
        data=excel_data,
        file_name="smart_data_analyzer_processed.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
else:
    st.warning("Excel export unavailable. Install openpyxl to enable .xlsx download.")
