from __future__ import annotations

import io
import json

import streamlit as st

from core.cognitive.analyzer import analyze_dataset
from core.cognitive.reasoning import reason_from_diagnostics
from core.cognitive.scoring import dataset_complexity_index
from core.data_quality import compute_data_quality_score
from utils.session import initialize_state, reset_app

st.set_page_config(page_title="CDAS • Report Generator", layout="wide")
initialize_state()

with st.sidebar:
    st.header("CDAS Control")
    st.session_state["ui_mode"] = st.radio("Mode", ["Learning", "Expert"], horizontal=True)
    if st.button("Reset App"):
        reset_app()
        st.success("State cleared.")
        st.stop()

st.title("📝 CDAS — 7) Report Generator")
st.caption("Create downloadable research-style reports from current session artifacts.")

if st.session_state.get("engineered_data") is None:
    st.warning("No processed dataset found.")
    st.stop()

clean_df = st.session_state.get("cleaned_data", st.session_state["engineered_data"])
diag = analyze_dataset(clean_df, st.session_state.get("target_column"))
issues = reason_from_diagnostics(diag)
dci, dci_label = dataset_complexity_index(diag["missing_ratio"], diag["n_cols"], diag["correlation_max"], diag["target_balance"])
quality = compute_data_quality_score(clean_df)
model_payload = st.session_state.get("model_payload")

report = {
    "dataset_summary": {"rows": int(diag["n_rows"]), "columns": int(diag["n_cols"])},
    "dataset_complexity_index": {"score": dci, "label": dci_label},
    "data_health": quality,
    "issues": issues,
    "model_performance": model_payload if model_payload else "No model trained",
}

st.info("Key Insight: Report is generated from live cognitive diagnostics and model outcomes.")
st.divider()

with st.expander("Report preview", expanded=st.session_state.get("ui_mode") == "Learning"):
    st.json(report)

json_bytes = json.dumps(report, indent=2).encode("utf-8")
st.download_button("Download JSON Report", data=json_bytes, file_name="cdas_report.json", mime="application/json")

csv_bytes = clean_df.to_csv(index=False).encode("utf-8")
st.download_button("Download Processed CSV", data=csv_bytes, file_name="cdas_processed_data.csv", mime="text/csv")

excel_data = None
try:
    import openpyxl  # noqa: F401

    buffer = io.BytesIO()
    clean_df.to_excel(buffer, index=False)
    excel_data = buffer.getvalue()
except Exception:
    excel_data = None

if excel_data:
    st.download_button(
        "Download Processed Excel",
        data=excel_data,
        file_name="cdas_processed_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
