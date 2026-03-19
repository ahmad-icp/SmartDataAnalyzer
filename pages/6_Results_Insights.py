from __future__ import annotations

import streamlit as st

from core.cognitive.analyzer import analyze_dataset
from core.cognitive.reasoning import reason_from_diagnostics
from core.cognitive.scoring import dataset_complexity_index
from core.data_quality import compute_data_quality_score
from utils.session import initialize_state, reset_app

st.set_page_config(page_title="CDAS • Results & Insights", layout="wide")
initialize_state()

with st.sidebar:
    st.header("CDAS Control")
    st.session_state["ui_mode"] = st.radio("Mode", ["Learning", "Expert"], horizontal=True)
    if st.button("Reset App"):
        reset_app()
        st.success("State cleared.")
        st.stop()

st.title("📈 CDAS — 6) Results & Insights")
st.caption("Unified decision view with quality, complexity, and model outcomes.")

if st.session_state.get("engineered_data") is None:
    st.warning("Complete previous steps first.")
    st.stop()

df = st.session_state["engineered_data"]
clean_df = st.session_state.get("cleaned_data", df)
payload = st.session_state.get("model_payload")

diag = analyze_dataset(clean_df, st.session_state.get("target_column"))
issues = reason_from_diagnostics(diag)
dci, dci_label = dataset_complexity_index(diag["missing_ratio"], diag["n_cols"], diag["correlation_max"], diag["target_balance"])
quality = compute_data_quality_score(clean_df)

key = f"Quality {quality['score']}/100 | DCI {dci}/10 ({dci_label})"
st.info(f"Key Insight: {key}")
st.divider()

c1, c2, c3 = st.columns(3)
c1.metric("Data Quality", f"{quality['score']}/100")
c2.metric("Dataset Complexity", f"{dci}/10")
if payload:
    c3.metric(payload["metric"].upper(), f"{payload['score']:.4f}")
else:
    c3.info("No model run yet")

with st.expander("Structured issue reasoning", expanded=st.session_state.get("ui_mode") == "Learning"):
    for issue in issues:
        st.write(
            f"**{issue['issue']}** | Severity: {issue['severity']} | "
            f"Recommendation: {issue['recommendation']} | Confidence: {issue['confidence']:.2f}"
        )
