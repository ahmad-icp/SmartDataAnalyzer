from __future__ import annotations

import streamlit as st

from core.cognitive.analyzer import analyze_dataset
from core.cognitive.reasoning import reason_from_diagnostics
from core.cognitive.scoring import dataset_complexity_index
from utils.session import initialize_state, reset_app

st.set_page_config(page_title="CDAS • Diagnosis", layout="wide")
initialize_state()

with st.sidebar:
    st.header("CDAS Control")
    st.session_state["ui_mode"] = st.radio("Mode", ["Learning", "Expert"], horizontal=True)
    if st.button("Reset App"):
        reset_app()
        st.success("State cleared.")
        st.stop()

st.title("📊 CDAS — 2) Data Diagnosis")
st.caption("Medical-style dataset health diagnosis and complexity scoring.")

if st.session_state.get("data") is None:
    st.warning("Upload data first.")
    st.stop()

df = st.session_state["data"]
diag = analyze_dataset(df)
issues = reason_from_diagnostics(diag)
st.session_state["diagnostics"] = diag
st.session_state["issues"] = issues

complexity, complexity_label = dataset_complexity_index(
    missing_ratio=diag["missing_ratio"],
    n_features=diag["n_cols"],
    correlation_max=diag["correlation_max"],
    imbalance=diag["target_balance"],
)
st.session_state["dci"] = (complexity, complexity_label)
st.info(f"Key Insight: Dataset Complexity Score = {complexity} / 10 ({complexity_label})")
st.divider()

completeness = 1 - diag["missing_ratio"]
consistency = 1 - diag["duplicate_ratio"]
redundancy = 1 - min(1.0, diag["correlation_max"])
balance = diag["target_balance"] if diag["target_balance"] is not None else 1.0

for label, value in [
    ("Completeness", completeness),
    ("Consistency", consistency),
    ("Redundancy", redundancy),
    ("Balance", balance),
]:
    if value >= 0.8:
        st.success(f"{label}: {value:.2f}")
    elif value >= 0.5:
        st.warning(f"{label}: {value:.2f}")
    else:
        st.error(f"{label}: {value:.2f}")

with st.expander("Structured Reasoning", expanded=st.session_state.get("ui_mode") == "Learning"):
    for issue in issues:
        st.markdown(
            f"**Issue:** {issue['issue']}\n\n"
            f"- Severity: {issue['severity']}\n"
            f"- Reasoning: {issue['reasoning']}\n"
            f"- Recommendation: {issue['recommendation']}\n"
            f"- Confidence: {issue['confidence']:.2f}"
        )
        st.divider()
