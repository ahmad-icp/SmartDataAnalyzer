from __future__ import annotations

import streamlit as st

from core.feature_engineering import encode_features, scale_numeric_features, suggest_highly_correlated_features
from utils.session import initialize_state, reset_app

st.set_page_config(page_title="CDAS • Feature Intelligence", layout="wide")
initialize_state()

with st.sidebar:
    st.header("CDAS Control")
    st.session_state["ui_mode"] = st.radio("Mode", ["Learning", "Expert"], horizontal=True)
    if st.button("Reset App"):
        reset_app()
        st.success("State cleared.")
        st.stop()

st.title("🧬 CDAS — 4) Feature Intelligence")
st.caption("Feature diagnostics and guided transformation controls.")

if st.session_state.get("cleaned_data") is None:
    st.warning("Complete Smart Cleaning first.")
    st.stop()

df = st.session_state["cleaned_data"].copy()
correlated = suggest_highly_correlated_features(df)
st.info("Key Insight: " + (f"{len(correlated)} high-correlation pairs detected." if correlated else "No high-correlation pairs detected."))
st.divider()

apply_encoding = st.checkbox("Apply one-hot encoding", value=True)
apply_scaling = st.checkbox("Apply standard scaling", value=True)
drop_corr = st.checkbox("Drop highly correlated numeric features", value=False)

if st.button("Run Feature Intelligence"):
    out = df.copy()
    if apply_encoding:
        out = encode_features(out)
    if drop_corr and correlated:
        drop_cols = sorted({pair[1] for pair in correlated if pair[1] in out.columns})
        out = out.drop(columns=drop_cols)
    if apply_scaling:
        out = scale_numeric_features(out)

    st.session_state["engineered_data"] = out
    st.success("Feature intelligence pipeline executed.")

with st.expander("Detailed feature diagnostics", expanded=st.session_state.get("ui_mode") == "Learning"):
    st.write("Highly correlated pairs:", correlated)
    st.dataframe(st.session_state.get("engineered_data", df).head(50), use_container_width=True)
