from __future__ import annotations

import streamlit as st

from utils.session import initialize_state, reset_app

st.set_page_config(page_title="CDAS", layout="wide")
initialize_state()

with st.sidebar:
    st.header("CDAS Control Panel")
    mode = st.radio("Mode", ["Learning", "Expert"], horizontal=True)
    st.session_state["ui_mode"] = mode
    if st.button("Reset App"):
        reset_app()
        st.success("Session reset completed.")
        st.stop()
    st.info("Navigate through pages 1→7 for end-to-end cognitive analysis.")

st.title("🧠 Cognitive Data Analysis System (CDAS)")
st.caption("Research-grade, API-free intelligent data analysis platform")
st.info("Key Insight: CDAS simulates AI reasoning using local statistical logic and machine learning heuristics.")
st.divider()

st.markdown(
    """
### Navigation
1. **Upload**
2. **Data Diagnosis**
3. **Smart Cleaning**
4. **Feature Intelligence**
5. **Model Advisor**
6. **Results & Insights**
7. **Report Generator**
"""
)

with st.expander("System capabilities", expanded=mode == "Learning"):
    st.markdown(
        """
- Multi-step issue reasoning (severity, rationale, recommendation, confidence)
- Dataset Complexity Index (DCI)
- Data Health diagnosis (completeness, consistency, redundancy, balance)
- Adaptive preprocessing pipeline generator
- Explainable model advisor + error simulation
- Downloadable research-style reports
"""
    )
