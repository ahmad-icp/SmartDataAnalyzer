"""Session-state helpers for multi-page Streamlit app."""

from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st


STATE_DEFAULTS: dict[str, object] = {
    "data": None,
    "cleaned_data": None,
    "engineered_data": None,
    "target_column": None,
    "model_payload": None,
    "insights": [],
    "upload_hash": None,
}


def initialize_state() -> None:
    """Ensure expected session keys exist."""
    for key, default in STATE_DEFAULTS.items():
        st.session_state.setdefault(key, default)


def compute_file_hash(file_bytes: bytes) -> str:
    """Compute deterministic hash for upload reset checks."""
    return hashlib.sha256(file_bytes).hexdigest()


def set_dataset(df: pd.DataFrame, upload_hash: str) -> None:
    """Set dataset and reset downstream artifacts on new file."""
    if st.session_state.get("upload_hash") == upload_hash:
        return

    st.session_state["upload_hash"] = upload_hash
    st.session_state["data"] = df.copy()
    st.session_state["cleaned_data"] = df.copy()
    st.session_state["engineered_data"] = df.copy()
    st.session_state["target_column"] = None
    st.session_state["model_payload"] = None
    st.session_state["insights"] = []


def reset_app() -> None:
    """Reset all Streamlit session state."""
    st.session_state.clear()
