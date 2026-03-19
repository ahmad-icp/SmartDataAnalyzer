"""Reusable Streamlit UI components."""

from __future__ import annotations

import streamlit as st

TOTAL_STEPS = 7


def render_page_intro() -> None:
    """Render product-style header."""
    st.title("🧠 SmartDataAnalyzer Pro")
    st.caption(
        "An intelligent platform for data quality diagnostics, automated modeling, explainability, and insights."
    )


def render_step_header(step: int, title: str, description: str) -> None:
    """Render a styled step header with progress bar."""
    st.progress(step / TOTAL_STEPS)
    st.markdown(f"### Step {step}/{TOTAL_STEPS}: {title}")
    st.info(description)


def render_info_card(title: str, value: str, delta: str | None = None) -> None:
    """Render compact metric card."""
    st.metric(title, value=value, delta=delta)


def render_key_insights(insights: list[str]) -> None:
    """Render key insights block."""
    st.markdown("#### 🔍 Automated Insights")
    for item in insights:
        st.markdown(f"- {item}")
