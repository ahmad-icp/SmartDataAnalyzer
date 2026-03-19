"""Reusable UI components for premium dashboard-style Streamlit pages."""

from __future__ import annotations

import streamlit as st

PAGE_FLOW: list[tuple[str, str, str]] = [
    ("upload", "Data Upload", "📂"),
    ("eda", "EDA", "📊"),
    ("cleaning", "Data Cleaning", "🧹"),
    ("feature", "Feature Engineering", "⚙️"),
    ("model", "Modeling", "🤖"),
    ("results", "Results", "📈"),
]


def render_sidebar(active_page: str) -> None:
    """Render fixed sidebar style navigation and reset region."""
    st.header("Smart Data Analyzer")
    st.caption("Premium analytics workflow")
    st.divider()

    for key, label, icon in PAGE_FLOW:
        prefix = "✅" if key == active_page else "▫️"
        st.markdown(f"{prefix} {icon} {label}")

    completed = PAGE_FLOW.index(next(p for p in PAGE_FLOW if p[0] == active_page)) + 1
    st.divider()
    st.caption("Progress")
    st.progress(completed / len(PAGE_FLOW))


def render_page_header(icon: str, title: str, description: str) -> None:
    """Render standardized page heading."""
    st.header(f"{icon} {title}")
    st.caption(description)


def render_insight_banners(insights: list[str]) -> None:
    """Render one or two key insights prominently."""
    for item in insights[:2]:
        if "warning" in item.lower() or "missing" in item.lower() or "strong" in item.lower():
            st.warning(item)
        else:
            st.info(item)


def render_stat_cards(stats: list[tuple[str, str, str]]) -> None:
    """Render icon + label + value cards in columns."""
    cols = st.columns(len(stats))
    for col, (icon, label, value) in zip(cols, stats):
        with col:
            st.metric(f"{icon} {label}", value=value)
