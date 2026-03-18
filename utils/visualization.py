"""Visualization helpers with interactive Plotly charts."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    """Build an interactive styled correlation heatmap."""
    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        text_auto=".2f",
        title="Feature Correlation Matrix",
    )
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def distribution_plot(df: pd.DataFrame, column: str) -> go.Figure:
    """Build an interactive univariate distribution chart."""
    fig = px.histogram(
        df,
        x=column,
        marginal="box",
        nbins=30,
        title=f"Distribution of {column}",
        template="plotly_white",
    )
    return fig


def model_comparison_chart(performance_table: pd.DataFrame, metric_name: str) -> go.Figure:
    """Build model comparison bar chart."""
    fig = px.bar(
        performance_table,
        x="model",
        y=metric_name,
        color="model",
        title=f"Model Comparison by {metric_name.upper()}",
        template="plotly_white",
    )
    fig.update_layout(showlegend=False)
    return fig


def feature_importance_chart(importance_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Build a horizontal bar chart for feature importances."""
    data = importance_df.head(top_n).sort_values("importance", ascending=True)
    fig = px.bar(
        data,
        x="importance",
        y="feature",
        orientation="h",
        title=f"Top {min(top_n, len(data))} Feature Importances",
        template="plotly_white",
    )
    return fig
