import plotly.express as px
import pandas as pd


def make_chart(df: pd.DataFrame, x: str | None, y: str | None, chart_type: str):
    if chart_type == "scatter":
        return px.scatter(df, x=x, y=y)
    if chart_type == "line":
        return px.line(df, x=x, y=y)
    if chart_type == "bar":
        return px.bar(df, x=x, y=y)
    if chart_type == "histogram":
        return px.histogram(df, x=x)
    if chart_type == "box":
        return px.box(df, x=x, y=y)
    if chart_type == "heatmap":
        num = df.select_dtypes(include=["number"])
        corr = num.corr()
        return px.imshow(corr, text_auto=True)
    return px.scatter(df, x=x, y=y)


def export_plotly_png(fig):
    # requires kaleido
    return fig.to_image(format="png")
"""Plotting helpers for Smart Data Analyzer."""

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff


def plot_chart(df: pd.DataFrame, chart_type: str, x: str = None, y: str = None):
    """Create a Plotly chart based on type and selected columns."""

    chart_type = chart_type.lower()

    if chart_type == "histogram":
        return px.histogram(df, x=x, title=f"Histogram of {x}")

    if chart_type == "bar chart":
        return px.bar(df, x=x, y=y, title=f"Bar chart: {y} vs {x}")

    if chart_type == "line chart":
        return px.line(df, x=x, y=y, title=f"Line chart: {y} vs {x}")

    if chart_type == "scatter plot":
        return px.scatter(df, x=x, y=y, title=f"Scatter plot: {y} vs {x}")

    if chart_type == "box plot":
        return px.box(df, x=x, y=y, points="all", title=f"Box plot: {y} by {x}")

    raise ValueError(f"Unsupported chart type: {chart_type}")


def plot_correlation_heatmap(df: pd.DataFrame):
    """Create a heatmap of correlation coefficients."""

    numeric = df.select_dtypes(include=["number"])
    corr = numeric.corr()
    if corr.empty:
        raise ValueError("No numeric columns available for correlation heatmap.")

    z = corr.values
    x = list(corr.columns)
    y = list(corr.index)

    fig = ff.create_annotated_heatmap(
        z,
        x=x,
        y=y,
        annotation_text=corr.round(2).astype(str).values,
        colorscale="Viridis",
        hoverinfo="z",
    )
    fig.update_layout(title_text="Correlation Heatmap", width=800, height=700)
    return fig


def plot_pairplot(df: pd.DataFrame, numeric_cols: list):
    """Create a scatter matrix (pair plot) for numeric columns."""

    if not numeric_cols:
        raise ValueError("No numeric columns available for pair plot.")

    return px.scatter_matrix(
        df[numeric_cols],
        dimensions=numeric_cols,
        title="Pair Plot (Scatter Matrix)",
        height=800,
        width=800,
    )
