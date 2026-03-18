"""Streamlit entry point for SmartDataAnalyzer intelligent workflow."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ui_components import (  # noqa: E402
    render_info_card,
    render_key_insights,
    render_page_intro,
    render_step_header,
)
from core.cleaning import (  # noqa: E402
    correct_data_types,
    detect_outliers_iqr,
    handle_missing_values,
    remove_duplicates,
    suggest_cleaning_actions,
)
from core.correction import apply_corrections, suggest_corrections  # noqa: E402
from core.data_quality import compute_data_quality_score  # noqa: E402
from core.eda import correlation_matrix, dataset_overview, missing_values_summary  # noqa: E402
from core.evaluation import evaluate_predictions  # noqa: E402
from core.explainability import compute_shap_summary, explain_single_prediction  # noqa: E402
from core.feature_engineering import run_feature_engineering  # noqa: E402
from core.insights import generate_all_insights  # noqa: E402
from core.model_selection import run_model_selection  # noqa: E402
from core.uncertainty import bayesian_prediction_interval  # noqa: E402
from utils.validators import load_csv_bytes, validate_uploaded_file  # noqa: E402
from utils.visualization import (  # noqa: E402
    correlation_heatmap,
    distribution_plot,
    feature_importance_chart,
    model_comparison_chart,
)

st.set_page_config(page_title="SmartDataAnalyzer", layout="wide")
render_page_intro()


@st.cache_data(show_spinner=False)
def parse_upload(name: str, data: bytes) -> pd.DataFrame:
    """Cached loader for uploaded CSV bytes."""
    validation = validate_uploaded_file(name, data)
    if not validation.is_valid:
        raise ValueError(validation.message)
    return load_csv_bytes(data)


@st.cache_data(show_spinner=False)
def cached_eda(df: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Cache frequently re-used EDA outputs."""
    return dataset_overview(df), missing_values_summary(df), correlation_matrix(df)


upload = st.file_uploader("Upload dataset (CSV only)", type=["csv"])
if upload is None:
    st.info("Start by uploading a CSV dataset to begin the intelligent analysis flow.")
    st.stop()

try:
    df_raw = parse_upload(upload.name, upload.getvalue())
except Exception as exc:
    st.error(f"Unable to load file: {exc}")
    st.stop()

upload_signature = f"{upload.name}:{len(upload.getvalue())}"
if st.session_state.get("upload_signature") != upload_signature:
    st.session_state["upload_signature"] = upload_signature
    st.session_state.pop("model_result", None)
    st.session_state["normalized_columns"] = []

render_step_header(1, "Upload", "Validate upload and capture high-level data profile.")
overview, missing, corr = cached_eda(df_raw)
up_cols = st.columns(3)
with up_cols[0]:
    render_info_card("Rows", str(overview["shape"][0]))
with up_cols[1]:
    render_info_card("Columns", str(overview["shape"][1]))
with up_cols[2]:
    render_info_card("Missing Cells", str(int(df_raw.isna().sum().sum())))

render_step_header(2, "EDA", "Explore schema, missingness, distribution, and correlations.")
eda_col1, eda_col2 = st.columns([1, 2])
with eda_col1:
    st.markdown("#### Column Types")
    st.json(overview["dtypes"])
    st.markdown("#### Missing Summary")
    st.dataframe(missing)
with eda_col2:
    if not corr.empty:
        st.plotly_chart(correlation_heatmap(corr), use_container_width=True)
    num_cols = df_raw.select_dtypes(include="number").columns.tolist()
    if num_cols:
        st.plotly_chart(distribution_plot(df_raw, num_cols[0]), use_container_width=True)

render_step_header(3, "Cleaning", "Apply quality improvements with smart recommendations.")
cleaning_recommendations = suggest_cleaning_actions(df_raw)
for rec in cleaning_recommendations:
    st.markdown(f"- {rec}")
clean_strategy = st.selectbox("Missing value strategy", ["mean", "median", "mode", "drop"])
df_clean = correct_data_types(remove_duplicates(handle_missing_values(df_raw, strategy=clean_strategy)))
outlier_counts = detect_outliers_iqr(df_clean)
st.write("Outlier counts (IQR):", outlier_counts)

render_step_header(4, "Correction", "Normalize inconsistent categories via fuzzy grouping.")
normalized_columns = st.session_state.get("normalized_columns", [])
cat_cols = df_clean.select_dtypes(include="object").columns.tolist()
if cat_cols:
    corr_col = st.selectbox("Column for correction", cat_cols)
    correction_suggestions = suggest_corrections(df_clean[corr_col])
    if correction_suggestions:
        st.json(correction_suggestions)
        if st.button("Apply normalization"):
            df_clean = apply_corrections(df_clean, corr_col, correction_suggestions)
            if corr_col not in normalized_columns:
                normalized_columns.append(corr_col)
            st.session_state["normalized_columns"] = normalized_columns
            st.success(f"Normalized values in '{corr_col}'.")
    else:
        st.info("No strong correction candidates found.")
else:
    st.info("No categorical columns available for correction.")

render_step_header(5, "Feature Engineering", "Encode, scale, and flag weak/redundant features.")
fe_result = run_feature_engineering(df_clean)
fe_col1, fe_col2 = st.columns(2)
with fe_col1:
    st.write("Dropped low-variance features:", fe_result.dropped_low_variance)
with fe_col2:
    st.write("Highly correlated pairs:", fe_result.high_correlation_pairs)
if fe_result.recommendations:
    st.warning("\n".join(fe_result.recommendations))

render_step_header(6, "Model Training", "Train candidates and select best model automatically.")
target_column = st.selectbox("Target column", df_clean.columns)
if st.button("Train models"):
    try:
        model_result = run_model_selection(df_clean.dropna(), target_column)
        st.session_state["model_result"] = model_result
        st.success(f"Best model selected: {model_result.best_model_name}")
    except Exception as exc:
        st.error(f"Model training failed: {exc}")

render_step_header(7, "Insights & Results", "Summarize quality, performance, explainability, and recommendations.")
result = st.session_state.get("model_result")
quality_report = compute_data_quality_score(df_clean)
res_cols = st.columns(3)
with res_cols[0]:
    render_info_card("Data Quality", f"{quality_report['score']}/100")
with res_cols[1]:
    if result:
        render_info_card("Best Model", result.best_model_name)
with res_cols[2]:
    if result:
        render_info_card(result.metric_name.upper(), f"{result.best_test_score:.4f}")

st.caption(quality_report["label"])
st.json(quality_report["penalties"])

feature_importance_df: pd.DataFrame | None = None
if result:
    st.dataframe(result.performance_table)
    st.plotly_chart(model_comparison_chart(result.performance_table, result.metric_name), use_container_width=True)
    metrics = evaluate_predictions(result.task_type, result.y_test, result.y_pred)
    st.write("Evaluation", metrics)

    if st.checkbox("Run SHAP Explainability", value=False):
        try:
            shap_summary = compute_shap_summary(result.best_model, result.X_test.head(50))
            feature_importance_df = shap_summary["feature_importance"]
            st.plotly_chart(feature_importance_chart(feature_importance_df), use_container_width=True)
            st.write("Local Explanation", explain_single_prediction(result.best_model, result.X_test.head(1)))
        except Exception as exc:
            st.warning(f"SHAP unavailable in this runtime: {exc}")

    if result.task_type == "regression":
        try:
            uncertainty_df = bayesian_prediction_interval(
                result.X_train,
                result.y_train,
                result.X_test.head(10),
            )
            st.dataframe(uncertainty_df)
            st.info("Prediction format: value ± uncertainty")
        except Exception as exc:
            st.warning(f"Uncertainty estimation unavailable: {exc}")

insights = generate_all_insights(
    df=df_clean,
    model_result=result,
    feature_importance=feature_importance_df,
    normalized_columns=normalized_columns,
)
render_key_insights(insights)
