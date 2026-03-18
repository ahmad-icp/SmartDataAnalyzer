"""Streamlit entry point for SmartDataAnalyzer intelligent + user-controlled workflow."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ui_components import render_info_card, render_key_insights, render_page_intro  # noqa: E402
from core.cleaning import (  # noqa: E402
    correct_data_types,
    detect_outliers_iqr,
    handle_missing_values,
    remove_duplicates,
    suggest_cleaning_actions,
)
from core.correction import suggest_corrections  # noqa: E402
from core.data_quality import compute_data_quality_score  # noqa: E402
from core.eda import correlation_matrix, dataset_overview, missing_values_summary  # noqa: E402
from core.evaluation import evaluate_predictions  # noqa: E402
from core.explainability import compute_shap_summary, explain_single_prediction  # noqa: E402
from core.feature_engineering import (  # noqa: E402
    encode_features,
    run_feature_engineering,
    scale_numeric_features,
    suggest_highly_correlated_features,
)
from core.insights import generate_all_insights  # noqa: E402
from core.model_selection import ModelSelectionResult, run_model_selection  # noqa: E402
from core.uncertainty import bayesian_prediction_interval  # noqa: E402
from utils.helpers import infer_task_type  # noqa: E402
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


@st.cache_resource(show_spinner=False)
def cached_run_automl(df: pd.DataFrame, target_column: str) -> ModelSelectionResult:
    """Cache expensive AutoML training."""
    return run_model_selection(df, target_column)


@st.cache_resource(show_spinner=False)
def cached_train_manual(df: pd.DataFrame, target_column: str, model_choice: str) -> dict[str, Any]:
    """Cache manual model training runs."""
    X = pd.get_dummies(df.drop(columns=[target_column]), drop_first=False)
    y = df[target_column]
    task_type = infer_task_type(y)
    stratify = y if task_type == "classification" and y.nunique(dropna=True) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    if task_type == "classification":
        model_map = {
            "Random Forest": RandomForestClassifier(random_state=42),
            "Linear": LogisticRegression(max_iter=1000),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        }
        scorer = accuracy_score
        metric_name = "accuracy"
    else:
        model_map = {
            "Random Forest": RandomForestRegressor(random_state=42),
            "Linear": LinearRegression(),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        }
        scorer = lambda yt, yp: float(mean_squared_error(yt, yp) ** 0.5)
        metric_name = "rmse"

    model = model_map[model_choice]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "task_type": task_type,
        "metric_name": metric_name,
        "model_name": model_choice,
        "model": model,
        "score": float(scorer(y_test, y_pred)),
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def apply_column_level_cleaning(df: pd.DataFrame, actions: dict[str, str], drop_duplicates: bool) -> pd.DataFrame:
    """Apply column-wise cleaning actions selected by the user."""
    out = df.copy()
    for col, action in actions.items():
        if action == "None":
            continue
        if action == "Drop":
            out = out.drop(columns=[col])
            continue
        if col not in out.columns or out[col].isna().sum() == 0:
            continue
        if action == "Mean" and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(out[col].mean())
        elif action == "Median" and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(out[col].median())
        else:
            mode = out[col].mode(dropna=True)
            if not mode.empty:
                out[col] = out[col].fillna(mode.iloc[0])

    out = correct_data_types(out)
    if drop_duplicates:
        out = remove_duplicates(out)
    return out


def drop_correlated_features(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Drop one side of each highly correlated pair."""
    pairs = suggest_highly_correlated_features(df, threshold=threshold)
    to_drop = {pair[1] for pair in pairs}
    if not to_drop:
        return df
    return df.drop(columns=[c for c in to_drop if c in df.columns])


def initialize_state(df_raw: pd.DataFrame) -> None:
    """Initialize default session state objects."""
    st.session_state.setdefault("cleaning_actions", {col: "None" for col in df_raw.columns})
    st.session_state.setdefault("cleaned_df", df_raw.copy())
    st.session_state.setdefault("corrected_df", st.session_state["cleaned_df"].copy())
    st.session_state.setdefault("engineered_df", st.session_state["corrected_df"].copy())
    st.session_state.setdefault("model_result", None)
    st.session_state.setdefault("manual_model_result", None)
    st.session_state.setdefault("normalized_columns", [])


def reset_state_for_new_upload(upload_signature: str, df_raw: pd.DataFrame) -> None:
    """Reset dependent state only when upload changes."""
    if st.session_state.get("upload_signature") == upload_signature:
        return

    st.session_state["upload_signature"] = upload_signature
    st.session_state["cleaning_actions"] = {col: "None" for col in df_raw.columns}
    st.session_state["cleaned_df"] = df_raw.copy()
    st.session_state["corrected_df"] = df_raw.copy()
    st.session_state["engineered_df"] = df_raw.copy()
    st.session_state["model_result"] = None
    st.session_state["manual_model_result"] = None
    st.session_state["normalized_columns"] = []


upload = st.file_uploader("Upload dataset (CSV only)", type=["csv"])
if upload is None:
    st.info("Start by uploading a CSV dataset to begin the intelligent analysis flow.")
    st.stop()

try:
    file_bytes = upload.getvalue()
    df_raw = parse_upload(upload.name, file_bytes)
except Exception as exc:
    st.error(f"Unable to load file: {exc}")
    st.stop()

upload_signature = f"{upload.name}:{len(file_bytes)}"
reset_state_for_new_upload(upload_signature, df_raw)
initialize_state(df_raw)

mode = st.radio("Select Mode", ["Auto", "Manual"], horizontal=True)
st.info(
    "Auto mode applies recommended processing quickly. "
    "Manual mode enables fine-grained control over cleaning, correction, feature engineering, and model training."
)

# 1. Upload Data
st.header("1) Upload Data")
overview, missing_summary, corr = cached_eda(df_raw)
col_a, col_b, col_c = st.columns(3)
with col_a:
    render_info_card("Rows", str(overview["shape"][0]))
with col_b:
    render_info_card("Columns", str(overview["shape"][1]))
with col_c:
    render_info_card("Missing Cells", str(int(df_raw.isna().sum().sum())))
st.divider()

# 2. EDA
st.header("2) EDA (Read-only)")
eda_left, eda_right = st.columns([1, 2])
with eda_left:
    st.markdown("#### Column Types")
    st.json(overview["dtypes"])
    st.markdown("#### Missing Summary")
    st.dataframe(missing_summary, use_container_width=True)
with eda_right:
    if not corr.empty:
        st.plotly_chart(correlation_heatmap(corr), use_container_width=True)
    numeric_cols = df_raw.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        selected_num_col = st.selectbox("Distribution column", numeric_cols, key="dist_col")
        st.plotly_chart(distribution_plot(df_raw, selected_num_col), use_container_width=True)
st.divider()

# 3. Cleaning
st.header("3) Data Cleaning (User-controlled)")
for suggestion in suggest_cleaning_actions(df_raw):
    st.write(f"- {suggestion}")

missing_pct = (df_raw.isna().sum() / max(len(df_raw), 1) * 100).round(2)
for col in df_raw.columns:
    default_choice = st.session_state["cleaning_actions"].get(col, "None")
    st.write(f"**{col}** - Missing: {missing_pct[col]}%")
    st.session_state["cleaning_actions"][col] = st.selectbox(
        f"Cleaning method for {col}",
        ["None", "Mean", "Median", "Mode", "Drop"],
        index=["None", "Mean", "Median", "Mode", "Drop"].index(default_choice)
        if default_choice in ["None", "Mean", "Median", "Mode", "Drop"]
        else 0,
        key=f"cleaning_action_{col}",
    )

remove_dupes = st.checkbox("Remove duplicate rows", value=True)
if st.button("Apply Cleaning"):
    st.session_state["cleaned_df"] = apply_column_level_cleaning(
        df_raw,
        st.session_state["cleaning_actions"],
        drop_duplicates=remove_dupes,
    )
    st.session_state["corrected_df"] = st.session_state["cleaned_df"].copy()
    st.session_state["engineered_df"] = st.session_state["corrected_df"].copy()
    st.success("Cleaning applied successfully.")

st.dataframe(st.session_state["cleaned_df"].head(20), use_container_width=True)
st.write("Outlier counts (IQR):", detect_outliers_iqr(st.session_state["cleaned_df"]))
st.divider()

# 4. Data Correction
st.header("4) Data Correction (Editable fuzzy matching)")
working_df = st.session_state["corrected_df"].copy()
cat_cols = working_df.select_dtypes(include="object").columns.tolist()
if not cat_cols:
    st.info("No categorical columns found for correction.")
else:
    selected_cat_col = st.selectbox("Column for fuzzy correction", cat_cols)
    groups = suggest_corrections(working_df[selected_cat_col])
    if not groups:
        st.info("No strong correction groups detected.")
    else:
        st.warning("Review each detected group and edit canonical values before applying.")
        edited_groups: list[dict[str, object]] = []
        for idx, group in enumerate(groups):
            st.write(f"Detected values: {group['original']}")
            canonical = st.text_input(
                f"Canonical value for group {idx + 1}",
                value=str(group["suggested"]),
                key=f"canonical_{selected_cat_col}_{idx}",
            )
            edited_groups.append({"original": group["original"], "suggested": canonical})

        if st.button("Apply Corrections"):
            corrected = working_df.copy()
            for item in edited_groups:
                corrected[selected_cat_col] = corrected[selected_cat_col].replace(item["original"], item["suggested"])
            st.session_state["corrected_df"] = corrected
            if selected_cat_col not in st.session_state["normalized_columns"]:
                st.session_state["normalized_columns"].append(selected_cat_col)
            st.session_state["engineered_df"] = corrected.copy()
            st.success(f"Corrections applied for '{selected_cat_col}'.")

st.dataframe(st.session_state["corrected_df"].head(20), use_container_width=True)
st.divider()

# 5. Feature Engineering
st.header("5) Feature Engineering (User + Auto)")
fe_input_df = st.session_state["corrected_df"].copy()
fe_diag = run_feature_engineering(fe_input_df)
if fe_diag.recommendations:
    st.info("System recommendations")
    for rec in fe_diag.recommendations:
        st.write(f"- {rec}")

fe_col1, fe_col2, fe_col3 = st.columns(3)
with fe_col1:
    apply_encoding = st.checkbox("Apply encoding", value=True)
with fe_col2:
    apply_scaling = st.checkbox("Apply scaling", value=True)
with fe_col3:
    drop_corr = st.checkbox("Drop highly correlated features", value=False)

if st.button("Apply Feature Engineering"):
    engineered = fe_input_df.copy()
    if apply_encoding:
        engineered = encode_features(engineered)
    if drop_corr:
        engineered = drop_correlated_features(engineered)
    if apply_scaling:
        engineered = scale_numeric_features(engineered)

    st.session_state["engineered_df"] = engineered
    st.success("Feature engineering applied.")

st.write("Highly correlated pairs:", fe_diag.high_correlation_pairs)
st.dataframe(st.session_state["engineered_df"].head(20), use_container_width=True)
st.divider()

# 6. Model Training
st.header("6) Model Training (Manual + Auto)")
training_df = st.session_state["engineered_df"].dropna().copy()
if training_df.empty or training_df.shape[1] < 2:
    st.warning("Need at least one feature and one target column after preprocessing.")
else:
    target_column = st.selectbox("Target column", training_df.columns)
    model_choice = st.selectbox(
        "Select model",
        ["Auto", "Random Forest", "Linear", "Gradient Boosting"],
    )

    if mode == "Auto":
        st.info("Auto mode will prioritize the full intelligent pipeline.")

    if st.button("Train Model"):
        try:
            if model_choice == "Auto":
                model_result = cached_run_automl(training_df, target_column)
                st.session_state["model_result"] = model_result
                st.session_state["manual_model_result"] = None
                st.success(f"AutoML selected: {model_result.best_model_name}")
            else:
                manual_result = cached_train_manual(training_df, target_column, model_choice)
                st.session_state["manual_model_result"] = manual_result
                st.session_state["model_result"] = None
                st.success(f"Manual model trained: {manual_result['model_name']}")
        except Exception as exc:
            st.error(f"Model training failed: {exc}")
st.divider()

# 7. Results & Insights
st.header("7) Results & Insights")
quality_report = compute_data_quality_score(st.session_state["corrected_df"])
res_left, res_mid, res_right = st.columns(3)
with res_left:
    render_info_card("Data Quality", f"{quality_report['score']}/100")
with res_mid:
    st.caption(quality_report["label"])
with res_right:
    st.json(quality_report["penalties"])

feature_importance_df: pd.DataFrame | None = None
active_result = st.session_state.get("model_result")
manual_result = st.session_state.get("manual_model_result")

if active_result is not None:
    render_info_card("Best Model", active_result.best_model_name)
    render_info_card(active_result.metric_name.upper(), f"{active_result.best_test_score:.4f}")
    st.dataframe(active_result.performance_table, use_container_width=True)
    st.plotly_chart(
        model_comparison_chart(active_result.performance_table, active_result.metric_name),
        use_container_width=True,
    )
    metrics = evaluate_predictions(active_result.task_type, active_result.y_test, active_result.y_pred)
    st.write("Evaluation", metrics)

    if st.checkbox("Run SHAP Explainability", value=False):
        try:
            shap_summary = compute_shap_summary(active_result.best_model, active_result.X_test.head(50))
            feature_importance_df = shap_summary["feature_importance"]
            st.plotly_chart(feature_importance_chart(feature_importance_df), use_container_width=True)
            st.write("Local Explanation", explain_single_prediction(active_result.best_model, active_result.X_test.head(1)))
        except Exception as exc:
            st.warning(f"SHAP unavailable in this runtime: {exc}")

    if active_result.task_type == "regression":
        try:
            uncertainty_df = bayesian_prediction_interval(
                active_result.X_train,
                active_result.y_train,
                active_result.X_test.head(10),
            )
            st.dataframe(uncertainty_df, use_container_width=True)
            st.info("Prediction format: value ± uncertainty")
        except Exception as exc:
            st.warning(f"Uncertainty estimation unavailable: {exc}")
elif manual_result is not None:
    render_info_card("Selected Model", manual_result["model_name"])
    render_info_card(manual_result["metric_name"].upper(), f"{manual_result['score']:.4f}")
    metrics = evaluate_predictions(manual_result["task_type"], manual_result["y_test"], manual_result["y_pred"])
    st.write("Evaluation", metrics)
else:
    st.info("Train a model to view performance and explainability outputs.")

insights = generate_all_insights(
    df=st.session_state["corrected_df"],
    model_result=active_result,
    feature_importance=feature_importance_df,
    normalized_columns=st.session_state.get("normalized_columns", []),
)
render_key_insights(insights)
st.divider()

# 8. Download
st.header("8) Download")
download_df = st.session_state["engineered_df"].copy()
csv_bytes = download_df.to_csv(index=False).encode("utf-8")
st.download_button("Download cleaned CSV", data=csv_bytes, file_name="smart_data_analyzer_output.csv", mime="text/csv")

excel_buffer = io.BytesIO()
try:
    download_df.to_excel(excel_buffer, index=False, engine="openpyxl")
    st.download_button(
        "Download Excel",
        data=excel_buffer.getvalue(),
        file_name="smart_data_analyzer_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
except Exception:
    st.warning("Excel export requires openpyxl. Install it to enable .xlsx downloads.")
