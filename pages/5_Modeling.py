"""Modeling page."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from core.intelligence_engine import suggest_model_type
from utils.helpers import infer_task_type
from utils.session import initialize_state, reset_app

st.set_page_config(page_title="Modeling", layout="wide")
initialize_state()

with st.sidebar:
    st.header("Control Panel")
    if st.button("Reset App"):
        reset_app()
        st.success("Application state cleared.")
        st.stop()
    st.progress(0.8)
    st.caption("Step 5/6")

st.header("5) Modeling")
st.subheader("Train baseline models with automatic task detection")
st.divider()

if st.session_state.get("engineered_data") is None:
    st.warning("Complete Feature Engineering first.")
    st.stop()

df = st.session_state["engineered_data"].dropna().copy()
if df.shape[1] < 2:
    st.warning("Need at least one feature and one target column.")
    st.stop()

target = st.selectbox("Select target column", df.columns)
st.session_state["target_column"] = target
st.info(suggest_model_type(df, target))

problem_type = infer_task_type(df[target])
st.write(f"Detected task type: **{problem_type}**")

X = pd.get_dummies(df.drop(columns=[target]), drop_first=False)
y = df[target]
stratify = y if problem_type == "classification" and y.nunique(dropna=True) > 1 else None

if st.button("Train Model"):
    with st.spinner("Training model..."):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=stratify,
            )

            if problem_type == "classification":
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = float(accuracy_score(y_test, y_pred))
                metric_name = "accuracy"
            else:
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = float(r2_score(y_test, y_pred))
                metric_name = "r2"

            st.session_state["model_payload"] = {
                "model": model,
                "problem_type": problem_type,
                "metric_name": metric_name,
                "score": score,
                "X_test": X_test,
                "y_test": y_test,
                "y_pred": y_pred,
            }
            st.success("Model training completed.")
        except Exception as exc:
            st.error(f"Model training failed: {exc}")

payload = st.session_state.get("model_payload")
if payload is not None:
    st.metric(payload["metric_name"].upper(), f"{payload['score']:.4f}")
    with st.expander("Prediction preview", expanded=False):
        preview = pd.DataFrame({"actual": payload["y_test"], "predicted": payload["y_pred"]})
        st.dataframe(preview.head(30), use_container_width=True)
