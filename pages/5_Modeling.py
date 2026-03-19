from __future__ import annotations

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from utils.helpers import infer_task_type
from utils.session import initialize_state, reset_app

st.set_page_config(page_title="Modeling", layout="wide")
initialize_state()

with st.sidebar:
    st.header("📂 Control Panel")
    if st.button("Reset App"):
        reset_app()
        st.success("Application state cleared.")
        st.stop()

st.header("🤖 Modeling")
st.caption("Automatic problem detection with model recommendation and training.")

if st.session_state.get("engineered_data") is None:
    st.warning("No engineered data found. Complete Feature Engineering first.")
    st.stop()

df = st.session_state["engineered_data"].dropna().copy()
if df.shape[1] < 2:
    st.error("Need at least one feature and one target column.")
    st.stop()

target = st.selectbox("Select target column", df.columns)
X = pd.get_dummies(df.drop(columns=[target]), drop_first=False)
y = df[target]
problem = infer_task_type(y)

st.info(f"Key Insight: Detected Problem Type: {problem.capitalize()}")
if problem == "classification":
    st.success("Recommended Model: Logistic Regression")
else:
    st.success("Recommended Model: Linear Regression")

if st.button("Train Model"):
    stratify = y if problem == "classification" and y.nunique(dropna=True) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)

    if problem == "classification":
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = float(accuracy_score(y_test, pred))
        metric = "accuracy"
    else:
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = float(r2_score(y_test, pred))
        metric = "r2"

    st.session_state["model_payload"] = {
        "problem": problem,
        "metric": metric,
        "score": score,
        "target": target,
        "rows_used": len(df),
    }
    st.success("Model training completed.")

with st.expander("Model Details", expanded=False):
    payload = st.session_state.get("model_payload")
    if payload:
        st.json(payload)
