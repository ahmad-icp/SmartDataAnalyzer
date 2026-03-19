from __future__ import annotations

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from app.ui_components import render_insight_banners, render_page_header, render_sidebar
from utils.helpers import infer_task_type
from utils.session import initialize_state, reset_app

st.set_page_config(page_title="Modeling", layout="wide")
initialize_state()

with st.sidebar:
    render_sidebar("model")
    if st.button("Reset App", use_container_width=True):
        reset_app()
        st.success("Session reset complete.")
        st.stop()

render_page_header("🤖", "Modeling", "Detect problem type automatically and train recommended baseline models.")

if st.session_state.get("engineered_data") is None:
    st.warning("No engineered dataset found. Complete Feature Engineering first.")
    st.stop()

df = st.session_state["engineered_data"].dropna().copy()
if df.shape[1] < 2:
    st.error("Need at least one feature and one target column.")
    st.stop()

target = st.selectbox("Select target column", df.columns)
X = pd.get_dummies(df.drop(columns=[target]), drop_first=False)
y = df[target]
problem = infer_task_type(y)

if problem == "classification":
    render_insight_banners([
        "🧠 Detected Problem Type: Classification",
        "Logistic Regression is suitable because target is categorical/low-cardinality.",
    ])
else:
    render_insight_banners([
        "🧠 Detected Problem Type: Regression",
        "Linear Regression is suitable because target appears continuous.",
    ])

if st.button("Train model"):
    stratify = y if problem == "classification" and y.nunique(dropna=True) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)

    if problem == "classification":
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metric = "accuracy"
        score = float(accuracy_score(y_test, pred))
    else:
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metric = "r2"
        score = float(r2_score(y_test, pred))

    st.session_state["model_payload"] = {
        "problem": problem,
        "metric": metric,
        "score": score,
        "target": target,
        "rows_used": len(df),
    }
    st.success("Training complete.")

with st.expander("Model details", expanded=False):
    if st.session_state.get("model_payload"):
        st.json(st.session_state["model_payload"])
