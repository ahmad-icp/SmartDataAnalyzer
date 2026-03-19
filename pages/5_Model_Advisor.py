from __future__ import annotations

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from core.cognitive.recommendations import model_advisor
from utils.helpers import infer_task_type
from utils.session import initialize_state, reset_app

st.set_page_config(page_title="CDAS • Model Advisor", layout="wide")
initialize_state()

with st.sidebar:
    st.header("CDAS Control")
    st.session_state["ui_mode"] = st.radio("Mode", ["Learning", "Expert"], horizontal=True)
    if st.button("Reset App"):
        reset_app()
        st.success("State cleared.")
        st.stop()

st.title("🤖 CDAS — 5) Model Advisor")
st.caption("Explainable model recommendation and training workspace.")

if st.session_state.get("engineered_data") is None:
    st.warning("Complete Feature Intelligence first.")
    st.stop()

df = st.session_state["engineered_data"].dropna().copy()
if df.shape[1] < 2:
    st.warning("Need at least one feature and one target column.")
    st.stop()

target = st.selectbox("Target column", df.columns)
st.session_state["target_column"] = target
problem = infer_task_type(df[target])
classes = int(df[target].nunique()) if problem == "classification" else None

recs = model_advisor(problem, classes)
st.info("Key Insight: " + recs[0]["why"])
st.divider()

for rec in recs:
    st.write(f"- **{rec['model']}** → {rec['why']}")

model_name = st.selectbox("Model", [item["model"] for item in recs])
if st.button("Train Recommended Model"):
    X = pd.get_dummies(df.drop(columns=[target]), drop_first=False)
    y = df[target]
    stratify = y if problem == "classification" and y.nunique(dropna=True) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Random Forest Classifier":
        model = RandomForestClassifier(random_state=42)
    else:
        model = RandomForestRegressor(random_state=42)

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = float(accuracy_score(y_test, pred)) if problem == "classification" else float(r2_score(y_test, pred))

    st.session_state["model_payload"] = {
        "model_name": model_name,
        "problem_type": problem,
        "score": score,
        "metric": "accuracy" if problem == "classification" else "r2",
        "y_test": y_test,
        "pred": pred,
    }
    st.success("Model trained.")

with st.expander("Training artifacts", expanded=st.session_state.get("ui_mode") == "Learning"):
    payload = st.session_state.get("model_payload")
    if payload:
        st.metric(payload["metric"].upper(), f"{payload['score']:.4f}")
        preview = pd.DataFrame({"actual": payload["y_test"], "predicted": payload["pred"]})
        st.dataframe(preview.head(40), use_container_width=True)
