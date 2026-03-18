# 🔥 SmartDataAnalyzer Pro: Intelligent Data Analysis Platform

SmartDataAnalyzer Pro is a production-style analytics platform that transforms a raw CSV into actionable intelligence through automated diagnostics, data quality scoring, model selection, explainability, and uncertainty-aware predictions.

## Problem Statement

Teams often spend most of their project time manually inspecting messy datasets before they can model anything useful. SmartDataAnalyzer Pro reduces this friction by orchestrating a guided, end-to-end workflow that surfaces quality issues, recommendations, and model insights in one professional interface.

## Key Features

- **7-step guided workflow UI** (Upload → EDA → Cleaning → Correction → Feature Engineering → Model Training → Insights & Results).
- **Automated Insight Engine** that generates human-readable data, model, and feature insights.
- **Data Quality Score (0–100)** with penalty breakdown for missing values, duplicates, outliers, and inconsistent categories.
- **Smart recommendations** for imputation strategy, correlated feature handling, and model family suitability.
- **AutoML-lite training pipeline** for both classification and regression.
- **Evaluation suite** with confusion matrix / RMSE-MAE and overfitting signals.
- **SHAP explainability** for global and local model interpretation.
- **Uncertainty estimation** using Bayesian Ridge prediction intervals.
- **Interactive visual analytics** powered by Plotly.

## System Architecture

```text
SmartDataAnalyzer/
├── app/
│   ├── main.py                # Product-grade Streamlit workflow
│   └── ui_components.py       # Reusable UI blocks and progress flow
├── core/
│   ├── cleaning.py            # Data cleaning + recommendations
│   ├── correction.py          # Fuzzy category normalization
│   ├── data_quality.py        # Quality score computation (0-100)
│   ├── eda.py                 # EDA summaries and diagnostics
│   ├── evaluation.py          # Task-specific metrics
│   ├── explainability.py      # SHAP integration
│   ├── feature_engineering.py # Encoding/scaling/feature diagnostics
│   ├── insights.py            # Automated human-readable insight engine
│   ├── model_selection.py     # AutoML-lite model benchmarking
│   └── uncertainty.py         # Bayesian uncertainty intervals
├── utils/
│   ├── helpers.py
│   ├── validators.py
│   └── visualization.py
└── tests/
    ├── test_cleaning.py
    ├── test_correction.py
    ├── test_model_selection.py
    ├── test_insights.py
    └── test_data_quality.py
```

## Screenshots (Placeholders)

- `docs/screenshots/01_workflow_overview.png`
- `docs/screenshots/02_data_quality_dashboard.png`
- `docs/screenshots/03_model_insights.png`

## Demo Instructions

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/main.py
```

## Tech Stack

- **Frontend:** Streamlit
- **Data:** pandas, numpy
- **ML:** scikit-learn
- **Explainability:** SHAP
- **Visualization:** Plotly, seaborn/matplotlib (optional fallback)
- **Matching:** rapidfuzz
- **Testing:** pytest

## Future Improvements

- Drift monitoring and automated retraining triggers
- Experiment tracking (MLflow)
- Role-based authentication and project workspaces
- LLM-assisted narrative reporting and executive summaries
