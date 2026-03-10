# Smart Data Analyzer
A professional Streamlit-based data analysis platform. Key capabilities:

- Editable data table with `st.data_editor` (dynamic rows, paste from Excel, drag-fill).
- Advanced cleaning: fill missing (mean/median/mode/custom), deduplication (exact + fuzzy), text standardization, type conversion.
- Automated profiling via YData/Pandas-Profiling with downloadable HTML report.
- AI-driven insights (OpenAI optional) to summarize dataset, spot correlations/outliers, and suggest charts.
- Feature engineering (date parts, log, normalize, group aggregates).
- Dashboard builder with interactive filters and multi-chart export.
- Export cleaned data as CSV/Excel and charts as PNG.

Security & privacy

- No secrets are hard-coded. Put API keys (e.g., `OPENAI_API_KEY`) into Streamlit secrets or environment variables.
- Uploaded data is kept in-memory in the Streamlit session and not persisted by default. Do not upload sensitive data unless you control the deployment environment.

Installation

1. Create and activate a Python virtual environment:

```bash
python -m venv .venv
.
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run locally:

```bash
streamlit run app.py
```

Deployment

- Streamlit Community Cloud: create a new app, link to this repository, and add any secrets (OpenAI key) under the app settings.
- For enterprise deployment, run behind a secure HTTPS endpoint and follow your org's SOC2 policies. Use secrets management for API keys and do not embed them in source.

Streamlit Cloud secrets

- Open your app settings -> Secrets and paste your TOML (use `.streamlit/secrets.toml` as the template).
- The app reads `st.secrets` first, then environment variables for the same keys.

Docker

```bash
docker build -t smart-data-analyzer .
docker run -p 8501:8501 smart-data-analyzer
```

Integrations

- Tableau / Power BI: export cleaned datasets as CSV/Excel and import into Tableau or Power BI. For automated flows, write the cleaned file to a shared location or cloud storage (S3/GC Storage) and connect Tableau/Power BI to that source.

Notes

- PNG export of Plotly charts requires `kaleido`.
- PDF reports are generated with `reportlab` (no external binaries required).
- S3 uploads: provide AWS credentials via Streamlit secrets or environment variables. The app can upload generated reports and create presigned URLs which you can use in Tableau/Power BI as a data source.

Tableau & Power BI publishing

- Tableau: this repo includes `tableau_publisher.py` which uses `tableauserverclient` and optionally `tableauhyperapi` to publish a DataFrame as a datasource (best-effort). Install `tableauserverclient` and, for reliable datasource publishing, install `tableauhyperapi` and `hyper` runtime following Tableau's docs.
- Power BI: `powerbi_publisher.py` demonstrates creating a push dataset and ingesting rows via the Power BI REST API. Requires an Azure AD app (client id/secret) or delegated auth; see MSAL docs and Power BI REST API docs for permissions.

Security note: never commit service credentials in source. Use Streamlit secrets or environment variables to store AWS, Tableau, or Azure credentials.

If you'd like, I can also:

- Add S3 upload support, or
- Wire direct Tableau/Power BI publishing steps.
