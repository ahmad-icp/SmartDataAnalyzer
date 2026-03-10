import streamlit as st
import pandas as pd
from pathlib import Path
from io import BytesIO
import uuid

from data_loading import load_dataframe, to_excel_bytes
from cleaning_tools import (
    fill_missing,
    remove_duplicates,
    standardize_text_columns,
    suggest_fuzzy_matches,
    apply_mapping,
    convert_types,
)
from visualization import make_chart, export_plotly_png
from ai_insights import generate_insights
from reporting import generate_html_report, html_to_pdf_bytes, upload_bytes_to_s3, generate_presigned_url
from modules.background_tasks import run_in_background, get_task_status
from modules.autosave import save_html_result
from feature_engineering import generate_basic_features
from tableau_publisher import publish_dataframe_to_tableau
from powerbi_publisher import publish_dataframe_to_powerbi

st.set_page_config(page_title="Smart Data Analyzer - Pro", layout="wide")

st.markdown("""
<style>
    .reportview-container .main .block-container{padding-top:1rem}
</style>
""", unsafe_allow_html=True)

st.title("Smart Data Analyzer - Pro")

with st.sidebar.expander("Upload & Settings", expanded=True):
    uploaded_file = st.file_uploader("Upload CSV / Excel / Parquet", type=["csv", "xlsx", "parquet"])
    sample_n = st.number_input("If large, sample rows (0 = no sampling)", min_value=0, value=0, step=100)
    row_limit = st.number_input("Row read limit (0 = no limit)", min_value=0, value=0, step=1000)
    st.markdown("---")
    st.markdown("Data is processed in-memory and not persisted.")


@st.cache_data
def _load(file, sample_n=0, limit=0):
    return load_dataframe(file, sample_n=sample_n, limit=limit)


if uploaded_file is not None:
    try:
        df = _load(uploaded_file, sample_n=int(sample_n), limit=int(row_limit))
        st.session_state["df_original"] = df.copy()
        st.session_state["df"] = df.copy()
    except Exception as e:
        st.error(f"Failed to read file: {e}")


tabs = st.tabs(["Data Overview", "Data Preview", "Cleaning", "Feature Engineering", "Dashboard", "AI Insights"])

# Data Overview & Profiling
with tabs[0]:
    st.header("Data Overview & Profiling")
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.info("Upload a dataset to see automatic profiling and overview.")
    else:
        try:
            from ydata_profiling import ProfileReport

            profile = ProfileReport(st.session_state["df"], minimal=True)
            html = profile.to_html()
            st.download_button("Download profiling HTML", data=html, file_name="profiling.html", mime="text/html")
            st.components.v1.html(html, height=600, scrolling=True)

            st.markdown("---")
            st.subheader("Background profiling")
            if st.button("Run profiling in background"):
                def _make_profile(d):
                    from ydata_profiling import ProfileReport as _PR

                    p = _PR(d, minimal=False)
                    return p.to_html()

                task_id = run_in_background(_make_profile, st.session_state["df"], task_name=f"profile_{uuid.uuid4().hex}")
                st.session_state["last_profile_task"] = task_id
                st.success(f"Submitted profiling job (task id: {task_id}). Refresh to check status.")

            if "last_profile_task" in st.session_state:
                tid = st.session_state["last_profile_task"]
                status, result = get_task_status(tid)
                st.write("Background task:", tid)
                if status == "running":
                    st.info("Profiling is still running in background. Refresh to check status.")
                elif status == "done":
                    st.success("Profiling complete. You can download the HTML below.")
                    if isinstance(result, str):
                        path, data = save_html_result(result)
                        st.write(f"Saved background result to: {path}")
                        st.download_button("Download background profiling HTML", data=data, file_name=Path(path).name, mime="text/html")
                        st.components.v1.html(result, height=600, scrolling=True)
                elif status == "error":
                    st.error(f"Background profiling failed: {result}")
                elif status == "cancelled":
                    st.warning("Background profiling was cancelled.")
                else:
                    st.write(status)

            # Reporting: generate downloadable HTML / PDF
            with st.expander("Generate report (HTML / PDF)"):
                extra_html = st.text_area("Optional notes to include in report")
                report_html = generate_html_report(html, extra_html)
                st.download_button("Download report HTML", data=report_html, file_name="report.html", mime="text/html")
                pdf_bytes = html_to_pdf_bytes(report_html)
                if pdf_bytes:
                    st.download_button("Download report PDF", data=pdf_bytes, file_name="report.pdf", mime="application/pdf")
                else:
                    st.info("PDF export not available (requires wkhtmltopdf installed). You can still download HTML.")

                # S3 upload
                st.markdown("**Upload report to S3 (optional)**")
                s3_bucket = st.text_input("S3 bucket (optional)")
                s3_key = st.text_input("S3 key (path) (optional)")
                if st.button("Upload HTML to S3") and s3_bucket and s3_key:
                    try:
                        upload_bytes_to_s3(s3_bucket, s3_key, report_html.encode("utf-8"))
                        url = generate_presigned_url(s3_bucket, s3_key)
                        st.success("Uploaded and generated presigned URL")
                        st.write(url)
                    except Exception as e:
                        st.error(f"S3 upload failed: {e}")

                # Tableau publishing
                st.markdown("**Publish report / datasource to Tableau**")
                dry_run = st.checkbox("Dry-run (simulate publish, no external calls)", value=True)
                tableau_server = st.text_input("Tableau Server URL (e.g. https://your-server)", value=st.secrets.get("TABLEAU_SERVER") if hasattr(st, "secrets") else "")
                tableau_site = st.text_input("Tableau site", value=st.secrets.get("TABLEAU_SITE") if hasattr(st, "secrets") else "")
                tableau_project = st.text_input("Tableau project name", value=st.secrets.get("TABLEAU_PROJECT") if hasattr(st, "secrets") else "")
                tableau_datasource = st.text_input("Datasource name", value="smartdata_ds")
                tableau_auth = st.selectbox("Auth method", ["pat", "userpass"])
                if tableau_auth == "pat":
                    token_name = st.text_input("Token name", value=st.secrets.get("TABLEAU_TOKEN_NAME") if hasattr(st, "secrets") else "")
                    token_value = st.text_input("Token value (keep secret)", value=st.secrets.get("TABLEAU_TOKEN_VALUE") if hasattr(st, "secrets") else "", type="password")
                    username = None
                    password = None
                else:
                    username = st.text_input("Tableau username", value=st.secrets.get("TABLEAU_USERNAME") if hasattr(st, "secrets") else "")
                    password = st.text_input("Tableau password", value=st.secrets.get("TABLEAU_PASSWORD") if hasattr(st, "secrets") else "", type="password")
                    token_name = token_value = None

                if st.button("Publish current dataframe to Tableau"):
                    try:
                        df_to_publish = st.session_state.get("df")
                        if df_to_publish is None:
                            st.error("No dataframe to publish.")
                        else:
                            if dry_run:
                                csv_bytes = df_to_publish.to_csv(index=False).encode("utf-8")
                                st.info("Dry-run: no external calls will be made. Showing payload preview.")
                                st.write("Tableau publish metadata:")
                                st.json({"server": tableau_server, "site": tableau_site, "project": tableau_project, "datasource": tableau_datasource, "auth_method": tableau_auth})
                                st.write("CSV sample:")
                                st.code(df_to_publish.head(5).to_csv(index=False))
                                st.download_button("Download CSV payload", data=csv_bytes, file_name="tableau_payload.csv", mime="text/csv")
                            else:
                                res = publish_dataframe_to_tableau(
                                    df_to_publish,
                                    server=tableau_server,
                                    site=tableau_site or "",
                                    project_name=tableau_project,
                                    datasource_name=tableau_datasource,
                                    auth_method=tableau_auth,
                                    token_name=token_name,
                                    token_value=token_value,
                                    username=username,
                                    password=password,
                                )
                                st.success("Published to Tableau")
                                st.json(res)
                    except Exception as e:
                        st.error(f"Tableau publish failed: {e}")

                # Power BI publishing
                st.markdown("**Publish dataset to Power BI (push dataset)**")
                p_tenant = st.text_input("Azure Tenant ID", value=st.secrets.get("PBI_TENANT") if hasattr(st, "secrets") else "")
                p_client = st.text_input("Azure Client ID", value=st.secrets.get("PBI_CLIENT_ID") if hasattr(st, "secrets") else "")
                p_secret = st.text_input("Azure Client Secret (keep secret)", value=st.secrets.get("PBI_CLIENT_SECRET") if hasattr(st, "secrets") else "", type="password")
                p_group = st.text_input("Power BI Workspace (group) ID", value=st.secrets.get("PBI_GROUP_ID") if hasattr(st, "secrets") else "")
                p_dataset = st.text_input("Dataset name", value="smartdata_dataset")
                if st.button("Publish to Power BI"):
                    try:
                        df_to_publish = st.session_state.get("df")
                        if df_to_publish is None:
                            st.error("No dataframe to publish.")
                        else:
                            if dry_run:
                                st.info("Dry-run: showing Power BI dataset payload (no external calls).")
                                cols = []
                                for col in df_to_publish.columns:
                                    dt = df_to_publish[col].dtype
                                    if pd.api.types.is_integer_dtype(dt):
                                        ptype = "Int64"
                                    elif pd.api.types.is_float_dtype(dt):
                                        ptype = "Double"
                                    elif pd.api.types.is_bool_dtype(dt):
                                        ptype = "Boolean"
                                    else:
                                        ptype = "string"
                                    cols.append({"name": col, "dataType": ptype})
                                payload = {"name": p_dataset, "defaultMode": "Push", "tables": [{"name": p_dataset, "columns": cols}]}
                                st.write("Dataset schema payload:")
                                st.json(payload)
                                st.write("Sample rows (first 5):")
                                st.json(df_to_publish.head(5).to_dict(orient="records"))
                                try:
                                    import json as _json

                                    st.download_button("Download Power BI payload (JSON)", data=_json.dumps(payload, indent=2).encode("utf-8"), file_name="powerbi_payload.json", mime="application/json")
                                except Exception:
                                    st.warning("Could not prepare download for payload.")
                            else:
                                res = publish_dataframe_to_powerbi(df_to_publish, tenant_id=p_tenant, client_id=p_client, client_secret=p_secret, group_id=p_group, dataset_name=p_dataset)
                                st.success("Published to Power BI")
                                st.json(res)
                    except Exception as e:
                        st.error(f"Power BI publish failed: {e}")
        # top-level profiling/overview error handler
        except Exception as e:
            st.error(f"Profiling failed: {e}")

# Data Preview tab
with tabs[1]:
    st.header("Editable Data Table")
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.info("Upload a dataset to preview and edit.")
    else:
        st.markdown("Use the table below to edit cells, add/delete rows, paste from Excel, or drag-fill.")
        edited = st.data_editor(st.session_state["df"], num_rows="dynamic", use_container_width=True)
        st.session_state["df"] = edited
        st.write(f"Rows: {len(edited)} - edits saved to session state")

# Cleaning tab
with tabs[2]:
    st.header("Cleaning Tools")
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.info("Upload data first.")
    else:
        df = st.session_state["df"]
        st.subheader("Quick actions")
        c1, c2 = st.columns(2)
        with c1:
            fill_opt = st.selectbox("Fill missing with", ["None", "Mean", "Median", "Mode", "Custom"])
            custom_val = None
            if fill_opt == "Custom":
                custom_val = st.text_input("Custom value (applies to all columns)")
            if st.button("Apply Fill Missing"):
                df = fill_missing(df, strategy=fill_opt.lower(), custom=custom_val)
                st.session_state["df"] = df
                st.success("Filled missing values.")
        with c2:
            if st.button("Remove exact duplicates"):
                df = remove_duplicates(df, fuzzy=False)
                st.session_state["df"] = df
                st.success("Exact duplicates removed.")
            if st.checkbox("Suggest fuzzy duplicates (slower)") and st.button("Run fuzzy suggestions"):
                suggestions = suggest_fuzzy_matches(df)
                st.session_state["suggestions"] = suggestions
                st.write("Suggested near-duplicate groups (sample):")
                st.write({k: v[:10] for k, v in suggestions.items()})
                if st.button("Apply suggestions (automated mapping)"):
                    mapping = {}
                    for col, pairs in suggestions.items():
                        for a, b, score in pairs:
                            mapping.setdefault(col, {})
                            mapping[col][b] = a
                    df = apply_mapping(df, mapping)
                    st.session_state["df"] = df
                    st.success("Applied suggested mappings.")

        st.subheader("Text standardization")
        text_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if text_cols:
            sel = st.selectbox("Text column to standardize", [None] + text_cols)
            if sel and st.button("Standardize selected column"):
                df = standardize_text_columns(df, [sel])
                st.session_state["df"] = df
                st.success(f"Standardized {sel}")

        st.subheader("Type conversions")
        col_to_conv = st.selectbox("Column to convert", [None] + df.columns.tolist())
        conv_to = st.selectbox("Convert to", ["numeric", "datetime", "category", "text"]) if col_to_conv else None
        if col_to_conv and st.button("Convert"):
            df = convert_types(df, {col_to_conv: conv_to})
            st.session_state["df"] = df
            st.success("Conversion applied.")

# Feature engineering
with tabs[3]:
    st.header("Feature Engineering")
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.info("Upload data to engineer features.")
    else:
        if st.button("Generate basic features (date parts, group aggs, transforms)"):
            st.session_state["df"] = generate_basic_features(st.session_state["df"]) or st.session_state["df"]
            st.success("Features generated")
        st.dataframe(st.session_state["df"].head())

# Dashboard builder
with tabs[4]:
    st.header("Dashboard Builder")
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.info("Upload data to create dashboards.")
    else:
        df = st.session_state["df"]
        cols = df.columns.tolist()
        x = st.selectbox("X column", [None] + cols)
        y = st.selectbox("Y column", [None] + cols)
        chart_type = st.selectbox("Chart type", ["scatter", "line", "bar", "histogram", "box", "heatmap"]) 
        with st.expander("Filters"):
            filters = {}
            for c in cols:
                if pd.api.types.is_numeric_dtype(df[c]):
                    lo, hi = float(df[c].min()), float(df[c].max())
                    filters[c] = st.slider(f"{c}", min_value=lo, max_value=hi, value=(lo, hi))
                else:
                    vals = df[c].dropna().unique().tolist()
                    filters[c] = st.multiselect(f"{c}", options=vals, default=vals[:5])
        dff = df.copy()
        for c, sel in filters.items():
            if isinstance(sel, tuple):
                dff = dff[(dff[c] >= sel[0]) & (dff[c] <= sel[1])]
            elif isinstance(sel, list) and sel:
                dff = dff[dff[c].isin(sel)]
        if st.button("Render Chart"):
            fig = make_chart(dff, x, y, chart_type)
            st.plotly_chart(fig, use_container_width=True)
            try:
                png = export_plotly_png(fig)
                st.download_button("Download chart PNG", data=png, file_name="chart.png", mime="image/png")
            except Exception:
                st.info("PNG export requires kaleido; ensure it's installed.")

# AI insights
with tabs[5]:
    st.header("AI-Driven Insights")
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.info("Upload data to get automatic insights.")
    else:
        if st.button("Suggest Insights"):
            out = generate_insights(st.session_state["df"]) or ["No insights available."]
            for s in out:
                st.write(s)

st.sidebar.markdown("---")
if "df" in st.session_state and st.session_state["df"] is not None:
    st.sidebar.download_button("Download cleaned CSV", data=st.session_state["df"].to_csv(index=False).encode("utf-8"), file_name="cleaned.csv", mime="text/csv")
    st.sidebar.download_button("Download cleaned Excel", data=to_excel_bytes(st.session_state["df"]), file_name="cleaned.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.sidebar.markdown("---")
st.sidebar.info("Security: Use Streamlit secrets for API keys. Data is not persisted by this app.")

if __name__ == "__main__":
    # run via `streamlit run app.py`
    pass

