"""Helpers to push DataFrame to Power BI as a push dataset using the Power BI REST API.

This implements a simple push dataset creation and ingestion flow using an
Azure AD app (client credentials). Requires that the service principal has
access to the Power BI workspace or the app uses delegated permission.
"""
import requests
import pandas as pd
from typing import Optional

try:
    import msal
except Exception:
    msal = None


def _ensure_msal():
    if msal is None:
        raise ImportError("msal is required for Power BI auth. Install with `pip install msal`.")
    return msal


def get_powerbi_token(tenant_id: str, client_id: str, client_secret: str, scope: str = "https://analysis.windows.net/powerbi/api/.default") -> str:
    msal_mod = _ensure_msal()
    app = msal_mod.ConfidentialClientApplication(client_id, authority=f"https://login.microsoftonline.com/{tenant_id}", client_credential=client_secret)
    resp = app.acquire_token_for_client(scopes=[scope])
    if "access_token" not in resp:
        raise RuntimeError(f"Failed to acquire token: {resp}")
    return resp["access_token"]


def create_push_dataset(token: str, group_id: str, dataset_name: str, df: pd.DataFrame) -> dict:
    url = f"https://api.powerbi.com/v1.0/myorg/groups/{group_id}/datasets"
    # build simple dataset schema from df
    columns = []
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_integer_dtype(dtype):
            ptype = "Int64"
        elif pd.api.types.is_float_dtype(dtype):
            ptype = "Double"
        elif pd.api.types.is_bool_dtype(dtype):
            ptype = "Boolean"
        else:
            ptype = "string"
        columns.append({"name": col, "dataType": ptype})

    payload = {
        "name": dataset_name,
        "defaultMode": "Push",
        "tables": [{"name": dataset_name, "columns": columns}],
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    r = requests.post(url, json=payload, headers=headers)
    r.raise_for_status()
    return r.json()


def push_rows_to_dataset(token: str, group_id: str, dataset_id: str, table_name: str, df: pd.DataFrame) -> dict:
    url = f"https://api.powerbi.com/v1.0/myorg/groups/{group_id}/datasets/{dataset_id}/tables/{table_name}/rows"
    rows = df.fillna(None).to_dict(orient="records")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"rows": rows}
    r = requests.post(url, json=payload, headers=headers)
    r.raise_for_status()
    return r.json()


def publish_dataframe_to_powerbi(df: pd.DataFrame, tenant_id: str, client_id: str, client_secret: str, group_id: str, dataset_name: str):
    token = get_powerbi_token(tenant_id, client_id, client_secret)
    created = create_push_dataset(token, group_id, dataset_name, df)
    dataset_id = created.get("id")
    push_result = push_rows_to_dataset(token, group_id, dataset_id, dataset_name, df)
    return {"dataset": created, "push": push_result}
