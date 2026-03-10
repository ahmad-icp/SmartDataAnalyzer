"""Helpers to publish DataFrame to Tableau Server using the Tableau Server Client (TSC).

This module provides a helper that exports a DataFrame to CSV (or .hyper if
the Hyper API is available) and attempts to publish it as a datasource to
Tableau Server / Tableau Online using a Personal Access Token or username/password.

Notes:
- Requires `tableauserverclient` installed. For .hyper creation, `tableauhyperapi` is required.
- Publishing directly from CSV may not be supported on some Tableau versions; creating a .hyper is more reliable.
"""
from typing import Optional
import tempfile
import os
import pandas as pd


def _ensure_tsc():
    try:
        import tableauserverclient as TSC

        return TSC
    except Exception as e:
        raise ImportError("tableauserverclient is required: pip install tableauserverclient") from e


def _maybe_create_hyper(df: pd.DataFrame) -> str:
    """Attempt to create a .hyper from DataFrame if tableauhyperapi is installed.
    Returns path to created hyper or raises if unavailable.
    """
    try:
        from tableauhyperapi import HyperProcess, Connection, TableDefinition, SqlType, Telemetry, Inserter
    except Exception:
        raise ImportError("To publish reliably as a datasource you should install tableauhyperapi to create .hyper files")

    tmp = tempfile.NamedTemporaryFile(suffix=".hyper", delete=False)
    tmp_path = tmp.name
    tmp.close()

    # Simple schema: map pandas dtypes to Hyper types (best-effort)
    def _col_type(series: pd.Series):
        if pd.api.types.is_integer_dtype(series):
            return SqlType.big_int()
        if pd.api.types.is_float_dtype(series):
            return SqlType.double()
        if pd.api.types.is_bool_dtype(series):
            return SqlType.bool()
        # fallback
        return SqlType.text()

    table_name = "Extract"
    cols = [TableDefinition.Column(name, _col_type(df[name])) for name in df.columns]

    with HyperProcess(telemetry=Telemetry.SEND_USAGE_DATA_TO_TABLEAU) as hyper:
        with Connection(endpoint=hyper.endpoint, database=tmp_path) as connection:
            schema = TableDefinition(table_name=table_name, columns=cols)
            connection.catalog.create_table(schema)
            with Inserter(connection, schema) as inserter:
                inserter.add_rows(df.itertuples(index=False, name=None))
                inserter.execute()

    return tmp_path


def publish_dataframe_to_tableau(
    df: pd.DataFrame,
    server: str,
    site: str,
    project_name: str,
    datasource_name: str,
    auth_method: str = "pat",
    token_name: Optional[str] = None,
    token_value: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    create_hyper: bool = True,
) -> dict:
    """Publish a DataFrame as a datasource to Tableau Server.

    Returns a dict with publish result or raises an informative exception.
    """
    TSC = _ensure_tsc()

    if auth_method == "pat":
        if not token_name or not token_value:
            raise ValueError("token_name and token_value are required for PAT auth")
        auth = TSC.PersonalAccessTokenAuth(token_name, token_value, site)
    else:
        if not username or not password:
            raise ValueError("username and password required for basic auth")
        auth = TSC.TableauAuth(username, password, site)

    server_obj = TSC.Server(server, use_server_version=True)

    # create a file to publish: hyper preferred, fallback to csv
    if create_hyper:
        try:
            file_path = _maybe_create_hyper(df)
            file_type = "hyper"
        except Exception:
            # fallback to csv
            file_path = _df_to_temp_csv(df)
            file_type = "csv"
    else:
        file_path = _df_to_temp_csv(df)
        file_type = "csv"

    try:
        with server_obj.auth.sign_in(auth):
            all_projects, pagination_item = server_obj.projects.get()
            project = next((p for p in all_projects if p.name == project_name), None)
            if project is None:
                raise RuntimeError(f"Project not found: {project_name}")

            new_datasource = TSC.DatasourceItem(project.id, name=datasource_name)
            publish_mode = TSC.Server.PublishMode.Overwrite
            datasource = server_obj.datasources.publish(new_datasource, file_path, publish_mode)
            return {"id": datasource.id, "content_url": datasource.content_url}
    finally:
        try:
            os.remove(file_path)
        except Exception:
            pass


def _df_to_temp_csv(df: pd.DataFrame) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    path = tmp.name
    tmp.close()
    df.to_csv(path, index=False)
    return path
