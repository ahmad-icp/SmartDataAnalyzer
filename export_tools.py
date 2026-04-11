from io import BytesIO
import pandas as pd


def export_df_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def export_df_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    buf.seek(0)
    return buf.getvalue()


def export_df_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    return buf.getvalue()


__all__ = ["export_df_csv_bytes", "export_df_excel_bytes", "export_df_parquet_bytes"]
