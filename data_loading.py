import pandas as pd
from io import BytesIO


def load_dataframe(uploaded_file, sample_n: int = 0, limit: int = 0) -> pd.DataFrame:
    name = getattr(uploaded_file, "name", "uploaded")
    if name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.lower().endswith(".parquet"):
        df = pd.read_parquet(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    if limit and limit > 0:
        df = df.head(limit)
    if sample_n and sample_n > 0 and sample_n < len(df):
        df = df.sample(sample_n)
    return df


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    buf.seek(0)
    return buf.getvalue()
