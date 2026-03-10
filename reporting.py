from typing import Optional
import os
import tempfile
import pdfkit
import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError


def generate_html_report(profile_html: str, extra_html: str = "") -> str:
    """Combine profiling HTML and extra HTML into a standalone report string."""
    # Simple concatenation inside a container
    return f"<!doctype html><html><head><meta charset='utf-8'></head><body>{profile_html}<hr>{extra_html}</body></html>"


def html_to_pdf_bytes(html: str) -> Optional[bytes]:
    """Convert HTML string to PDF bytes using wkhtmltopdf (via pdfkit).
    Returns None if conversion failed or wkhtmltopdf not available.
    """
    try:
        # pdfkit requires wkhtmltopdf installed on the system
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            pdfkit.from_string(html, tmp.name)
            tmp.flush()
            tmp.seek(0)
            return tmp.read()
    except Exception:
        return None


def upload_bytes_to_s3(bucket: str, key: str, data: bytes, region: Optional[str] = None, acl: str = "private") -> dict:
    s3 = boto3.client("s3", region_name=region)
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=data, ACL=acl)
        return {"bucket": bucket, "key": key}
    except (BotoCoreError, NoCredentialsError) as e:
        raise


def generate_presigned_url(bucket: str, key: str, expires_in: int = 3600, region: Optional[str] = None) -> str:
    s3 = boto3.client("s3", region_name=region)
    return s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires_in)
