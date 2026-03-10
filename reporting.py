from typing import Optional
import os
import tempfile
import shutil

try:
    import pdfkit
except Exception:
    pdfkit = None


def generate_html_report(profile_html: str, extra_html: str = "") -> str:
    """Combine profiling HTML and extra HTML into a standalone report string."""
    # Simple concatenation inside a container
    return f"<!doctype html><html><head><meta charset='utf-8'></head><body>{profile_html}<hr>{extra_html}</body></html>"


def _resolve_wkhtmltopdf() -> Optional[str]:
    path = os.environ.get("WKHTMLTOPDF_PATH") or os.environ.get("WKHTMLTOPDF")
    if path and os.path.exists(path):
        return path
    return shutil.which("wkhtmltopdf")


def html_to_pdf_bytes(html: str) -> Optional[bytes]:
    """Convert HTML string to PDF bytes using wkhtmltopdf (via pdfkit).
    Returns None if conversion failed or wkhtmltopdf not available.
    """
    if pdfkit is None:
        return None
    tmp_path = None
    try:
        # pdfkit requires wkhtmltopdf installed on the system
        wk_path = _resolve_wkhtmltopdf()
        config = pdfkit.configuration(wkhtmltopdf=wk_path) if wk_path else None
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp_path = tmp.name
        tmp.close()
        if config is None:
            pdfkit.from_string(html, tmp_path)
        else:
            pdfkit.from_string(html, tmp_path, configuration=config)
        with open(tmp_path, "rb") as f:
            return f.read()
    except Exception:
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def upload_bytes_to_s3(bucket: str, key: str, data: bytes, region: Optional[str] = None, acl: str = "private") -> dict:
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, NoCredentialsError
    except Exception as e:
        raise ImportError("boto3 is required for S3 uploads. Install with `pip install boto3`.") from e

    s3 = boto3.client("s3", region_name=region)
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=data, ACL=acl)
        return {"bucket": bucket, "key": key}
    except (BotoCoreError, NoCredentialsError) as e:
        raise


def generate_presigned_url(bucket: str, key: str, expires_in: int = 3600, region: Optional[str] = None) -> str:
    try:
        import boto3
    except Exception as e:
        raise ImportError("boto3 is required for S3 presigned URLs. Install with `pip install boto3`.") from e

    s3 = boto3.client("s3", region_name=region)
    return s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires_in)
