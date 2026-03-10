from typing import Optional
import re
from html import unescape
from io import BytesIO

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import simpleSplit
    from reportlab.pdfgen import canvas
except Exception:
    letter = None
    simpleSplit = None
    canvas = None


def generate_html_report(profile_html: str, extra_html: str = "") -> str:
    """Combine profiling HTML and extra HTML into a standalone report string."""
    # Simple concatenation inside a container
    return f"<!doctype html><html><head><meta charset='utf-8'></head><body>{profile_html}<hr>{extra_html}</body></html>"


def _html_to_text(html: str) -> str:
    # Strip script/style and tags, then unescape entities.
    text = re.sub(r"(?is)<(script|style).*?>.*?</\\1>", "", html)
    text = re.sub(r"(?s)<.*?>", "", text)
    text = unescape(text)
    # Normalize whitespace
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join([ln for ln in lines if ln])


def html_to_pdf_bytes(html: str) -> Optional[bytes]:
    """Convert HTML string to PDF bytes using ReportLab.
    Returns None if conversion failed or ReportLab not available.
    """
    if canvas is None or letter is None or simpleSplit is None:
        return None
    try:
        text = _html_to_text(html)
        buf = BytesIO()
        c = canvas.Canvas(buf, pagesize=letter)
        width, height = letter
        margin = 72
        font_name = "Helvetica"
        font_size = 10
        line_height = 12
        max_width = width - (2 * margin)
        y = height - margin

        c.setFont(font_name, font_size)
        for raw_line in text.splitlines():
            if not raw_line.strip():
                y -= line_height
            else:
                wrapped = simpleSplit(raw_line, font_name, font_size, max_width)
                for line in wrapped:
                    c.drawString(margin, y, line)
                    y -= line_height
                    if y <= margin:
                        c.showPage()
                        c.setFont(font_name, font_size)
                        y = height - margin
            if y <= margin:
                c.showPage()
                c.setFont(font_name, font_size)
                y = height - margin

        c.save()
        return buf.getvalue()
    except Exception:
        return None


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
