"""Autosave helper to persist string results (e.g., HTML) to a temporary file
and return the file path and bytes for download.
"""
from pathlib import Path
from typing import Tuple

TMP_DIR = Path(".tmp")
TMP_DIR.mkdir(exist_ok=True)


def save_html_result(html: str, filename: str | None = None) -> Tuple[str, bytes]:
    """Save HTML string to a temporary file and return (path, bytes).

    If filename is not provided a unique name is generated.
    """
    if filename is None:
        import uuid

        filename = f"result_{uuid.uuid4().hex}.html"
    path = TMP_DIR / filename
    data = html.encode("utf-8")
    path.write_bytes(data)
    return str(path), data


def read_bytes(path: str) -> bytes:
    p = Path(path)
    return p.read_bytes()
