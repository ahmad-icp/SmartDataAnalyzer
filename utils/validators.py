"""Validation utilities for SmartDataAnalyzer."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import pandas as pd

MAX_FILE_SIZE_MB = 25
ALLOWED_EXTENSIONS = {"csv"}


@dataclass
class ValidationResult:
    """Validation status and message."""

    is_valid: bool
    message: str = ""


def validate_uploaded_file(filename: str, data: bytes, max_size_mb: int = MAX_FILE_SIZE_MB) -> ValidationResult:
    """Validate upload extension and size."""
    if "." not in filename:
        return ValidationResult(False, "File must have an extension.")

    ext = filename.rsplit(".", 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return ValidationResult(False, "Only CSV files are supported.")

    size_mb = len(data) / (1024 * 1024)
    if size_mb > max_size_mb:
        return ValidationResult(False, f"File exceeds {max_size_mb}MB size limit.")

    return ValidationResult(True, "")


def load_csv_bytes(data: bytes) -> pd.DataFrame:
    """Load CSV content from bytes with safe defaults."""
    return pd.read_csv(BytesIO(data))
