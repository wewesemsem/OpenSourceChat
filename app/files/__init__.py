"""Read-only access to allowlisted local files."""

from app.files.extract import ERROR_CODES, ExtractResult, message_for, read_text
from app.files.safe import FileAccessError, safe_path

__all__ = [
    "ERROR_CODES",
    "ExtractResult",
    "FileAccessError",
    "message_for",
    "read_text",
    "safe_path",
]
