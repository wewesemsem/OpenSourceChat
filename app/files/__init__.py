"""Read-only access to allowlisted local files."""

from app.files.extract import ERROR_CODES, ExtractResult, message_for, read_text
from app.files.safe import FileAccessError, indexable_directory, safe_path

__all__ = [
    "ERROR_CODES",
    "ExtractResult",
    "FileAccessError",
    "indexable_directory",
    "message_for",
    "read_text",
    "safe_path",
]
