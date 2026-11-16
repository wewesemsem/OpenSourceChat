"""Extract text from an allowlisted file. Failures stay per file.

txt, md, and csv use the standard library. docx is a zip of XML, read with
zipfile and ElementTree so a document library is unnecessary. pdf uses pypdf
because PDF text extraction is not practical without a parser.
"""

from __future__ import annotations

import csv
import logging
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.config import FilesConfig
from app.files.safe import FileAccessError, safe_path

# pypdf prints the leading bytes of a corrupt file. Keep that off the console.
logging.getLogger("pypdf").setLevel(logging.ERROR)

WORD_TEXT = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
WORD_PARA = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"
ERROR_CODES = (
    "disabled",
    "rejected",
    "excluded",
    "too_large",
    "unsupported",
    "corrupt",
    "timeout",
    "too_many_pages",
    "too_long",
    "not_found",
    "not_a_file",
)


@dataclass(frozen=True)
class ExtractResult:
    name: str
    text: str
    error_code: str | None

    @property
    def ok(self) -> bool:
        return self.error_code is None


def message_for(code: str, strings: dict[str, str]) -> str:
    return strings[f"file_{code}"]


def read_text(raw: str | Path, files: FilesConfig) -> ExtractResult:
    """Read one file. safe_path() runs immediately before the file is opened."""
    name = Path(str(raw)).name
    try:
        real = safe_path(raw, files)
    except FileAccessError as exc:
        return ExtractResult(name=name, text="", error_code=exc.code)
    return read_opened(real, files)


def read_opened(real: Path, files: FilesConfig) -> ExtractResult:
    """Extract a path the caller has already checked."""
    try:
        text = _within_timeout(lambda: _extract_real(real, files), files.extract_timeout_seconds)
    except FileAccessError as exc:
        return ExtractResult(name=real.name, text="", error_code=exc.code)
    return ExtractResult(name=real.name, text=text, error_code=None)


def _within_timeout(func, timeout: float):
    """Wait up to timeout. A slow parser is abandoned, not killed.

    Python cannot stop a running thread. The file-size cap is what bounds
    how much that thread can read.
    """
    box: dict = {}

    def run() -> None:
        try:
            box["value"] = func()
        except Exception as exc:
            box["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise FileAccessError("timeout")
    if "error" in box:
        error = box["error"]
        if isinstance(error, FileAccessError):
            raise error
        raise FileAccessError("corrupt") from None
    return box.get("value", "")


def _extract_real(path: Path, files: FilesConfig) -> str:
    kind = path.suffix.lower().lstrip(".")
    if kind in {"txt", "md"}:
        return _plain(path, files.max_extract_chars)
    if kind == "csv":
        return _csv(path, files.max_extract_chars)
    if kind == "docx":
        return _docx(path, files.max_extract_chars, files.max_file_size_bytes)
    if kind == "pdf":
        return _pdf(path, files.max_pages, files.max_extract_chars)
    raise FileAccessError("unsupported")


def _plain(path: Path, max_chars: int) -> str:
    data = path.read_bytes()
    if b"\x00" in data:
        raise FileAccessError("corrupt")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise FileAccessError("corrupt") from None
    _check_length(text, max_chars)
    return text


def _csv(path: Path, max_chars: int) -> str:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            lines = []
            total = 0
            for row in csv.reader(handle):
                line = "\t".join(row)
                total += len(line) + 1
                if total > max_chars:
                    raise FileAccessError("too_long")
                lines.append(line)
    except FileAccessError:
        raise
    except (UnicodeDecodeError, csv.Error):
        raise FileAccessError("corrupt") from None
    return "\n".join(lines)


def _docx(path: Path, max_chars: int, max_bytes: int) -> str:
    try:
        with zipfile.ZipFile(path) as package:
            info = package.getinfo("word/document.xml")
            if info.file_size > max_bytes:
                raise FileAccessError("too_large")
            xml = package.read("word/document.xml")
    except FileAccessError:
        raise
    except (zipfile.BadZipFile, KeyError, OSError):
        raise FileAccessError("corrupt") from None
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError:
        raise FileAccessError("corrupt") from None
    paragraphs = []
    total = 0
    for paragraph in root.iter(WORD_PARA):
        line = "".join(node.text or "" for node in paragraph.iter(WORD_TEXT)).strip()
        if not line:
            continue
        total += len(line) + 1
        if total > max_chars:
            raise FileAccessError("too_long")
        paragraphs.append(line)
    return "\n".join(paragraphs)


def _pdf(path: Path, max_pages: int, max_chars: int) -> str:
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            raise FileAccessError("corrupt")
        count = len(reader.pages)
    except FileAccessError:
        raise
    except (PdfReadError, OSError, ValueError):
        raise FileAccessError("corrupt") from None
    except Exception:
        raise FileAccessError("corrupt") from None
    if count > max_pages:
        raise FileAccessError("too_many_pages")
    parts = []
    total = 0
    for page in reader.pages:
        try:
            piece = page.extract_text() or ""
        except Exception:
            raise FileAccessError("corrupt") from None
        total += len(piece) + 2
        if total > max_chars:
            raise FileAccessError("too_long")
        parts.append(piece)
    return "\n\n".join(part for part in parts if part)


def _check_length(text: str, max_chars: int) -> None:
    if len(text) > max_chars:
        raise FileAccessError("too_long")
