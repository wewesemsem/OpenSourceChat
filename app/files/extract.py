"""Extract text from an allowlisted file. Failures stay per file.

txt, md, and csv use the standard library. docx is a zip of XML, read with
zipfile and ElementTree so a document library is unnecessary. pdf uses pypdf
because PDF text extraction is not practical without a parser.
"""

from __future__ import annotations

import csv
import logging
import re
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

WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WORD_TEXT = f"{{{WORD_NS}}}t"
WORD_PARA = f"{{{WORD_NS}}}p"
WORD_PPR = f"{{{WORD_NS}}}pPr"
WORD_STYLE = f"{{{WORD_NS}}}pStyle"
WORD_VAL = f"{{{WORD_NS}}}val"
_HEADING = re.compile(r"^(#{1,6})[ \t]+(\S.*?)\s*$")
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


@dataclass(frozen=True)
class Segment:
    """One heading, page, or table row. Offsets stay inside the extracted text."""

    text: str
    page: int | None = None
    heading: str | None = None


@dataclass(frozen=True)
class SegmentedText:
    name: str
    kind: str
    parts: tuple[Segment, ...]
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


def read_segments(raw: str | Path, files: FilesConfig) -> SegmentedText:
    """Extract structured text. safe_path() runs immediately before the file is opened."""
    name = Path(str(raw)).name
    try:
        real = safe_path(raw, files)
    except FileAccessError as exc:
        return SegmentedText(name=name, kind="text", parts=(), error_code=exc.code)
    return segments_opened(real, files)


def segments_opened(real: Path, files: FilesConfig) -> SegmentedText:
    """Segment a path the caller has already checked."""
    try:
        kind, parts = _within_timeout(lambda: _segment_real(real, files), files.extract_timeout_seconds)
    except FileAccessError as exc:
        return SegmentedText(name=real.name, kind="text", parts=(), error_code=exc.code)
    return SegmentedText(name=real.name, kind=kind, parts=tuple(parts), error_code=None)


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
        return "\n".join(_csv_rows(path, files.max_extract_chars))
    if kind == "docx":
        return "\n".join(text for _, text in _docx_paragraphs(path, files.max_extract_chars, files.max_file_size_bytes))
    if kind == "pdf":
        return "\n\n".join(page for page in _pdf_pages(path, files.max_pages, files.max_extract_chars) if page)
    raise FileAccessError("unsupported")


def _segment_real(path: Path, files: FilesConfig) -> tuple[str, list[Segment]]:
    kind = path.suffix.lower().lstrip(".")
    if kind == "txt":
        text = _plain(path, files.max_extract_chars)
        return "text", [Segment(text=text)] if text.strip() else []
    if kind == "md":
        text = _plain(path, files.max_extract_chars)
        return "text", _markdown_segments(text)
    if kind == "csv":
        rows = _csv_rows(path, files.max_extract_chars)
        return "table", [Segment(text=row) for row in rows]
    if kind == "docx":
        return "text", _docx_segments(path, files.max_extract_chars, files.max_file_size_bytes)
    if kind == "pdf":
        pages = _pdf_pages(path, files.max_pages, files.max_extract_chars)
        return "text", [
            Segment(text=page, page=number)
            for number, page in enumerate(pages, start=1)
            if page.strip()
        ]
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


def _csv_rows(path: Path, max_chars: int) -> list[str]:
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
    return lines


def _docx_paragraphs(path: Path, max_chars: int, max_bytes: int) -> list[tuple[str, str]]:
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
        paragraphs.append((_paragraph_style(paragraph), line))
    return paragraphs


def _docx_segments(path: Path, max_chars: int, max_bytes: int) -> list[Segment]:
    heading: str | None = None
    saw_body = False
    segments: list[Segment] = []
    for style, text in _docx_paragraphs(path, max_chars, max_bytes):
        if _is_heading_style(style):
            if heading and not saw_body:
                segments.append(Segment(text="", heading=heading))
            heading = text
            saw_body = False
            continue
        segments.append(Segment(text=text, heading=heading))
        saw_body = True
    if heading and not saw_body:
        segments.append(Segment(text="", heading=heading))
    return segments


def _paragraph_style(paragraph: ElementTree.Element) -> str:
    properties = paragraph.find(WORD_PPR)
    if properties is None:
        return ""
    style = properties.find(WORD_STYLE)
    if style is None:
        return ""
    return style.get(WORD_VAL, "")


def _is_heading_style(style: str) -> bool:
    lowered = style.lower()
    return lowered == "title" or lowered.startswith("heading")


def _markdown_segments(text: str) -> list[Segment]:
    if not text.strip():
        return []
    if not re.search(r"(?m)^#{1,6}[ \t]+\S", text):
        return [Segment(text=text)]
    segments: list[Segment] = []
    heading: str | None = None
    body: list[str] = []

    def flush() -> None:
        nonlocal body
        content = "\n".join(body).strip()
        if content or heading:
            segments.append(Segment(text=content, heading=heading))
        body = []

    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            flush()
            heading = match.group(2).strip()
            continue
        body.append(line)
    flush()
    return segments


def _pdf_pages(path: Path, max_pages: int, max_chars: int) -> list[str]:
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
    pages = []
    total = 0
    for page in reader.pages:
        try:
            piece = page.extract_text() or ""
        except Exception:
            raise FileAccessError("corrupt") from None
        total += len(piece) + 2
        if total > max_chars:
            raise FileAccessError("too_long")
        pages.append(piece)
    return pages


def _check_length(text: str, max_chars: int) -> None:
    if len(text) > max_chars:
        raise FileAccessError("too_long")
