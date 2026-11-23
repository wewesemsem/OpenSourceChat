"""Resolve a file path and reject anything outside the allowlist.

Every read calls safe_path() again. A path is accepted only when its real
location is a regular file inside an allowlisted folder.
"""

from __future__ import annotations

import os
import stat
from fnmatch import fnmatchcase
from pathlib import Path

from app.config import FilesConfig


class FileAccessError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def safe_path(raw: str | Path, files: FilesConfig, *, kind: str = "file") -> Path:
    """Return the real path of one allowlisted file or directory.

    Relative paths are rejected so the process working directory never
    becomes an extra root. Symlinks are resolved before the allowlist check.
    """
    if not files.allowed_folders:
        raise FileAccessError("disabled")
    text = raw if isinstance(raw, str) else str(raw)
    if not text or "\x00" in text:
        raise FileAccessError("rejected")
    candidate = Path(text)
    if not candidate.is_absolute():
        raise FileAccessError("rejected")
    try:
        real = Path(os.path.realpath(candidate))
    except (OSError, ValueError):
        raise FileAccessError("rejected") from None
    folder = _containing_folder(real, files.allowed_folders)
    if folder is None:
        raise FileAccessError("rejected")
    if _excluded(real.relative_to(folder).parts, files.exclude_patterns):
        raise FileAccessError("excluded")
    try:
        info = real.stat()
    except OSError:
        raise FileAccessError("not_found") from None
    if kind == "dir":
        if not stat.S_ISDIR(info.st_mode):
            raise FileAccessError("not_a_file")
        return real
    if kind != "file":
        raise FileAccessError("rejected")
    extension = real.suffix.lower().lstrip(".")
    if extension not in files.allowed_extensions:
        raise FileAccessError("unsupported")
    if not stat.S_ISREG(info.st_mode):
        raise FileAccessError("not_a_file")
    if info.st_size > files.max_file_size_bytes:
        raise FileAccessError("too_large")
    return real


def indexable_directory(raw: str | Path, files: FilesConfig) -> Path:
    """Return the real path of a directory the indexer may walk.

    The allowlisted folder itself is accepted. Anything deeper must pass
    safe_path(). Symlinks are resolved first, so a link that leaves the
    allowlist is rejected before a file inside it can be opened.
    """
    if not files.allowed_folders:
        raise FileAccessError("disabled")
    text = raw if isinstance(raw, str) else str(raw)
    if not text or "\x00" in text:
        raise FileAccessError("rejected")
    candidate = Path(text)
    if not candidate.is_absolute():
        raise FileAccessError("rejected")
    try:
        real = Path(os.path.realpath(candidate))
    except (OSError, ValueError):
        raise FileAccessError("rejected") from None
    for folder in files.allowed_folders:
        if real == folder:
            if not real.exists():
                raise FileAccessError("not_found")
            if not real.is_dir():
                raise FileAccessError("not_a_file")
            return real
    return safe_path(real, files, kind="dir")


def screen_name(name: str, files: FilesConfig) -> str:
    """Keep a single file name. Reject paths, hidden names, and other excluded names."""
    if not isinstance(name, str) or not name or "\x00" in name or "/" in name or "\\" in name:
        raise FileAccessError("rejected")
    cleaned = Path(name).name
    if cleaned != name or cleaned in {".", ".."}:
        raise FileAccessError("rejected")
    if _excluded((cleaned,), files.exclude_patterns):
        raise FileAccessError("excluded")
    extension = Path(cleaned).suffix.lower().lstrip(".")
    if extension not in files.allowed_extensions:
        raise FileAccessError("unsupported")
    return cleaned


def _containing_folder(real: Path, folders: tuple[Path, ...]) -> Path | None:
    for folder in folders:
        try:
            relative = real.relative_to(folder)
        except ValueError:
            continue
        if relative.parts and ".." not in relative.parts:
            return folder
    return None


def _excluded(parts: tuple[str, ...], patterns: tuple[str, ...]) -> bool:
    for part in parts:
        if part in {".", ".."}:
            continue
        lowered = part.lower()
        for pattern in patterns:
            if fnmatchcase(lowered, pattern.lower()):
                return True
    return False
