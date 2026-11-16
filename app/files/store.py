"""Copies of files the user picked. The original path is never stored.

The browser file window does not tell the page where a file lives. The
bytes are saved under the app data directory and read from there later.
"""

from __future__ import annotations

import os
import shutil
import stat
import uuid
from pathlib import Path

from app.config import FilesConfig
from app.files.safe import FileAccessError, screen_name


def uploads_root(data_dir: Path, *, create: bool = False) -> Path:
    root = (data_dir / "uploads").resolve()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def save_upload(data_dir: Path, filename: str, payload: bytes, files: FilesConfig) -> tuple[str, Path]:
    name = screen_name(filename, files)
    if len(payload) > files.max_file_size_bytes:
        raise FileAccessError("too_large")
    upload_id = str(uuid.uuid4())
    folder = uploads_root(data_dir, create=True) / upload_id
    folder.mkdir()
    path = folder / name
    path.write_bytes(payload)
    return upload_id, path


def find_upload(data_dir: Path, upload_id: str) -> Path | None:
    folder = _folder(data_dir, upload_id)
    if folder is None or not folder.is_dir():
        return None
    try:
        children = [item for item in folder.iterdir() if item.is_file() and not item.is_symlink()]
    except OSError:
        return None
    if len(children) != 1:
        return None
    try:
        return stored_file(children[0], uploads_root(data_dir), files=None)
    except FileAccessError:
        return None


def stored_file(raw: str | Path, root: Path, files: FilesConfig | None) -> Path:
    """Accept a file only when its real path stays inside root."""
    text = raw if isinstance(raw, str) else str(raw)
    if not text or "\x00" in text:
        raise FileAccessError("rejected")
    candidate = Path(text)
    if not candidate.is_absolute():
        raise FileAccessError("rejected")
    try:
        real = Path(os.path.realpath(candidate))
        root_real = Path(os.path.realpath(root))
        relative = real.relative_to(root_real)
    except (OSError, ValueError):
        raise FileAccessError("rejected") from None
    if not relative.parts or ".." in relative.parts:
        raise FileAccessError("rejected")
    try:
        info = real.stat()
    except OSError:
        raise FileAccessError("not_found") from None
    if not stat.S_ISREG(info.st_mode):
        raise FileAccessError("not_a_file")
    if files is not None:
        screen_name(real.name, files)
        if info.st_size > files.max_file_size_bytes:
            raise FileAccessError("too_large")
    return real


def upload_id_of(path: str | Path, data_dir: Path) -> str | None:
    try:
        relative = Path(os.path.realpath(path)).relative_to(uploads_root(data_dir))
    except (OSError, ValueError):
        return None
    if len(relative.parts) < 2:
        return None
    ident = relative.parts[0]
    if _folder(data_dir, ident) is None:
        return None
    return ident


def discard_upload(data_dir: Path, upload_id: str) -> bool:
    folder = _folder(data_dir, upload_id)
    if folder is None or not folder.exists():
        return False
    shutil.rmtree(folder)
    return True


def _folder(data_dir: Path, upload_id: str) -> Path | None:
    try:
        parsed = uuid.UUID(upload_id)
    except (TypeError, ValueError):
        return None
    if str(parsed) != upload_id:
        return None
    root = uploads_root(data_dir)
    folder = root / upload_id
    try:
        real = Path(os.path.realpath(folder))
        real.relative_to(root)
    except (OSError, ValueError):
        return None
    if real == root:
        return None
    return real
