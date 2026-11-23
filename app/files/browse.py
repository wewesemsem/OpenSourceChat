"""List allowlisted folders. Callers pick an id this module issued."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from app.config import FilesConfig
from app.files.safe import FileAccessError, safe_path


def file_id(path: Path) -> str:
    return hashlib.sha256(str(path).encode()).hexdigest()[:32]


def list_entries(files: FilesConfig, folder_id: str | None = None) -> dict:
    if not files.allowed_folders:
        return {"enabled": False, "parent_id": None, "truncated": False, "entries": []}
    if folder_id is None:
        entries = []
        for folder in files.allowed_folders:
            if not folder.is_dir():
                continue
            real = Path(os.path.realpath(folder))
            entries.append({"id": file_id(real), "name": real.name, "kind": "folder"})
        return _page(entries, files, enabled=True, parent_id=None)
    folder = find_directory(files, folder_id)
    if folder is None:
        raise FileAccessError("rejected")
    entries = []
    try:
        children = sorted(os.scandir(folder), key=lambda entry: entry.name.lower())
    except OSError:
        children = []
    for child in children:
        if child.is_symlink():
            continue
        if child.is_dir(follow_symlinks=False):
            try:
                real = safe_path(child.path, files, kind="dir")
            except FileAccessError:
                continue
            entries.append({"id": file_id(real), "name": real.name, "kind": "folder"})
            continue
        if child.is_file(follow_symlinks=False):
            try:
                real = safe_path(child.path, files)
            except FileAccessError:
                continue
            entries.append({"id": file_id(real), "name": real.name, "kind": "file"})
    entries.sort(key=lambda item: (item["kind"] != "folder", item["name"].lower()))
    return _page(entries, files, enabled=True, parent_id=_parent_id(folder, files))


def resolve_file(files: FilesConfig, wanted: str) -> Path | None:
    """Find a file id by walking the allowlist. Unknown ids are not opened."""
    for root in files.allowed_folders:
        if not root.is_dir():
            continue
        found = _search(Path(os.path.realpath(root)), files, wanted, depth=1)
        if found is not None:
            return found
    return None


def find_directory(files: FilesConfig, wanted: str) -> Path | None:
    for root in files.allowed_folders:
        if not root.is_dir():
            continue
        real = Path(os.path.realpath(root))
        if file_id(real) == wanted:
            return real
        found = _search_dir(real, files, wanted, depth=1)
        if found is not None:
            return found
    return None


def _search(directory: Path, files: FilesConfig, wanted: str, depth: int) -> Path | None:
    if depth > files.max_list_depth:
        return None
    try:
        children = list(os.scandir(directory))
    except OSError:
        return None
    for child in children:
        if child.is_symlink():
            continue
        if child.is_file(follow_symlinks=False):
            try:
                real = safe_path(child.path, files)
            except FileAccessError:
                continue
            if file_id(real) == wanted:
                return real
        elif child.is_dir(follow_symlinks=False) and depth < files.max_list_depth:
            try:
                nested = safe_path(child.path, files, kind="dir")
            except FileAccessError:
                continue
            found = _search(nested, files, wanted, depth + 1)
            if found is not None:
                return found
    return None


def _search_dir(directory: Path, files: FilesConfig, wanted: str, depth: int) -> Path | None:
    if depth > files.max_list_depth:
        return None
    try:
        children = list(os.scandir(directory))
    except OSError:
        return None
    for child in children:
        if child.is_symlink() or not child.is_dir(follow_symlinks=False):
            continue
        try:
            real = safe_path(child.path, files, kind="dir")
        except FileAccessError:
            continue
        if file_id(real) == wanted:
            return real
        found = _search_dir(real, files, wanted, depth + 1)
        if found is not None:
            return found
    return None


def allowlisted_roots(files: FilesConfig) -> list[Path]:
    """The configured roots, as real directories. Nested folders are not listed."""
    roots: list[Path] = []
    for folder in files.allowed_folders:
        if not folder.is_dir():
            continue
        roots.append(Path(os.path.realpath(folder)))
    return roots


def root_by_id(files: FilesConfig, wanted: str) -> Path | None:
    """Resolve a server-issued folder id. A client path never matches."""
    for root in allowlisted_roots(files):
        if file_id(root) == wanted:
            return root
    return None


def _parent_id(folder: Path, files: FilesConfig) -> str | None:
    for root in files.allowed_folders:
        root_real = Path(os.path.realpath(root))
        if folder == root_real:
            return None
        try:
            folder.relative_to(root_real)
        except ValueError:
            continue
        return file_id(folder.parent)
    return None


def _page(entries: list[dict], files: FilesConfig, *, enabled: bool, parent_id: str | None) -> dict:
    truncated = len(entries) > files.max_list_entries
    return {
        "enabled": enabled,
        "parent_id": parent_id,
        "truncated": truncated,
        "entries": entries[: files.max_list_entries],
    }
