"""Index one allowlisted folder.

Unchanged files are skipped from size and mtime without reading them. A
changed mtime with the same hash updates the mtime and does not re-chunk.
That trusts the size and mtime for the fast path. A file whose bytes change
without either of those changing is missed until the next real change.

Directory symlinks are not followed. A link that leaves the allowlist is
never opened. A directory that can only be reached through an in-folder
symlink is not indexed; the real directory is indexed when it is inside the
folder being scanned. File symlinks are accepted only when safe_path() says
the target stays inside this folder, and the real file is stored once.

The list depth used by the folder browser does not apply here. File count,
total size, and chunk count do.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

from app.config import AppConfig, FilesConfig
from app.db import fts5_available
from app.files.extract import read_segments
from app.files.safe import FileAccessError, indexable_directory, safe_path
from app.index.chunk import chunk_document, metadata_json
from app.index.store import (
    chunk_bytes,
    count_chunks,
    delete_unseen,
    finish_folder,
    get_file,
    save_file,
    start_folder,
    touch_file,
)

# Errors that will not change until the file's size or mtime changes.
_SETTLED_ERRORS = frozenset({"too_large", "too_many_pages"})


@dataclass(frozen=True)
class IndexResult:
    indexed: bool
    reason: str | None
    folder_name: str
    retrieval_mode: str
    folder_id: str | None = None
    files: int = 0
    chunks: int = 0
    skipped: int = 0
    failed: int = 0
    removed: int = 0
    seconds: float = 0.0
    chunk_bytes: int = 0
    table_files: int = 0
    count: int = 0
    limit: int = 0
    size_mb: float = 0.0


@dataclass(frozen=True)
class _Candidate:
    path: Path
    size: int
    mtime_ns: int
    error: str | None


def index_folder(raw: str | Path, config: AppConfig) -> IndexResult:
    """Index one folder, or refuse without writing when the feature or the limits say so."""
    name = _folder_name(raw)
    mode = config.index.retrieval_mode
    if not config.index.enabled:
        return _refused(name, "disabled", mode)
    if not config.files.allowed_folders:
        return _refused(name, "no_folders", mode)
    if not fts5_available():
        return _refused(name, "fts_missing", mode)
    try:
        root = indexable_directory(raw, config.files)
    except FileAccessError as exc:
        reason = "not_allowed" if exc.code == "rejected" else exc.code
        return _refused(name, reason, mode)

    started = time.perf_counter()
    candidates = _collect(root, config.files)
    total_bytes = sum(item.size for item in candidates)
    if len(candidates) > config.index.max_files:
        return _refused(
            root.name,
            "too_many_files",
            mode,
            seconds=time.perf_counter() - started,
            count=len(candidates),
            limit=config.index.max_files,
        )
    if total_bytes > config.index.max_total_size_bytes:
        return _refused(
            root.name,
            "too_large",
            mode,
            seconds=time.perf_counter() - started,
            size_mb=total_bytes / (1024 * 1024),
            limit=config.index.max_total_size_bytes // (1024 * 1024),
        )

    folder = start_folder(str(root))
    seen: set[str] = set()
    ok_files = 0
    skipped = 0
    failed = 0
    table_files = 0
    stopped = False
    for candidate in candidates:
        outcome = _index_candidate(root, folder["id"], candidate, config, seen)
        if outcome == "limit":
            stopped = True
            break
        if outcome == "skipped":
            skipped += 1
        elif outcome == "ok":
            ok_files += 1
        elif outcome == "failed":
            failed += 1
        if outcome in {"ok", "skipped"} and candidate.path.suffix.lower() == ".csv":
            table_files += 1
        if outcome in {"ok", "failed"} and config.index.index_throttle > 0:
            time.sleep(config.index.index_throttle)

    removed = 0
    if not stopped:
        removed = delete_unseen(folder["id"], seen)
    if stopped or failed:
        status = "partial"
    elif ok_files == 0:
        status = "empty"
    else:
        status = "ready"
    finish_folder(folder["id"], status)
    return IndexResult(
        indexed=True,
        reason="chunk_limit" if stopped else None,
        folder_name=root.name,
        retrieval_mode=mode,
        folder_id=folder["id"],
        files=ok_files + skipped,
        chunks=count_chunks(folder["id"]),
        skipped=skipped,
        failed=failed,
        removed=removed,
        seconds=time.perf_counter() - started,
        chunk_bytes=chunk_bytes(folder["id"]),
        table_files=table_files,
        limit=config.index.max_chunks_per_folder,
    )


def _index_candidate(
    root: Path,
    folder_id: str,
    candidate: _Candidate,
    config: AppConfig,
    seen: set[str],
) -> str:
    path = str(candidate.path)
    existing = get_file(folder_id, path)
    if candidate.error == "too_large":
        seen.add(path)
        if _settled(existing, candidate):
            return "settled"
        save_file(
            folder_id=folder_id,
            path=path,
            digest="",
            size=candidate.size,
            mtime_ns=candidate.mtime_ns,
            status="error",
            error="too_large",
            chunks=[],
        )
        return "failed"
    try:
        real = safe_path(candidate.path, config.files)
    except FileAccessError as exc:
        if exc.code != "too_large":
            return "gone"
        seen.add(path)
        save_file(
            folder_id=folder_id,
            path=path,
            digest="",
            size=candidate.size,
            mtime_ns=candidate.mtime_ns,
            status="error",
            error="too_large",
            chunks=[],
        )
        return "failed"
    if not _inside(root, real):
        return "gone"
    info = real.stat()
    seen.add(str(real))
    current = get_file(folder_id, str(real))
    if _unchanged(current, info):
        return "skipped"
    digest = _hash_file(real)
    if current and current["status"] == "ok" and current["hash"] == digest:
        touch_file(current["id"], info.st_size, info.st_mtime_ns)
        return "skipped"
    extracted = read_segments(real, config.files)
    if not extracted.ok:
        save_file(
            folder_id=folder_id,
            path=str(real),
            digest=digest,
            size=info.st_size,
            mtime_ns=info.st_mtime_ns,
            status="error",
            error=extracted.error_code,
            chunks=[],
        )
        return "failed"
    drafts = chunk_document(
        extracted,
        chunk_tokens=config.index.chunk_tokens,
        overlap_tokens=config.index.chunk_overlap_tokens,
    )
    if not drafts:
        save_file(
            folder_id=folder_id,
            path=str(real),
            digest=digest,
            size=info.st_size,
            mtime_ns=info.st_mtime_ns,
            status="error",
            error="empty",
            chunks=[],
        )
        return "failed"
    exclude = current["id"] if current else None
    kept = count_chunks(folder_id, exclude_file_id=exclude)
    if kept + len(drafts) > config.index.max_chunks_per_folder:
        return "limit"
    name = _display_name(root, real)
    save_file(
        folder_id=folder_id,
        path=str(real),
        digest=digest,
        size=info.st_size,
        mtime_ns=info.st_mtime_ns,
        status="ok",
        error=None,
        chunks=[(draft.text, draft.token_count, metadata_json(name, draft)) for draft in drafts],
    )
    return "ok"


def _collect(root: Path, files: FilesConfig) -> list[_Candidate]:
    found: dict[str, _Candidate] = {}
    stack = [root]
    seen_dirs = {root}
    while stack:
        directory = stack.pop()
        try:
            children = list(os.scandir(directory))
        except OSError:
            continue
        for child in children:
            child_path = Path(child.path)
            if child.is_symlink():
                if child.is_dir(follow_symlinks=True):
                    continue
                _consider_file(root, child_path, files, found)
                continue
            if child.is_dir(follow_symlinks=False):
                try:
                    real = safe_path(child_path, files, kind="dir")
                except FileAccessError:
                    continue
                if real in seen_dirs or not _inside(root, real):
                    continue
                seen_dirs.add(real)
                stack.append(real)
                continue
            if child.is_file(follow_symlinks=False):
                _consider_file(root, child_path, files, found)
    return sorted(found.values(), key=lambda item: str(item.path))


def _consider_file(root: Path, path: Path, files: FilesConfig, found: dict[str, _Candidate]) -> None:
    try:
        real = safe_path(path, files)
    except FileAccessError as exc:
        if exc.code == "too_large":
            _remember_too_large(root, path, found)
        return
    if not _inside(root, real):
        return
    _remember(real, found, error=None)


def _remember_too_large(root: Path, path: Path, found: dict[str, _Candidate]) -> None:
    try:
        real = Path(os.path.realpath(path))
    except OSError:
        return
    if not _inside(root, real):
        return
    _remember(real, found, error="too_large")


def _remember(real: Path, found: dict[str, _Candidate], *, error: str | None) -> None:
    key = str(real)
    if key in found:
        return
    try:
        info = real.stat()
    except OSError:
        return
    found[key] = _Candidate(real, info.st_size, info.st_mtime_ns, error)


def _unchanged(existing: dict | None, info: os.stat_result) -> bool:
    if existing is None:
        return False
    if existing["size"] != info.st_size or existing["mtime"] != info.st_mtime_ns:
        return False
    if existing["status"] == "ok":
        return True
    return existing["error"] in _SETTLED_ERRORS


def _settled(existing: dict | None, candidate: _Candidate) -> bool:
    return (
        existing is not None
        and existing["error"] == candidate.error
        and existing["size"] == candidate.size
        and existing["mtime"] == candidate.mtime_ns
    )


def _inside(root: Path, real: Path) -> bool:
    try:
        relative = real.relative_to(root)
    except ValueError:
        return False
    return ".." not in relative.parts


def _display_name(root: Path, path: Path) -> str:
    name = path.relative_to(root).as_posix()
    if not name or name.startswith("/") or ".." in Path(name).parts:
        raise ValueError("display name must be relative")
    return name


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _folder_name(raw: str | Path) -> str:
    return Path(raw).name or "folder"


def _refused(
    name: str,
    reason: str,
    mode: str,
    *,
    seconds: float = 0.0,
    count: int = 0,
    limit: int = 0,
    size_mb: float = 0.0,
) -> IndexResult:
    return IndexResult(
        indexed=False,
        reason=reason,
        folder_name=name,
        retrieval_mode=mode,
        seconds=seconds,
        count=count,
        limit=limit,
        size_mb=size_mb,
    )
