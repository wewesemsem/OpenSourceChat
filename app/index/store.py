"""Read and write the folder index. Chat tables are not touched."""

from __future__ import annotations

import uuid

from app.db import connection, now


def start_folder(path: str) -> dict:
    """Open a folder row. Status stays partial until the scan finishes."""
    stamp = now()
    with connection() as conn:
        row = conn.execute(
            "SELECT id, path, status, created_at, updated_at FROM indexed_folders WHERE path = ?",
            (path,),
        ).fetchone()
        if row is not None:
            conn.execute(
                "UPDATE indexed_folders SET status = ?, updated_at = ? WHERE id = ?",
                ("partial", stamp, row["id"]),
            )
            folder_id = row["id"]
        else:
            folder_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO indexed_folders (id, path, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (folder_id, path, "partial", stamp, stamp),
            )
        stored = conn.execute(
            "SELECT id, path, status, created_at, updated_at FROM indexed_folders WHERE id = ?",
            (folder_id,),
        ).fetchone()
    return dict(stored)


def finish_folder(folder_id: str, status: str) -> None:
    with connection() as conn:
        conn.execute(
            "UPDATE indexed_folders SET status = ?, updated_at = ? WHERE id = ?",
            (status, now(), folder_id),
        )


def get_file(folder_id: str, path: str) -> dict | None:
    with connection() as conn:
        row = conn.execute(
            """
            SELECT id, folder_id, path, hash, size, mtime, status, error
            FROM indexed_files WHERE folder_id = ? AND path = ?
            """,
            (folder_id, path),
        ).fetchone()
    return dict(row) if row else None


def touch_file(file_id: str, size: int, mtime_ns: int) -> None:
    """Record a new mtime when the bytes did not change."""
    with connection() as conn:
        conn.execute(
            "UPDATE indexed_files SET size = ?, mtime = ? WHERE id = ?",
            (size, mtime_ns, file_id),
        )


def count_chunks(folder_id: str, exclude_file_id: str | None = None) -> int:
    with connection() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*)
            FROM chunks c
            JOIN indexed_files f ON f.id = c.file_id
            WHERE f.folder_id = ?
              AND (? IS NULL OR c.file_id != ?)
            """,
            (folder_id, exclude_file_id, exclude_file_id),
        ).fetchone()
    return int(row[0])


def folder_record(path: str) -> dict | None:
    """Counts for one indexed root. The caller already holds the real path."""
    with connection() as conn:
        row = conn.execute(
            "SELECT id, status, updated_at FROM indexed_folders WHERE path = ?",
            (path,),
        ).fetchone()
        if row is None:
            return None
        counts = conn.execute(
            """
            SELECT
                SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END),
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END)
            FROM indexed_files WHERE folder_id = ?
            """,
            (row["id"],),
        ).fetchone()
    return {
        "status": row["status"],
        "updated_at": row["updated_at"],
        "files": int(counts[0] or 0),
        "failed": int(counts[1] or 0),
    }


def chunk_bytes(folder_id: str) -> int:
    with connection() as conn:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(LENGTH(c.text)), 0)
            FROM chunks c
            JOIN indexed_files f ON f.id = c.file_id
            WHERE f.folder_id = ?
            """,
            (folder_id,),
        ).fetchone()
    return int(row[0])


def save_file(
    *,
    folder_id: str,
    path: str,
    digest: str,
    size: int,
    mtime_ns: int,
    status: str,
    error: str | None,
    chunks: list[tuple[str, int, str]],
) -> str:
    """Replace one file's chunks. Each chunk is text, token count, and metadata JSON."""
    with connection() as conn:
        row = conn.execute(
            "SELECT id FROM indexed_files WHERE folder_id = ? AND path = ?",
            (folder_id, path),
        ).fetchone()
        if row is None:
            file_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO indexed_files (
                    id, folder_id, path, hash, size, mtime, status, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (file_id, folder_id, path, digest, size, mtime_ns, status, error),
            )
        else:
            file_id = row["id"]
            _delete_chunks(conn, file_id)
            conn.execute(
                """
                UPDATE indexed_files
                SET hash = ?, size = ?, mtime = ?, status = ?, error = ?
                WHERE id = ?
                """,
                (digest, size, mtime_ns, status, error, file_id),
            )
        for ordinal, (text, token_count, metadata) in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO chunks (id, file_id, ordinal, text, token_count, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (chunk_id, file_id, ordinal, text, token_count, metadata),
            )
            conn.execute(
                "INSERT INTO chunks_fts (chunk_id, text) VALUES (?, ?)",
                (chunk_id, text),
            )
    return file_id


def delete_unseen(folder_id: str, seen: set[str]) -> int:
    with connection() as conn:
        rows = conn.execute(
            "SELECT id, path FROM indexed_files WHERE folder_id = ?",
            (folder_id,),
        ).fetchall()
        removed = 0
        for row in rows:
            if row["path"] in seen:
                continue
            _delete_file(conn, row["id"])
            removed += 1
    return removed


def _delete_file(conn, file_id: str) -> None:
    _delete_chunks(conn, file_id)
    conn.execute("DELETE FROM indexed_files WHERE id = ?", (file_id,))


def _delete_chunks(conn, file_id: str) -> None:
    ids = conn.execute("SELECT id FROM chunks WHERE file_id = ?", (file_id,)).fetchall()
    for row in ids:
        conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (row["id"],))
    conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
