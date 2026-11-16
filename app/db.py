"""SQLite storage for conversations and messages."""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

_db_path: Path | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    system_prompt TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    model TEXT,
    tokens_in INTEGER,
    tokens_out INTEGER,
    latency_ms INTEGER,
    sources_json TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_by_conversation
    ON messages (conversation_id, created_at);
CREATE TABLE IF NOT EXISTS attachments (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL UNIQUE REFERENCES conversations(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    name TEXT NOT NULL,
    size INTEGER NOT NULL,
    hash TEXT NOT NULL,
    use_prefix INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
"""


def configure(path: Path) -> None:
    global _db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    _db_path = path
    with connection() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA)


def db_path() -> Path:
    if _db_path is None:
        raise RuntimeError("database is not configured")
    return _db_path


@contextmanager
def connection():
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def title_from_text(text: str, limit: int = 48) -> str:
    line = " ".join(text.strip().split())
    if len(line) <= limit:
        return line
    return line[: limit - 1].rstrip() + "…"


def create_conversation(title: str, system_prompt: str) -> dict:
    conversation_id = str(uuid.uuid4())
    stamp = now()
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO conversations (id, title, system_prompt, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, title, system_prompt, stamp, stamp),
        )
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise RuntimeError("conversation was not stored")
    return conversation


def list_conversations() -> list[dict]:
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT id, title, updated_at
            FROM conversations
            ORDER BY updated_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_conversation(conversation_id: str) -> dict | None:
    with connection() as conn:
        row = conn.execute(
            "SELECT id, title, system_prompt, created_at, updated_at FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
    return dict(row) if row else None


def rename_conversation(conversation_id: str, title: str) -> dict | None:
    cleaned = " ".join(title.strip().split())
    if not cleaned:
        return None
    stamp = now()
    with connection() as conn:
        updated = conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (cleaned, stamp, conversation_id),
        ).rowcount
    if not updated:
        return None
    return get_conversation(conversation_id)


def delete_conversation(conversation_id: str) -> bool:
    with connection() as conn:
        deleted = conn.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,),
        ).rowcount
    return bool(deleted)


def list_messages(conversation_id: str) -> list[dict]:
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT id, conversation_id, role, content, model, tokens_in, tokens_out,
                   latency_ms, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at, rowid
            """,
            (conversation_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def add_message(
    *,
    conversation_id: str,
    role: str,
    content: str,
    model: str | None,
    latency_ms: int | None = None,
) -> dict:
    message_id = str(uuid.uuid4())
    stamp = now()
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO messages (
                id, conversation_id, role, content, model, latency_ms, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (message_id, conversation_id, role, content, model, latency_ms, stamp),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (stamp, conversation_id),
        )
    messages = list_messages(conversation_id)
    return next(item for item in messages if item["id"] == message_id)


def set_message_model(message_id: str, model: str) -> None:
    with connection() as conn:
        conn.execute("UPDATE messages SET model = ? WHERE id = ?", (model, message_id))


def delete_last_assistant(conversation_id: str) -> None:
    with connection() as conn:
        row = conn.execute(
            """
            SELECT id, role FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at DESC, rowid DESC
            LIMIT 1
            """,
            (conversation_id,),
        ).fetchone()
        if row and row["role"] == "assistant":
            conn.execute("DELETE FROM messages WHERE id = ?", (row["id"],))


def public_attachment(row: dict) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "size": row["size"],
        "hash": row["hash"],
        "use_prefix": bool(row["use_prefix"]),
    }


def get_attachment(conversation_id: str) -> dict | None:
    with connection() as conn:
        row = conn.execute(
            """
            SELECT id, conversation_id, path, name, size, hash, use_prefix, created_at
            FROM attachments WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()
    return dict(row) if row else None


def set_attachment(
    *,
    conversation_id: str,
    path: str,
    name: str,
    size: int,
    digest: str,
    use_prefix: bool,
) -> dict:
    attachment_id = str(uuid.uuid4())
    stamp = now()
    with connection() as conn:
        conn.execute("DELETE FROM attachments WHERE conversation_id = ?", (conversation_id,))
        conn.execute(
            """
            INSERT INTO attachments (
                id, conversation_id, path, name, size, hash, use_prefix, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (attachment_id, conversation_id, path, name, size, digest, int(use_prefix), stamp),
        )
    stored = get_attachment(conversation_id)
    if stored is None:
        raise RuntimeError("attachment was not stored")
    return stored


def update_attachment_stat(attachment_id: str, size: int, digest: str) -> None:
    with connection() as conn:
        conn.execute(
            "UPDATE attachments SET size = ?, hash = ? WHERE id = ?",
            (size, digest, attachment_id),
        )


def delete_attachment(attachment_id: str) -> dict | None:
    with connection() as conn:
        row = conn.execute(
            """
            SELECT id, conversation_id, path, name, size, hash, use_prefix, created_at
            FROM attachments WHERE id = ?
            """,
            (attachment_id,),
        ).fetchone()
        if row is None:
            return None
        conn.execute("DELETE FROM attachments WHERE id = ?", (attachment_id,))
        return dict(row)


def delete_message(message_id: str) -> str | None:
    """Delete a message when it is the latest in its conversation."""
    with connection() as conn:
        row = conn.execute(
            "SELECT id, conversation_id FROM messages WHERE id = ?",
            (message_id,),
        ).fetchone()
        if row is None:
            return None
        last = conn.execute(
            """
            SELECT id FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at DESC, rowid DESC
            LIMIT 1
            """,
            (row["conversation_id"],),
        ).fetchone()
        if last["id"] != message_id:
            return None
        conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))
        return row["conversation_id"]
