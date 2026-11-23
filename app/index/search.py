"""Keyword search over indexed chunks.

The query is ordinary text. It is turned into an FTS5 OR query so a chunk
that contains more of the question ranks higher. SQLite's bm25() is more
negative for a better match; the score returned here is that magnitude, so
larger is better. The magnitude is not a 0–1 similarity.

Porter stemming is on, which matches "policy" to "policies" and also stems
some words aggressively. A fixed English stop list drops words that would
otherwise match almost every chunk. At most 32 terms are kept so a pasted
document cannot become a huge query.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass

from app.db import connection

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_STOP = frozenset(
    """
    a an the of to and or for in on at by from with as is are was were be been
    being this that these those it its their his her you your we they them i
    me my our not no nor but if then so than too very can will just about into
    over what which who how when where why
    """.split()
)
_MAX_TERMS = 32


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    name: str
    page: int | None
    heading: str | None
    position: int
    score: float
    ordinal: int
    text: str


def search_chunks(query: str, *, folder_id: str, top_k: int, min_score: float) -> list[Hit]:
    """Return the best keyword hits in one folder. Weak scores are dropped."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    match = fts_query(query)
    if not match:
        return []
    try:
        with connection() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.ordinal, c.text, c.metadata_json, bm25(chunks_fts) AS bm
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.chunk_id
                JOIN indexed_files f ON f.id = c.file_id
                WHERE chunks_fts MATCH ? AND f.folder_id = ? AND f.status = 'ok'
                ORDER BY bm ASC
                LIMIT ?
                """,
                (match, folder_id, top_k),
            ).fetchall()
    except sqlite3.OperationalError:
        return []
    hits: list[Hit] = []
    for row in rows:
        if row["bm"] is None:
            continue
        score = -float(row["bm"])
        if score < min_score:
            continue
        meta = _metadata(row["metadata_json"])
        if meta is None:
            continue
        hits.append(
            Hit(
                chunk_id=row["id"],
                name=meta["name"],
                page=meta["page"],
                heading=meta["heading"],
                position=meta["position"],
                score=score,
                ordinal=row["ordinal"],
                text=row["text"],
            )
        )
    return hits


def fts_query(text: str) -> str | None:
    """Build a quoted OR query. Returns None when nothing searchable remains."""
    terms: list[str] = []
    seen: set[str] = set()
    for raw in _TOKEN.findall(text.lower()):
        if raw in _STOP or len(raw) < 2 or raw in seen:
            continue
        seen.add(raw)
        terms.append('"' + raw.replace('"', '""') + '"')
        if len(terms) >= _MAX_TERMS:
            break
    if not terms:
        return None
    return " OR ".join(terms)


def _metadata(blob: str) -> dict | None:
    try:
        meta = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(meta, dict):
        return None
    name = meta.get("name")
    if not isinstance(name, str) or not name or name.startswith(("/", "\\")):
        return None
    page = meta.get("page")
    heading = meta.get("heading")
    position = meta.get("position")
    return {
        "name": name,
        "page": page if isinstance(page, int) else None,
        "heading": heading if isinstance(heading, str) else None,
        "position": position if isinstance(position, int) else 0,
    }
