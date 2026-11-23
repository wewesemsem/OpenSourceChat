"""Split extracted text into citation-sized chunks.

Sections stay apart: a chunk does not cross a heading or a page, so the
stored page or heading still describes the whole chunk. Overlap is copied
from the end of one chunk onto the next, and it stops at a word boundary.
A single word that is longer than the budget is kept whole.

CSV is packed as header plus data rows. The header is repeated on every
chunk. Rows are not split, so a wide row can exceed the budget. Keyword
search also finds tables less reliably than prose, because a row has little
language around the values.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from app.files.extract import Segment, SegmentedText
from app.memory.context import estimate_tokens


@dataclass(frozen=True)
class ChunkDraft:
    text: str
    token_count: int
    page: int | None
    heading: str | None
    position: int
    kind: str


def chunk_document(
    extracted: SegmentedText,
    *,
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[ChunkDraft]:
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
        raise ValueError("overlap must be smaller than the chunk size")
    if extracted.kind == "table":
        return _chunk_table(
            [part.text for part in extracted.parts],
            chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
        )
    return _chunk_prose(
        extracted.parts,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
    )


def metadata_json(name: str, draft: ChunkDraft) -> str:
    """Display fields only. The absolute path is not stored here."""
    if _absolute_name(name):
        raise ValueError("display name must be relative")
    payload = {
        "name": name,
        "page": draft.page,
        "heading": draft.heading,
        "position": draft.position,
        "kind": draft.kind,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _chunk_prose(
    segments: tuple[Segment, ...],
    *,
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[ChunkDraft]:
    drafts: list[ChunkDraft] = []
    bucket: list[_Piece] = []
    current: tuple[int | None, str | None] | None = None
    for piece in _pieces(segments, chunk_tokens):
        key = (piece.page, piece.heading)
        if current is None:
            current = key
        if key != current:
            drafts.extend(_pack(bucket, chunk_tokens, overlap_tokens))
            bucket = []
            current = key
        bucket.append(piece)
    drafts.extend(_pack(bucket, chunk_tokens, overlap_tokens))
    return drafts


def _chunk_table(
    rows: list[str],
    *,
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[ChunkDraft]:
    if not rows:
        return []
    header = rows[0]
    data = rows[1:]
    if not data:
        return [_draft(header, page=None, heading=None, position=0, kind="table")]
    drafts: list[ChunkDraft] = []
    index = 0
    while index < len(data):
        start = index
        chosen: list[str] = []
        while index < len(data):
            trial = "\n".join([header, *chosen, data[index]])
            if chosen and estimate_tokens(trial) > chunk_tokens:
                break
            chosen.append(data[index])
            index += 1
            if estimate_tokens("\n".join([header, *chosen])) >= chunk_tokens:
                break
        if not chosen:
            chosen = [data[start]]
            index = start + 1
        drafts.append(
            _draft("\n".join([header, *chosen]), page=None, heading=None, position=start, kind="table")
        )
        if index >= len(data):
            break
        if overlap_tokens > 0 and index > start + 1:
            back = index
            while back > start + 1:
                tail = "\n".join(data[back - 1 : index])
                if estimate_tokens(tail) > overlap_tokens:
                    break
                back -= 1
            index = back
    return drafts


@dataclass(frozen=True)
class _Piece:
    text: str
    page: int | None
    heading: str | None
    position: int
    split: bool = False


def _pieces(segments: tuple[Segment, ...], chunk_tokens: int) -> list[_Piece]:
    pieces: list[_Piece] = []
    offset = 0
    for segment in segments:
        paragraphs = _paragraphs(segment.text)
        if not paragraphs and segment.heading:
            pieces.append(_Piece("", segment.page, segment.heading, offset))
        for text, local in paragraphs:
            pieces.extend(_explode(text, segment, offset + local, chunk_tokens))
        offset += len(segment.text) + 2
    return pieces


def _paragraphs(text: str) -> list[tuple[str, int]]:
    items: list[tuple[str, int]] = []
    for match in re.finditer(r"[^\n]+(?:\n[^\n]+)*", text):
        block = match.group(0).strip()
        if block:
            items.append((block, match.start()))
    return items


def _explode(text: str, segment: Segment, position: int, chunk_tokens: int) -> list[_Piece]:
    """Keep a short paragraph whole. Split a long one into words so overlap can fit."""
    whole = _Piece(text, segment.page, segment.heading, position)
    if estimate_tokens(_render(segment.heading, "", [whole])) <= chunk_tokens:
        return [whole]
    words = text.split()
    if not words:
        return []
    return [
        _Piece(word, segment.page, segment.heading, position + _word_offset(text, index), True)
        for index, word in enumerate(words)
    ]


def _word_offset(text: str, word_index: int) -> int:
    count = 0
    for index, word in enumerate(text.split()):
        if index == word_index:
            return count
        count += len(word) + 1
    return count


def _pack(pieces: list[_Piece], chunk_tokens: int, overlap_tokens: int) -> list[ChunkDraft]:
    if not pieces:
        return []
    drafts: list[ChunkDraft] = []
    carry = ""
    index = 0
    while index < len(pieces):
        start = index
        chosen: list[_Piece] = []
        while index < len(pieces):
            trial = _render(pieces[0].heading, carry, chosen + [pieces[index]])
            over = estimate_tokens(trial) > chunk_tokens
            if not chosen and over and carry:
                carry = ""
                continue
            if chosen and over:
                break
            chosen.append(pieces[index])
            index += 1
            if estimate_tokens(_render(pieces[0].heading, carry, chosen)) >= chunk_tokens:
                break
        if not chosen:
            chosen = [pieces[start]]
            index = start + 1
            carry = ""
        heading = chosen[0].heading
        text = _render(heading, carry, chosen)
        drafts.append(
            _draft(
                text,
                page=chosen[0].page,
                heading=heading,
                position=chosen[0].position,
                kind="text",
            )
        )
        if index <= start:
            index = start + 1
        if index >= len(pieces):
            break
        carry = _tail_words(_body(chosen), overlap_tokens)
    return drafts


def _body(pieces: list[_Piece]) -> str:
    """Join split words with spaces and whole paragraphs with a blank line."""
    blocks: list[str] = []
    words: list[str] = []

    def flush() -> None:
        if words:
            blocks.append(" ".join(words))
            words.clear()

    for piece in pieces:
        if not piece.text:
            continue
        if piece.split:
            words.append(piece.text)
        else:
            flush()
            blocks.append(piece.text)
    flush()
    return "\n\n".join(blocks)


def _render(heading: str | None, carry: str, pieces: list[_Piece]) -> str:
    body = _body(pieces)
    if carry:
        if body and _continues(pieces) and "\n" not in carry:
            body = f"{carry} {body}"
        elif body:
            body = f"{carry}\n\n{body}"
        else:
            body = carry
    parts: list[str] = []
    if heading:
        parts.append(heading)
    if body:
        parts.append(body)
    return "\n\n".join(parts)


def _continues(pieces: list[_Piece]) -> bool:
    for piece in pieces:
        if piece.text:
            return piece.split
    return False


def _tail_words(text: str, overlap_tokens: int) -> str:
    """Copy the last words that fit in the overlap, leaving at least one word behind."""
    words = text.split()
    if overlap_tokens <= 0 or len(words) < 2:
        return ""
    chosen: list[str] = []
    for word in reversed(words):
        if len(chosen) >= len(words) - 1:
            break
        candidate = " ".join([word, *chosen])
        if chosen and estimate_tokens(candidate) > overlap_tokens:
            break
        chosen.insert(0, word)
    return " ".join(chosen)


def _draft(
    text: str,
    *,
    page: int | None,
    heading: str | None,
    position: int,
    kind: str,
) -> ChunkDraft:
    return ChunkDraft(
        text=text,
        token_count=estimate_tokens(text),
        page=page,
        heading=heading,
        position=position,
        kind=kind,
    )


def _absolute_name(name: str) -> bool:
    if not name or name.startswith(("/", "\\")) or ".." in Path(name).parts:
        return True
    return len(name) >= 2 and name[1] == ":"
