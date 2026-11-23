"""Score keyword search against a question file.

A hit means one expected display name appears in the top results. The
report stores names, scores are not required, and it does not store chunk
text or the folder's absolute path.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from app.config import ROOT, AppConfig
from app.index.scan import IndexResult, index_folder
from app.index.search import search_chunks


class EvalFileError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    expect: tuple[str, ...]


@dataclass(frozen=True)
class EvalResult:
    ok: bool
    reason: str | None
    output: str
    index: IndexResult | None = None
    report: dict | None = None


def run_evaluation(path: Path, config: AppConfig) -> EvalResult:
    questions, folder, output, shown, k = _load(path, config.index.top_k)
    if not config.index.enabled:
        return EvalResult(ok=False, reason="disabled", output=shown)
    if not config.files.allowed_folders:
        return EvalResult(ok=False, reason="no_folders", output=shown)
    if config.index.retrieval_mode != "keyword":
        return EvalResult(ok=False, reason="keyword_only", output=shown)
    indexed = index_folder(folder, config)
    if not indexed.indexed or indexed.folder_id is None:
        return EvalResult(ok=False, reason=indexed.reason, output=shown, index=indexed)

    rows = []
    hits = 0
    elapsed = 0
    for question in questions:
        started = time.perf_counter()
        found = search_chunks(
            question.text,
            folder_id=indexed.folder_id,
            top_k=k,
            min_score=config.index.min_score,
        )
        retrieval_ms = int((time.perf_counter() - started) * 1000)
        elapsed += retrieval_ms
        names = list(dict.fromkeys(hit.name for hit in found))
        hit = _is_hit(names, question.expect)
        hits += int(hit)
        rows.append(
            {
                "id": question.id,
                "hit": hit,
                "retrieval_ms": retrieval_ms,
                "matches": names,
            }
        )
    report = {
        "retrieval_mode": "keyword",
        "k": k,
        "questions": len(questions),
        "hits": hits,
        "hit_rate": round(hits / len(questions), 4),
        "mean_retrieval_ms": int(elapsed / len(questions)),
        "index_seconds": round(indexed.seconds, 3),
        "table_files": indexed.table_files,
        "results": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return EvalResult(ok=True, reason=None, output=shown, index=indexed, report=report)


def _load(path: Path, default_k: int) -> tuple[tuple[Question, ...], str, Path, str, int]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise EvalFileError("bad") from None
    if not isinstance(raw, dict):
        raise EvalFileError("bad")
    folder = raw.get("folder")
    if not isinstance(folder, str) or not Path(folder).is_absolute():
        raise EvalFileError("folder")
    shown, output = _output_path(raw.get("output"))
    k = raw.get("k", default_k)
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise EvalFileError("bad")
    items = raw.get("questions")
    if not isinstance(items, list) or not items:
        raise EvalFileError("empty")
    questions = tuple(_question(item) for item in items)
    ids = [item.id for item in questions]
    if len(ids) != len(set(ids)):
        raise EvalFileError("question")
    return questions, folder, output, shown, k


def _question(item: object) -> Question:
    if not isinstance(item, dict):
        raise EvalFileError("question")
    ident = item.get("id")
    text = item.get("question")
    expect = item.get("expect")
    if not isinstance(ident, str) or not ident.strip():
        raise EvalFileError("question")
    if not isinstance(text, str) or not text.strip():
        raise EvalFileError("question")
    if not isinstance(expect, list) or not expect:
        raise EvalFileError("question")
    names = []
    for name in expect:
        if not isinstance(name, str) or not name.strip():
            raise EvalFileError("question")
        names.append(name.strip())
    return Question(ident.strip(), text.strip(), tuple(names))


def _output_path(value: object) -> tuple[str, Path]:
    if not isinstance(value, str) or not value.strip():
        raise EvalFileError("output")
    shown = value.strip()
    candidate = Path(shown)
    if candidate.is_absolute():
        raise EvalFileError("output")
    resolved = (ROOT / candidate).resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise EvalFileError("output")
    return shown, resolved


def _is_hit(names: list[str], expected: tuple[str, ...]) -> bool:
    for name in names:
        base = name.rsplit("/", 1)[-1]
        for item in expected:
            if item == name or item == base:
                return True
    return False
