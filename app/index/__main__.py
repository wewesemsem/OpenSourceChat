"""Index a folder, or score keyword search against a question file.

    python -m app.index index /absolute/allowlisted/folder
    python -m app.index eval eval/questions.sample.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app import db
from app.config import load_config
from app.index.evaluate import EvalFileError, EvalResult, run_evaluation
from app.index.scan import IndexResult, index_folder

STRINGS_PATH = Path(__file__).resolve().parents[1] / "strings.json"
STRING_KEYS = (
    "index_cli",
    "index_disabled",
    "index_no_folders",
    "index_not_allowed",
    "index_not_found",
    "index_excluded",
    "index_not_a_folder",
    "index_refused_files",
    "index_refused_size",
    "index_chunk_limit",
    "index_failed_run",
    "index_summary",
    "index_tables_note",
    "index_fts_missing",
    "index_keyword_only",
    "index_eval_summary",
    "index_eval_query",
    "index_eval_hit",
    "index_eval_miss",
    "index_eval_bad",
    "index_eval_folder",
    "index_eval_empty",
    "index_eval_question",
    "index_eval_output",
)

_REASONS = {
    "disabled": "index_disabled",
    "no_folders": "index_no_folders",
    "not_allowed": "index_not_allowed",
    "rejected": "index_not_allowed",
    "not_found": "index_not_found",
    "excluded": "index_excluded",
    "not_a_file": "index_not_a_folder",
    "too_many_files": "index_refused_files",
    "too_large": "index_refused_size",
    "fts_missing": "index_fts_missing",
    "chunk_limit": "index_chunk_limit",
    "failed": "index_failed_run",
    "keyword_only": "index_keyword_only",
    "bad": "index_eval_bad",
    "folder": "index_eval_folder",
    "empty": "index_eval_empty",
    "question": "index_eval_question",
    "output": "index_eval_output",
}


def load_strings() -> dict[str, str]:
    loaded = json.loads(STRINGS_PATH.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or any(not isinstance(value, str) for value in loaded.values()):
        raise ValueError("strings.json must be an object of strings")
    missing = [key for key in STRING_KEYS if key not in loaded]
    if missing:
        raise ValueError("strings.json missing " + ", ".join(missing))
    return loaded


def main(argv: list[str] | None = None) -> int:
    texts = load_strings()
    parser = argparse.ArgumentParser(prog="python -m app.index", description=texts["index_cli"])
    commands = parser.add_subparsers(dest="command", required=True)
    index_command = commands.add_parser("index")
    index_command.add_argument("folder")
    eval_command = commands.add_parser("eval")
    eval_command.add_argument("questions")
    args = parser.parse_args(argv)
    config = load_config()
    if args.command == "index":
        if config.index.enabled:
            db.configure(config.data_dir / db.DATABASE_NAME)
        folder = Path(args.folder).expanduser()
        if not folder.is_absolute():
            folder = (Path.cwd() / folder).resolve()
        result = index_folder(folder, config)
        for line in render_index(result, texts):
            print(line)
        return index_exit(result)
    if config.index.enabled:
        db.configure(config.data_dir / db.DATABASE_NAME)
    try:
        outcome = run_evaluation(Path(args.questions), config)
    except EvalFileError as exc:
        print(_line(exc.code, IndexResult(False, exc.code, "", config.index.retrieval_mode), texts))
        return 1
    for line in render_eval(outcome, texts):
        print(line)
    if outcome.reason == "fts_missing":
        return 2
    return 0 if outcome.ok else 1


def render_index(result: IndexResult, texts: dict[str, str]) -> list[str]:
    if not result.indexed:
        return [_line(result.reason or "disabled", result, texts)]
    lines = [
        _fill(
            texts["index_summary"],
            name=result.folder_name,
            files=result.files,
            chunks=result.chunks,
            skipped=result.skipped,
            failed=result.failed,
            removed=result.removed,
            seconds=f"{result.seconds:.1f}",
            size=result.chunk_bytes,
        )
    ]
    if result.reason:
        lines.append(_line(result.reason, result, texts))
    if result.table_files:
        lines.append(texts["index_tables_note"])
    if result.retrieval_mode != "keyword":
        lines.append(texts["index_keyword_only"])
    return lines


def render_eval(result: EvalResult, texts: dict[str, str]) -> list[str]:
    if not result.ok or result.report is None:
        reason = result.reason or "bad"
        index = result.index or IndexResult(False, reason, "", "keyword")
        return [_line(reason, index, texts)]
    report = result.report
    lines = [
        _fill(
            texts["index_eval_summary"],
            k=report["k"],
            hits=report["hits"],
            questions=report["questions"],
            rate=report["hit_rate"],
            ms=report["mean_retrieval_ms"],
            output=result.output,
        )
    ]
    for row in report["results"]:
        label = texts["index_eval_hit"] if row["hit"] else texts["index_eval_miss"]
        matches = ", ".join(row["matches"]) if row["matches"] else label
        lines.append(
            _fill(
                texts["index_eval_query"],
                id=row["id"],
                hit=label,
                ms=row["retrieval_ms"],
                matches=matches,
            )
        )
    if report.get("table_files"):
        lines.append(texts["index_tables_note"])
    return lines


def index_exit(result: IndexResult) -> int:
    if result.reason == "fts_missing":
        return 2
    if result.indexed and result.reason is None:
        return 0
    if result.reason == "disabled":
        return 0
    if result.indexed:
        return 1
    return 1


def _line(reason: str, result: IndexResult, texts: dict[str, str]) -> str:
    key = _REASONS.get(reason, "index_eval_bad")
    if reason == "too_many_files":
        return _fill(texts[key], count=result.count, limit=result.limit)
    if reason == "too_large":
        return _fill(texts[key], size=f"{result.size_mb:.1f}", limit=result.limit)
    if reason == "chunk_limit":
        return _fill(texts[key], chunks=result.chunks, limit=result.limit)
    return texts[key]


def _fill(template: str, **values: object) -> str:
    text = template
    for key, value in values.items():
        text = text.replace("{" + key + "}", str(value))
    return text


if __name__ == "__main__":
    sys.exit(main())
