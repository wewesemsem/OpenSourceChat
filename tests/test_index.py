"""Keyword index: allowlist, incremental updates, chunks, and FTS ranking."""

import json
import os
import sqlite3
import tempfile
import unittest
import zipfile
from pathlib import Path

import yaml

from app import db
from app.config import ROOT, load_config
from app.files.extract import Segment, read_text
from app.index.chunk import chunk_document
from app.index.evaluate import run_evaluation
from app.index.scan import index_folder
from app.index.search import search_chunks
from app.index.__main__ import load_strings, render_index
from app.memory.context import estimate_tokens


def _load(test: unittest.TestCase, folders: list[Path], **index):
    file_size = index.pop("max_file_size_mb", 20)
    values = {
        "enabled": True,
        "retrieval_mode": "keyword",
        "chunk_tokens": 80,
        "chunk_overlap_tokens": 8,
        "top_k": 4,
        "min_score": 0,
        "max_files": 50,
        "max_total_size_mb": 20,
        "max_chunks_per_folder": 100,
        "max_context_tokens_share": 0.25,
        "index_throttle": 0,
        "pause_indexing_during_chat": True,
        "followup_query_mode": "heuristic",
        "embedding_model": None,
    }
    values.update(index)
    raw = {
        "files": {
            "allowed_folders": [os.path.realpath(folder) for folder in folders],
            "allow_upload": False,
            "max_file_size_mb": file_size,
        },
        "index": values,
        "models": [{"name": "gemma3:4b", "base_url": "http://127.0.0.1:11434/v1"}],
    }
    directory = tempfile.TemporaryDirectory()
    test.addCleanup(directory.cleanup)
    path = Path(directory.name) / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(path)


class ChunkTests(unittest.TestCase):
    def test_headings_stay_in_separate_chunks(self):
        from app.files.extract import _markdown_segments

        text = "# Alpha\n\nalpha body is here\n\n# Beta\n\nbeta body is here\n"
        chunks = chunk_document(
            _segment(text, _markdown_segments(text)),
            chunk_tokens=200,
            overlap_tokens=20,
        )
        self.assertEqual([chunk.heading for chunk in chunks], ["Alpha", "Beta"])
        self.assertIn("alpha body", chunks[0].text)
        self.assertNotIn("beta body", chunks[0].text)
        self.assertIn("beta body", chunks[1].text)
        self.assertNotIn("alpha body", chunks[1].text)
        self.assertTrue(all(chunk.token_count <= 200 for chunk in chunks))

    def test_long_prose_splits_on_words_and_overlaps(self):
        words = [f"w{i}" for i in range(80)]
        text = " ".join(words)
        chunks = chunk_document(
            _segment(text, [Segment(text=text)]),
            chunk_tokens=30,
            overlap_tokens=8,
        )
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.token_count <= 30 for chunk in chunks))
        self.assertIn("w0 w1", chunks[0].text)
        self.assertNotIn("w0\n\nw1", chunks[0].text)
        self.assertIn("w79", chunks[-1].text)
        first_words = chunks[0].text.split()
        self.assertIn(first_words[-1], chunks[1].text.split())
        self.assertLess(chunks[0].position, chunks[-1].position)

    def test_csv_repeats_the_header(self):
        from app.index.chunk import _chunk_table

        header = "name\trole"
        rows = [header] + [f"person{i}\tclerk" for i in range(6)]
        budget = estimate_tokens(header + "\n" + rows[1])
        chunks = _chunk_table(rows, chunk_tokens=budget, overlap_tokens=1)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertTrue(chunk.text.startswith(header))
            self.assertEqual(chunk.kind, "table")
        self.assertLess(chunks[0].position, chunks[1].position)


def _segment(text: str, parts):
    from app.files.extract import SegmentedText

    return SegmentedText(name="notes.md", kind="text", parts=tuple(parts), error_code=None)


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "allowed"
        self.outside = Path(self.tmp.name) / "outside"
        self.root.mkdir()
        self.outside.mkdir()
        db.configure(Path(self.tmp.name) / "chat.sqlite")
        self.config = _load(self, [self.root])

    def test_migration_adds_index_tables_without_changing_messages(self):
        path = Path(self.tmp.name) / "legacy.sqlite"
        conn = sqlite3.connect(path)
        conn.execute(
            """
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT,
                role TEXT,
                content TEXT,
                model TEXT,
                tokens_in INTEGER,
                tokens_out INTEGER,
                latency_ms INTEGER,
                sources_json TEXT,
                created_at TEXT
            )
            """
        )
        conn.commit()
        conn.close()
        db.configure(path)
        with db.connection() as conn:
            names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
            columns = [row[1] for row in conn.execute("PRAGMA table_info(messages)")]
            folder_columns = [row[1] for row in conn.execute("PRAGMA table_info(indexed_folders)")]
            file_columns = [row[1] for row in conn.execute("PRAGMA table_info(indexed_files)")]
            chunk_columns = [row[1] for row in conn.execute("PRAGMA table_info(chunks)")]
        self.assertIn("indexed_folders", names)
        self.assertIn("indexed_files", names)
        self.assertIn("chunks", names)
        self.assertIn("chunks_fts", names)
        self.assertEqual(
            folder_columns,
            ["id", "path", "status", "created_at", "updated_at"],
        )
        self.assertEqual(
            file_columns,
            ["id", "folder_id", "path", "hash", "size", "mtime", "status", "error"],
        )
        self.assertEqual(
            chunk_columns,
            ["id", "file_id", "ordinal", "text", "token_count", "metadata_json"],
        )
        self.assertIn("content", columns)
        self.assertIn("ttft_ms", columns)

    def test_disabled_and_empty_allowlist_index_nothing(self):
        (self.root / "notes.txt").write_text("zephyr protocol", encoding="utf-8")
        disabled = _load(self, [self.root], enabled=False)
        result = index_folder(self.root / "missing", disabled)
        self.assertFalse(result.indexed)
        self.assertEqual(result.reason, "disabled")
        self.assertEqual(_chunk_count(), 0)

        empty = _load(self, [], enabled=True)
        result = index_folder(self.root, empty)
        self.assertEqual(result.reason, "no_folders")
        self.assertEqual(_chunk_count(), 0)

    def test_scanner_skips_escapes_and_excluded_names(self):
        (self.root / "notes.txt").write_text("visible zephyr protocol", encoding="utf-8")
        (self.root / ".env").write_text("TOKEN=SECRETVALUE", encoding="utf-8")
        (self.root / ".secret.txt").write_text("HIDDENSECRET", encoding="utf-8")
        nested = self.root / ".git"
        nested.mkdir()
        (nested / "notes.txt").write_text("GITSECRET", encoding="utf-8")
        secret = self.outside / "secret.txt"
        secret.write_text("OUTSIDESECRET", encoding="utf-8")
        (self.root / "leak.txt").symlink_to(secret)
        (self.root / "linked").symlink_to(self.outside, target_is_directory=True)
        (self.outside / "via-dir.txt").write_text("DIRSECRET", encoding="utf-8")
        target = self.root / "real.txt"
        target.write_text("inside alias target", encoding="utf-8")
        (self.root / "alias.txt").symlink_to(target)

        result = index_folder(self.root, self.config)
        self.assertTrue(result.indexed)
        blob = _all_text()
        self.assertIn("visible zephyr", blob)
        self.assertIn("inside alias", blob)
        for needle in ("SECRETVALUE", "HIDDENSECRET", "GITSECRET", "OUTSIDESECRET", "DIRSECRET"):
            self.assertNotIn(needle, blob)
        self.assertEqual(result.files, 2)
        paths = _file_paths()
        self.assertTrue(all(str(self.root) in path for path in paths))
        self.assertFalse(any(path.endswith(".env") or ".git" in path for path in paths))
        meta = _metadata_blob()
        self.assertNotIn(str(self.root), meta)
        self.assertNotIn(str(self.outside), meta)

    def test_refuses_a_folder_outside_the_allowlist(self):
        (self.outside / "notes.txt").write_text("outside zephyr", encoding="utf-8")
        result = index_folder(self.outside, self.config)
        self.assertFalse(result.indexed)
        self.assertEqual(result.reason, "not_allowed")
        self.assertEqual(_folder_count(), 0)
        self.assertEqual(_chunk_count(), 0)

    def test_unchanged_files_are_skipped_and_deleted_files_are_removed(self):
        path = self.root / "notes.txt"
        path.write_text("zephyr protocol stays", encoding="utf-8")
        first = index_folder(self.root, self.config)
        ids = _chunk_ids()
        self.assertEqual(first.skipped, 0)
        second = index_folder(self.root, self.config)
        self.assertEqual(second.skipped, 1)
        self.assertEqual(second.removed, 0)
        self.assertEqual(_chunk_ids(), ids)

        path.write_text("quartz timing replaced the old note", encoding="utf-8")
        third = index_folder(self.root, self.config)
        self.assertEqual(third.skipped, 0)
        self.assertEqual(search_chunks("zephyr", folder_id=third.folder_id, top_k=4, min_score=0), [])
        found = search_chunks("quartz", folder_id=third.folder_id, top_k=4, min_score=0)
        self.assertEqual(found[0].name, "notes.txt")
        self.assertNotEqual(_chunk_ids(), ids)

        path.unlink()
        fourth = index_folder(self.root, self.config)
        self.assertEqual(fourth.removed, 1)
        self.assertEqual(fourth.files, 0)
        self.assertEqual(_chunk_count(), 0)
        self.assertEqual(search_chunks("quartz", folder_id=fourth.folder_id, top_k=4, min_score=0), [])

    def test_a_bad_file_does_not_stop_the_scan(self):
        from pypdf import PdfWriter

        pdf = self.root / "bad.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.add_blank_page(width=72, height=72)
        writer.add_blank_page(width=72, height=72)
        with pdf.open("wb") as handle:
            writer.write(handle)
        (self.root / "good.txt").write_text("zephyr survived the bad pdf", encoding="utf-8")
        limited = _load(self, [self.root])
        from dataclasses import replace

        limited = replace(limited, files=replace(limited.files, max_pages=2))
        result = index_folder(self.root, limited)
        self.assertGreaterEqual(result.failed, 1)
        hits = search_chunks("zephyr", folder_id=result.folder_id, top_k=4, min_score=0)
        self.assertEqual(hits[0].name, "good.txt")

    def test_limits_refuse_the_folder_and_the_chunk_cap_stops(self):
        for name in ("a.txt", "b.txt", "c.txt"):
            (self.root / name).write_text(name, encoding="utf-8")
        capped = _load(self, [self.root], max_files=2)
        refused = index_folder(self.root, capped)
        self.assertEqual(refused.reason, "too_many_files")
        self.assertEqual(_folder_count(), 0)

        (self.root / "big.txt").write_bytes(b"x" * (2 * 1024 * 1024))
        for name in ("a.txt", "b.txt", "c.txt"):
            (self.root / name).unlink()
        huge = _load(self, [self.root], max_total_size_mb=1)
        refused = index_folder(self.root, huge)
        self.assertEqual(refused.reason, "too_large")
        self.assertEqual(_folder_count(), 0)
        (self.root / "big.txt").unlink()

        (self.root / "a.txt").write_text("alpha zephyr", encoding="utf-8")
        (self.root / "b.txt").write_text("beta quartz", encoding="utf-8")
        narrow = _load(self, [self.root], max_chunks_per_folder=1)
        partial = index_folder(self.root, narrow)
        self.assertTrue(partial.indexed)
        self.assertEqual(partial.reason, "chunk_limit")
        self.assertEqual(partial.chunks, 1)
        hits = search_chunks("zephyr", folder_id=partial.folder_id, top_k=4, min_score=0)
        self.assertEqual([hit.name for hit in hits], ["a.txt"])
        self.assertEqual(search_chunks("quartz", folder_id=partial.folder_id, top_k=4, min_score=0), [])

    def test_fts_ranks_the_obvious_file_and_drops_weak_scores(self):
        (self.root / "alpha.txt").write_text(
            "The zephyr protocol uses quartz timing for the radio link.",
            encoding="utf-8",
        )
        (self.root / "beta.txt").write_text(
            "Lunch is served at noon in the hall.",
            encoding="utf-8",
        )
        guides = self.root / "guides"
        guides.mkdir()
        (guides / "install.md").write_text("# Install\n\nUse the widget tool.\n", encoding="utf-8")
        result = index_folder(self.root, self.config)
        hits = search_chunks(
            "what is the zephyr protocol",
            folder_id=result.folder_id,
            top_k=4,
            min_score=0,
        )
        self.assertEqual(hits[0].name, "alpha.txt")
        self.assertGreater(hits[0].score, 0)
        self.assertNotIn(str(self.root), hits[0].text)
        headed = search_chunks("widget", folder_id=result.folder_id, top_k=4, min_score=0)
        self.assertEqual(headed[0].name, "guides/install.md")
        self.assertEqual(headed[0].heading, "Install")
        self.assertEqual(
            search_chunks("zephyr protocol", folder_id=result.folder_id, top_k=4, min_score=10_000),
            [],
        )
        other = Path(self.tmp.name) / "second"
        other.mkdir()
        (other / "other.txt").write_text("zephyr lives in the other folder", encoding="utf-8")
        both = _load(self, [self.root, other])
        other_index = index_folder(other, both)
        scoped = search_chunks("other folder", folder_id=result.folder_id, top_k=4, min_score=0)
        self.assertEqual(scoped, [])
        found = search_chunks("other folder", folder_id=other_index.folder_id, top_k=4, min_score=0)
        self.assertEqual(found[0].name, "other.txt")

    def test_messages_come_from_the_strings_file(self):
        texts = load_strings()
        result = index_folder(self.outside, self.config)
        lines = render_index(result, texts)
        self.assertEqual(lines, [texts["index_not_allowed"]])
        (self.root / "rows.csv").write_text("name,role\nada,clerk\n", encoding="utf-8")
        indexed = index_folder(self.root, self.config)
        summary = render_index(indexed, texts)
        self.assertIn(texts["index_tables_note"], summary)
        self.assertTrue(summary[0].startswith(self.root.name + ":"))
        self.assertNotIn(str(self.root), "\n".join(summary))


class EvalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "docs"
        self.root.mkdir()
        db.configure(Path(self.tmp.name) / "chat.sqlite")
        self.config = _load(self, [self.root])

    def test_reports_hit_rate_without_chunk_text_or_the_absolute_path(self):
        (self.root / "alpha.txt").write_text(
            "The zephyr protocol uses quartz timing for the radio link.",
            encoding="utf-8",
        )
        (self.root / "beta.txt").write_text("Lunch is served at noon.", encoding="utf-8")
        questions = {
            "folder": os.path.realpath(self.root),
            "output": "data/index-eval-test.json",
            "k": 4,
            "questions": [
                {
                    "id": "protocol",
                    "question": "Where is the zephyr protocol described?",
                    "expect": ["alpha.txt"],
                },
                {
                    "id": "missing",
                    "question": "Where are the spare badges stored?",
                    "expect": ["missing.md"],
                },
            ],
        }
        path = Path(self.tmp.name) / "questions.json"
        path.write_text(json.dumps(questions), encoding="utf-8")
        output = ROOT / "data" / "index-eval-test.json"
        if output.exists():
            output.unlink()
        self.addCleanup(lambda: output.unlink(missing_ok=True))
        result = run_evaluation(path, self.config)
        self.assertTrue(result.ok)
        self.assertEqual(result.report["hits"], 1)
        self.assertEqual(result.report["hit_rate"], 0.5)
        self.assertEqual(result.report["results"][0]["matches"], ["alpha.txt"])
        self.assertGreaterEqual(result.report["results"][0]["retrieval_ms"], 0)
        saved = output.read_text(encoding="utf-8")
        self.assertNotIn("radio link", saved)
        self.assertNotIn(os.path.realpath(self.root), saved)
        self.assertNotIn(str(self.root), saved)

    def test_sample_questions_are_placeholders(self):
        raw = json.loads((ROOT / "eval" / "questions.sample.json").read_text(encoding="utf-8"))
        self.assertIn("REPLACE", raw["folder"])
        self.assertEqual(len(raw["questions"]), 2)
        result = run_evaluation(
            ROOT / "eval" / "questions.sample.json",
            _load(self, [self.root], enabled=False),
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "disabled")

    def test_attach_still_returns_markdown_source(self):
        path = self.root / "note.md"
        path.write_text("# Alpha\n\nbody\n", encoding="utf-8")
        self.assertEqual(read_text(path, self.config.files).text, "# Alpha\n\nbody\n")


def _chunk_count() -> int:
    with db.connection() as conn:
        return int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])


def _chunk_ids() -> set[str]:
    with db.connection() as conn:
        return {row[0] for row in conn.execute("SELECT id FROM chunks")}


def _all_text() -> str:
    with db.connection() as conn:
        rows = conn.execute("SELECT text FROM chunks").fetchall()
    return "\n".join(row[0] for row in rows)


def _metadata_blob() -> str:
    with db.connection() as conn:
        rows = conn.execute("SELECT metadata_json FROM chunks").fetchall()
    return "\n".join(row[0] for row in rows)


def _file_paths() -> list[str]:
    with db.connection() as conn:
        return [row[0] for row in conn.execute("SELECT path FROM indexed_files")]


def _folder_count() -> int:
    with db.connection() as conn:
        return int(conn.execute("SELECT COUNT(*) FROM indexed_folders").fetchone()[0])


if __name__ == "__main__":
    unittest.main()
