"""Conversation storage and titles."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from app import db


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        db.configure(Path(self.tmp.name) / "chat.sqlite")

    def test_delete_removes_messages(self):
        conversation = db.create_conversation("New chat", "Be brief.")
        db.add_message(conversation_id=conversation["id"], role="user", content="Hi", model="gemma3:12b")
        db.delete_conversation(conversation["id"])
        self.assertEqual(db.list_messages(conversation["id"]), [])
        self.assertIsNone(db.get_conversation(conversation["id"]))

    def test_delete_message_only_removes_the_latest(self):
        conversation = db.create_conversation("New chat", "Be brief.")
        first = db.add_message(
            conversation_id=conversation["id"], role="user", content="Hi", model="gemma3:12b"
        )
        second = db.add_message(
            conversation_id=conversation["id"], role="assistant", content="Hello", model="gemma3:12b"
        )
        self.assertIsNone(db.delete_message(first["id"]))
        self.assertEqual(db.delete_message(second["id"]), conversation["id"])
        self.assertEqual(len(db.list_messages(conversation["id"])), 1)

    def test_stores_time_to_first_token_and_speed(self):
        conversation = db.create_conversation("New chat", "Be brief.")
        row = db.add_message(
            conversation_id=conversation["id"],
            role="assistant",
            content="Hello",
            model="gemma3:4b",
            latency_ms=1200,
            ttft_ms=400,
            tokens_per_second=12.5,
            tokens_in=30,
            tokens_out=8,
        )
        self.assertEqual(row["ttft_ms"], 400)
        self.assertEqual(row["tokens_per_second"], 12.5)
        self.assertEqual(row["tokens_in"], 30)
        self.assertEqual(row["tokens_out"], 8)

    def test_older_database_gains_timing_columns(self):
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
        conversation = db.create_conversation("New chat", "Be brief.")
        row = db.add_message(
            conversation_id=conversation["id"],
            role="assistant",
            content="Hello",
            model="gemma3:4b",
            ttft_ms=15,
            tokens_per_second=3.5,
        )
        self.assertEqual(row["ttft_ms"], 15)
        self.assertEqual(row["tokens_per_second"], 3.5)

    def test_title_from_text_is_one_short_line(self):
        self.assertEqual(db.title_from_text("  hello\nthere  "), "hello there")
        long = "word " * 30
        self.assertLessEqual(len(db.title_from_text(long)), 48)
        self.assertTrue(db.title_from_text(long).endswith("…"))
