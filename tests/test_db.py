"""Conversation storage and titles."""

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

    def test_title_from_text_is_one_short_line(self):
        self.assertEqual(db.title_from_text("  hello\nthere  "), "hello there")
        long = "word " * 30
        self.assertLessEqual(len(db.title_from_text(long)), 48)
        self.assertTrue(db.title_from_text(long).endswith("…"))
