"""One file can be attached. The client sends an id, not a path."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from app import db
from app.config import FilesConfig
from app.main import config, strings
from tests.test_chat import open_client


def _files(folder: Path) -> FilesConfig:
    return FilesConfig(
        allowed_folders=(folder.resolve(),),
        allowed_extensions=("txt",),
        max_file_size_bytes=1024 * 1024,
        max_pages=5,
        exclude_patterns=(".*", ".env", ".git", "*.key"),
        extract_timeout_seconds=2,
        max_extract_chars=100_000,
        context_share=0.5,
        context_instruction="Reference only.",
        context_begin="----- BEGIN FILE {name} -----",
        context_end="----- END FILE {name} -----",
        truncated_note="Only the beginning of the file is included.",
        max_list_entries=50,
        max_list_depth=4,
    )


def _app_config(files: FilesConfig):
    model = replace(config.model(), context_size=400, reserved_output_tokens=40)
    return replace(
        config,
        system_prompt="Be brief.",
        files=files,
        models=(model,),
        default_model=model.name,
    )


class AttachTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "notes"
        self.outside = Path(self.tmp.name) / "outside"
        self.root.mkdir()
        self.outside.mkdir()
        db.configure(Path(self.tmp.name) / "chat.sqlite")
        self.files = _files(self.root)
        self.config = _app_config(self.files)

    def test_list_hides_secrets_and_paths(self):
        (self.root / "notes.txt").write_text("hello notes", encoding="utf-8")
        (self.root / ".env").write_text("TOKEN=1", encoding="utf-8")
        secret = self.outside / "secret.txt"
        secret.write_text("hidden", encoding="utf-8")
        (self.root / "link.txt").symlink_to(secret)

        with patch("app.main.config", self.config):
            client = open_client()
            roots = client.get("/api/files").json()
            self.assertTrue(roots["enabled"])
            self.assertEqual(len(roots["entries"]), 1)
            listed = client.get(f"/api/files/{roots['entries'][0]['id']}").json()

        names = [item["name"] for item in listed["entries"]]
        self.assertEqual(names, ["notes.txt"])
        self.assertNotIn(str(self.root), str(listed))
        self.assertNotIn("TOKEN", str(listed))
        self.assertNotIn("hidden", str(listed))

    def test_attach_puts_the_file_in_the_slot_and_remove_clears_it(self):
        (self.root / "notes.txt").write_text("hello notes", encoding="utf-8")
        seen = {}

        class FakeAdapter:
            def __init__(self, model):
                pass

            async def stream(self, messages):
                seen["messages"] = list(messages)
                yield "ok"

        with patch("app.main.config", self.config), patch("app.main.ChatAdapter", FakeAdapter):
            client = open_client()
            file_id = self._file_id(client, "notes.txt")
            attached = client.post("/api/attachments", json={"file_id": file_id})
            self.assertEqual(attached.status_code, 200, attached.text)
            body = attached.json()
            self.assertEqual(body["attachment"]["name"], "notes.txt")
            self.assertNotIn("path", body["attachment"])
            self.assertNotIn(str(self.root), attached.text)
            conversation_id = body["conversation"]["id"]

            chat = client.post(
                "/api/chat",
                json={"conversation_id": conversation_id, "content": "What does the file say?"},
            )
            self.assertEqual(chat.status_code, 200, chat.text)
            roles = [item["role"] for item in seen["messages"]]
            self.assertEqual(roles, ["system", "system", "user", "user"])
            self.assertEqual(seen["messages"][1]["content"], "Reference only.")
            self.assertIn("hello notes", seen["messages"][2]["content"])
            self.assertNotIn("hello notes", seen["messages"][0]["content"])
            self.assertNotIn(str(self.root), seen["messages"][2]["content"])

            removed = client.delete(f"/api/attachments/{body['attachment']['id']}")
            self.assertEqual(removed.status_code, 200)
            client.post(
                "/api/chat",
                json={"conversation_id": conversation_id, "content": "Still there?"},
            )
            self.assertNotIn("hello notes", str(seen["messages"]))

    def test_oversized_file_is_not_attached_or_truncated(self):
        (self.root / "long.txt").write_text("word " * 2000 + "ENDMARKER", encoding="utf-8")
        with patch("app.main.config", self.config):
            client = open_client()
            file_id = self._file_id(client, "long.txt")
            refused = client.post("/api/attachments", json={"file_id": file_id})
            self.assertEqual(refused.status_code, 409)
            self.assertEqual(refused.json()["code"], "file_over_budget")
            self.assertNotIn("ENDMARKER", refused.text)
            self.assertEqual(client.get("/api/conversations").json()["conversations"], [])

            indexed = client.post(
                "/api/attachments",
                json={"file_id": file_id, "choice": "index"},
            )
            self.assertEqual(indexed.status_code, 409)
            self.assertEqual(indexed.json()["detail"], strings["file_index_unavailable"])
            self.assertEqual(client.get("/api/conversations").json()["conversations"], [])

            prefixed = client.post(
                "/api/attachments",
                json={"file_id": file_id, "choice": "prefix"},
            )
            self.assertEqual(prefixed.status_code, 200, prefixed.text)
            self.assertTrue(prefixed.json()["attachment"]["use_prefix"])
            conversation_id = prefixed.json()["conversation"]["id"]

        seen = {}

        class FakeAdapter:
            def __init__(self, model):
                pass

            async def stream(self, messages):
                seen["messages"] = list(messages)
                yield "ok"

        with patch("app.main.config", self.config), patch("app.main.ChatAdapter", FakeAdapter):
            client = open_client()
            chat = client.post(
                "/api/chat",
                json={"conversation_id": conversation_id, "content": "Summarize the start."},
            )
            self.assertEqual(chat.status_code, 200, chat.text)
            joined = "\n".join(item["content"] for item in seen["messages"])
            self.assertIn("BEGIN FILE", joined)
            self.assertIn("Only the beginning of the file is included.", joined)
            self.assertNotIn("ENDMARKER", joined)
            self.assertNotIn("word", seen["messages"][0]["content"])
            self.assertNotIn("word", seen["messages"][1]["content"])

    def test_a_path_is_rejected_and_delete_removes_the_row(self):
        (self.root / "notes.txt").write_text("hello notes", encoding="utf-8")
        with patch("app.main.config", self.config):
            client = open_client()
            file_id = self._file_id(client, "notes.txt")
            rejected = client.post(
                "/api/attachments",
                json={"file_id": file_id, "path": str(self.root / "notes.txt")},
            )
            self.assertEqual(rejected.status_code, 422)
            attached = client.post("/api/attachments", json={"file_id": file_id})
            conversation_id = attached.json()["conversation"]["id"]
            self.assertIsNotNone(db.get_attachment(conversation_id))
            deleted = client.delete(f"/api/conversations/{conversation_id}")
            self.assertEqual(deleted.status_code, 200)
            self.assertIsNone(db.get_attachment(conversation_id))

    def _file_id(self, client, name: str) -> str:
        roots = client.get("/api/files").json()
        listed = client.get(f"/api/files/{roots['entries'][0]['id']}").json()
        return next(item["id"] for item in listed["entries"] if item["name"] == name)


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name)
        (self.data / "unused").mkdir()
        db.configure(self.data / "chat.sqlite")
        self.config = replace(_app_config(_files(self.data / "unused")), data_dir=self.data)

    def test_upload_keeps_a_copy_and_remove_deletes_only_that_copy(self):
        seen = {}

        class FakeAdapter:
            def __init__(self, model):
                pass

            async def stream(self, messages):
                seen["messages"] = list(messages)
                yield "ok"

        with patch("app.main.config", self.config), patch("app.main.ChatAdapter", FakeAdapter):
            client = open_client()
            rejected = client.post(
                "/api/attachments/upload",
                params={"filename": "../.env"},
                content=b"TOKEN=1",
                headers={"Content-Type": "application/octet-stream"},
            )
            self.assertEqual(rejected.status_code, 400)
            self.assertEqual(self._copies(), [])

            attached = client.post(
                "/api/attachments/upload",
                params={"filename": "notes.txt"},
                content=b"hello notes",
                headers={"Content-Type": "application/octet-stream"},
            )
            self.assertEqual(attached.status_code, 200, attached.text)
            body = attached.json()
            self.assertEqual(body["attachment"]["name"], "notes.txt")
            self.assertNotIn("path", body["attachment"])
            self.assertNotIn(str(self.data), attached.text)
            conversation_id = body["conversation"]["id"]
            chat = client.post(
                "/api/chat",
                json={"conversation_id": conversation_id, "content": "What does it say?"},
            )
            self.assertEqual(chat.status_code, 200, chat.text)
            self.assertEqual(seen["messages"][1]["content"], "Reference only.")
            self.assertIn("hello notes", seen["messages"][2]["content"])
            self.assertNotIn("hello notes", seen["messages"][0]["content"])
            self.assertNotIn(str(self.data), seen["messages"][2]["content"])
            removed = client.delete(f"/api/attachments/{body['attachment']['id']}")
            self.assertEqual(removed.status_code, 200)
            self.assertEqual(self._copies(), [])

    def test_upload_over_budget_can_keep_a_prefix_or_drop_the_copy(self):
        payload = ("word " * 2000 + "ENDMARKER").encode()
        with patch("app.main.config", self.config):
            client = open_client()
            refused = self._upload(client, "long.txt", payload)
            self.assertEqual(refused.status_code, 409)
            upload_id = refused.json()["upload_id"]
            self.assertTrue(upload_id)
            indexed = client.post(
                "/api/attachments",
                json={"upload_id": upload_id, "choice": "index"},
            )
            self.assertEqual(indexed.status_code, 409)
            self.assertEqual(indexed.json()["detail"], strings["file_index_unavailable"])
            self.assertEqual(self._copies(), [])

            refused = self._upload(client, "long.txt", payload)
            upload_id = refused.json()["upload_id"]
            prefixed = client.post(
                "/api/attachments",
                json={"upload_id": upload_id, "choice": "prefix"},
            )
            self.assertEqual(prefixed.status_code, 200, prefixed.text)
            self.assertTrue(prefixed.json()["attachment"]["use_prefix"])
            conversation_id = prefixed.json()["conversation"]["id"]

        seen = {}

        class FakeAdapter:
            def __init__(self, model):
                pass

            async def stream(self, messages):
                seen["messages"] = list(messages)
                yield "ok"

        with patch("app.main.config", self.config), patch("app.main.ChatAdapter", FakeAdapter):
            client = open_client()
            chat = client.post(
                "/api/chat",
                json={"conversation_id": conversation_id, "content": "Summarize the start."},
            )
            self.assertEqual(chat.status_code, 200, chat.text)
            joined = "\n".join(item["content"] for item in seen["messages"])
            self.assertIn("BEGIN FILE", joined)
            self.assertIn("Only the beginning of the file is included.", joined)
            self.assertNotIn("ENDMARKER", joined)
            self.assertNotIn(str(self.data), joined)

    def _upload(self, client, name: str, payload: bytes):
        return client.post(
            "/api/attachments/upload",
            params={"filename": name},
            content=payload,
            headers={"Content-Type": "application/octet-stream"},
        )

    def _copies(self) -> list[Path]:
        root = self.data / "uploads"
        if not root.exists():
            return []
        return [path for path in root.rglob("*") if path.is_file()]
