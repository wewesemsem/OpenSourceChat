"""Chat routes stream one reply and store the turn."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import db
from app.main import app, config


def open_client() -> TestClient:
    return TestClient(
        app,
        base_url=f"http://127.0.0.1:{config.port}",
        headers={config.security.csrf_header: config.security.csrf_token},
    )


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        db.configure(Path(self.tmp.name) / "chat.sqlite")

    def test_streams_and_stores_the_turn(self):
        seen = {}

        class FakeAdapter:
            def __init__(self, model):
                seen["model"] = model.name

            async def stream(self, messages):
                seen["messages"] = list(messages)
                yield "Hello"
                yield " there"

        with patch("app.main.ChatAdapter", FakeAdapter):
            client = open_client()
            response = client.post("/api/chat", json={"content": "Hi"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertIn('data: {"delta": "Hello"}', response.text)
        self.assertIn('"done": true', response.text)
        self.assertEqual(seen["model"], "gemma3:12b")
        self.assertEqual(seen["messages"][0]["role"], "system")
        self.assertEqual(seen["messages"][1]["content"], "Hi")
        self.assertIn('"trimmed": false', response.text)

        client = open_client()
        listed = client.get("/api/conversations").json()["conversations"]
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["title"], "Hi")
        detail = client.get(f"/api/conversations/{listed[0]['id']}").json()
        self.assertEqual([item["content"] for item in detail["messages"]], ["Hi", "Hello there"])
        self.assertEqual(detail["messages"][1]["model"], "gemma3:12b")
        self.assertIsNotNone(detail["messages"][1]["latency_ms"])

    def test_regenerate_replaces_the_assistant_message(self):
        replies = iter(["First", "Second"])

        class FakeAdapter:
            def __init__(self, model):
                self.model = model

            async def stream(self, messages):
                yield next(replies)

        with patch("app.main.ChatAdapter", FakeAdapter):
            client = open_client()
            first = client.post("/api/chat", json={"content": "Hi"})
            self.assertIn("First", first.text)
            conversation_id = client.get("/api/conversations").json()["conversations"][0]["id"]
            second = client.post(
                "/api/chat",
                json={"conversation_id": conversation_id, "regenerate": True, "model": "gemma3:12b"},
            )
            self.assertIn("Second", second.text)
            detail = client.get(f"/api/conversations/{conversation_id}").json()
        self.assertEqual([item["content"] for item in detail["messages"]], ["Hi", "Second"])

    def test_rename_and_delete(self):
        class FakeAdapter:
            def __init__(self, model):
                pass

            async def stream(self, messages):
                yield "Ok"

        with patch("app.main.ChatAdapter", FakeAdapter):
            client = open_client()
            client.post("/api/chat", json={"content": "Hi"})
            conversation_id = client.get("/api/conversations").json()["conversations"][0]["id"]
            renamed = client.patch(
                f"/api/conversations/{conversation_id}",
                json={"title": "Notes"},
            )
            self.assertEqual(renamed.status_code, 200)
            self.assertEqual(renamed.json()["title"], "Notes")
            deleted = client.delete(f"/api/conversations/{conversation_id}")
            self.assertEqual(deleted.status_code, 200)
            self.assertEqual(client.get("/api/conversations").json()["conversations"], [])
            self.assertEqual(db.list_messages(conversation_id), [])

    def test_leaves_older_turns_out_of_the_prompt(self):
        from app.config import ModelConfig
        from app.main import config
        from app.memory.context import message_tokens

        short = "tail question"
        long = ("alpha " * 200).strip()
        budget = message_tokens(config.system_prompt) + message_tokens(short)
        tiny = ModelConfig(
            name="gemma3:12b",
            label="Gemma 3 12B",
            base_url="http://127.0.0.1:11434/v1",
            context_size=budget + 32,
            reserved_output_tokens=32,
            temperature=0,
        )
        seen = {}

        class FakeAdapter:
            def __init__(self, model):
                pass

            async def stream(self, messages):
                seen["messages"] = list(messages)
                yield "Ok"

        with (
            patch("app.main.ChatAdapter", FakeAdapter),
            patch("app.main.find_model", return_value=tiny),
        ):
            client = open_client()
            first = client.post("/api/chat", json={"content": long})
            self.assertEqual(first.status_code, 200)
            self.assertIn('"trimmed": true', first.text)
            conversation_id = client.get("/api/conversations").json()["conversations"][0]["id"]
            second = client.post(
                "/api/chat",
                json={"content": short, "conversation_id": conversation_id},
            )

        self.assertIn('"trimmed": true', second.text)
        self.assertEqual(seen["messages"][0]["content"], config.system_prompt)
        sent_user = [item["content"] for item in seen["messages"] if item["role"] == "user"]
        self.assertEqual(sent_user, [short])
        detail = client.get(f"/api/conversations/{conversation_id}").json()
        stored = [item["content"] for item in detail["messages"] if item["role"] == "user"]
        self.assertEqual(stored, [long, short])

    def test_rejects_an_unknown_model(self):
        client = open_client()
        response = client.post("/api/chat", json={"content": "Hi", "model": "nope"})
        self.assertEqual(response.status_code, 400)

    def test_meta_comes_from_config_and_strings(self):
        client = open_client()
        payload = client.get("/api/meta").json()
        self.assertEqual(payload["title"], "Local Chat")
        self.assertEqual(payload["default_model"], "gemma3:12b")
        self.assertEqual(payload["api_base"], "")
        self.assertEqual(payload["footer"], "Runs locally. Your chats stay on this machine.")
        self.assertEqual(payload["strings"]["send"], "Send")
        self.assertIn("Explain a token budget", payload["examples"][0])
        self.assertTrue(payload["files_enabled"])
        self.assertTrue(payload["upload_enabled"])
        self.assertIn("txt", payload["file_extensions"])

    def test_models_use_config_labels(self):
        async def ready(model):
            return "ready"

        with patch("app.main.probe_model", ready):
            client = open_client()
            payload = client.get("/api/models").json()
        self.assertEqual(payload["models"][0]["name"], "gemma3:12b")
        self.assertEqual(payload["models"][0]["label"], "Gemma 3 12B")
        self.assertEqual(payload["models"][0]["status"], "ready")
        self.assertEqual(payload["models"][1]["name"], "llama3.1:8b")
        self.assertEqual(payload["models"][1]["label"], "Llama 3.1 8B")
        self.assertEqual(payload["models"][1]["status"], "ready")
