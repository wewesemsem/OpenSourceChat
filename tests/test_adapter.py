"""Stream parsing: visible tokens only, direct localhost client."""

import asyncio
import unittest

import httpx

from app.config import ModelConfig
from app.models.adapter import AdapterError, chat_payload, direct_client, parse_sse_line


class ParseSseTests(unittest.TestCase):
    def test_returns_visible_content(self):
        line = 'data: {"choices":[{"delta":{"content":"Hi"}}]}'
        parsed = parse_sse_line(line)
        self.assertEqual(parsed.delta, "Hi")
        self.assertFalse(parsed.done)

    def test_prefers_content_over_reasoning(self):
        line = (
            'data: {"choices":[{"delta":'
            '{"reasoning_content":"hidden","content":"Shown"}}]}'
        )
        self.assertEqual(parse_sse_line(line).delta, "Shown")

    def test_skips_reasoning_only_chunks(self):
        line = 'data: {"choices":[{"delta":{"reasoning_content":"hidden"}}]}'
        self.assertIsNone(parse_sse_line(line))

    def test_done_and_ignorable_lines(self):
        self.assertTrue(parse_sse_line("data: [DONE]").done)
        self.assertIsNone(parse_sse_line(""))
        self.assertIsNone(parse_sse_line(": keep-alive"))
        self.assertIsNone(parse_sse_line('data: {"choices":[{"delta":{"role":"assistant"}}]}'))

    def test_error_and_invalid_json(self):
        with self.assertRaises(AdapterError):
            parse_sse_line('data: {"error":{"message":"missing model"}}')
        with self.assertRaises(AdapterError):
            parse_sse_line("data: {not-json")


class ChatPayloadTests(unittest.TestCase):
    def test_reserves_output_and_context(self):
        model = ModelConfig(
            name="gemma3:12b",
            label="Gemma",
            base_url="http://127.0.0.1:11434/v1",
            context_size=8192,
            reserved_output_tokens=1024,
            temperature=0,
        )
        payload = chat_payload(model, [{"role": "user", "content": "Hi"}])
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertEqual(payload["options"], {"num_ctx": 8192})
        self.assertEqual(payload["messages"][0]["content"], "Hi")


class DirectClientTests(unittest.TestCase):
    def test_ignores_proxy_environment(self):
        client = direct_client(httpx.Timeout(1.0))
        try:
            self.assertFalse(client._trust_env)
        finally:
            asyncio.run(client.aclose())
