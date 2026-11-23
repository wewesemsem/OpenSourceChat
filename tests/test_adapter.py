"""Stream parsing: visible tokens only, direct localhost client."""

import asyncio
import unittest

import httpx

from app.config import ModelConfig
from app.models.adapter import (
    AdapterError,
    chat_payload,
    direct_client,
    generation_rate,
    model_tool_support,
    native_base,
    parse_completion,
    parse_sse_line,
    same_model,
    tool_support_from_show,
)


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

    def test_reads_usage_without_visible_text(self):
        line = (
            'data: {"choices":[{"delta":{}}],'
            '"usage":{"prompt_tokens":12,"completion_tokens":4}}'
        )
        parsed = parse_sse_line(line)
        self.assertIsNone(parsed.delta)
        self.assertEqual(parsed.usage.prompt_tokens, 12)
        self.assertEqual(parsed.usage.completion_tokens, 4)


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
        self.assertFalse(payload["think"])
        self.assertEqual(payload["keep_alive"], "30m")
        self.assertEqual(payload["stream_options"], {"include_usage": True})
        self.assertEqual(payload["messages"][0]["content"], "Hi")
        self.assertNotIn("tools", payload)
        self.assertTrue(payload["stream"])

    def test_tools_are_optional_and_streaming_stays_the_default(self):
        model = ModelConfig(
            name="gemma3:12b",
            label="Gemma",
            base_url="http://127.0.0.1:11434/v1",
            context_size=8192,
            reserved_output_tokens=1024,
            temperature=0,
        )
        payload = chat_payload(
            model,
            [{"role": "user", "content": "Hi"}],
            tools=[{"type": "function"}],
            stream=False,
        )
        self.assertEqual(payload["tools"], [{"type": "function"}])
        self.assertFalse(payload["stream"])
        self.assertNotIn("stream_options", payload)


class CompletionParseTests(unittest.TestCase):
    def test_reads_a_tool_call(self):
        reply = parse_completion(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "office_lookup",
                                        "arguments": '{"topic":"reception_phone"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        )
        self.assertEqual(reply.content, "")
        self.assertEqual(reply.tool_calls[0].name, "office_lookup")
        self.assertEqual(reply.tool_calls[0].arguments, {"topic": "reception_phone"})
        self.assertFalse(reply.tool_calls[0].invalid)

    def test_marks_arguments_that_are_not_an_object(self):
        reply = parse_completion(
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {"function": {"name": "office_lookup", "arguments": "{not"}}
                            ],
                        }
                    }
                ]
            }
        )
        self.assertTrue(reply.tool_calls[0].invalid)
        self.assertIsNone(reply.tool_calls[0].arguments)

    def test_tool_support_comes_from_capabilities(self):
        self.assertEqual(
            tool_support_from_show({"capabilities": ["completion", "tools"]}),
            "supported",
        )
        self.assertEqual(tool_support_from_show({"capabilities": ["completion"]}), "unsupported")
        self.assertEqual(tool_support_from_show({}), "unknown")
        self.assertEqual(tool_support_from_show("nope"), "unknown")

    def test_unknown_when_the_model_server_is_down(self):
        model = ModelConfig(
            name="gemma3:12b",
            label="Gemma",
            base_url="http://127.0.0.1:1/v1",
            context_size=8192,
            reserved_output_tokens=1024,
            temperature=0,
        )
        support = asyncio.run(model_tool_support(model, timeout_seconds=0.2))
        self.assertEqual(support, "unknown")


class TimingHelperTests(unittest.TestCase):
    def test_rate_needs_a_real_interval(self):
        self.assertIsNone(generation_rate(0, 1))
        self.assertIsNone(generation_rate(10, 0.01))
        self.assertEqual(generation_rate(10, 0.5), 20.0)

    def test_native_base_and_loaded_name(self):
        self.assertEqual(native_base("http://127.0.0.1:11434/v1"), "http://127.0.0.1:11434")
        self.assertTrue(same_model("gemma3:4b", "gemma3:4b"))
        self.assertTrue(same_model("gemma3:4b:digest", "gemma3:4b"))
        self.assertFalse(same_model("gemma3:12b", "gemma3:4b"))


class DirectClientTests(unittest.TestCase):
    def test_ignores_proxy_environment(self):
        client = direct_client(httpx.Timeout(1.0))
        try:
            self.assertFalse(client._trust_env)
        finally:
            asyncio.run(client.aclose())
