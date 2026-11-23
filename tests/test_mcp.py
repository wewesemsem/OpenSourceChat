"""MCP spike: config-shaped client, allowlist, step cap, and the review scores."""

import asyncio
import os
import sys
import unittest
from dataclasses import replace
from unittest.mock import patch

from app.config import McpAuth, load_config
from app.mcp.connect import (
    call_allowed_tool,
    list_allowed_tools,
    open_session,
    stdio_parameters,
)
from app.mcp.errors import McpError, format_error
from app.mcp.loop import StepRecord, run_tool_loop
from app.mcp.spike import LoopResult, build_report, evaluate_model, load_strings, score_prompt
from app.mcp.spike_server import FACTS, TOOL_NAME
from app.models.adapter import ModelReply, ToolCall


def _texts():
    return load_strings()


class SpikeCatalogTests(unittest.TestCase):
    def test_prompts_match_the_read_only_server(self):
        config = load_config()
        prompts = config.mcp.spike.prompts
        self.assertEqual(len(prompts), 20)
        topics = []
        for prompt in prompts:
            self.assertEqual(prompt.tool, TOOL_NAME)
            topic = prompt.argument_map()["topic"]
            topics.append(topic)
            self.assertEqual(FACTS[topic], prompt.expect)
        self.assertEqual(set(topics), set(FACTS))


class StdioClientTests(unittest.TestCase):
    def test_parameters_use_this_interpreter_and_a_minimal_env(self):
        server = load_config().mcp.server("spike")
        os.environ["MCP_SHOULD_NOT_LEAK"] = "nope"
        self.addCleanup(os.environ.pop, "MCP_SHOULD_NOT_LEAK", None)
        params = stdio_parameters(server)
        self.assertEqual(params.command, sys.executable)
        self.assertIsNone(params.env)
        self.assertEqual(params.args, ["-m", "app.mcp.spike_server"])
        named = replace(server, env=("MCP_SHOULD_NOT_LEAK",))
        self.assertEqual(stdio_parameters(named).env, {"MCP_SHOULD_NOT_LEAK": "nope"})

    def test_lists_and_calls_only_the_allowlisted_tool(self):
        config = load_config()
        server = config.mcp.server("spike")

        async def run():
            async with open_session(server, config.mcp) as client:
                tools = await list_allowed_tools(client, server)
                self.assertEqual([tool.name for tool in tools], [TOOL_NAME])
                answer = await call_allowed_tool(
                    client,
                    server,
                    TOOL_NAME,
                    {"topic": "reception_phone"},
                    read_timeout_seconds=10,
                )
                self.assertTrue(answer.ok)
                self.assertEqual(answer.text, "555-0148")
                with self.assertRaises(McpError) as caught:
                    await call_allowed_tool(
                        client,
                        server,
                        "other",
                        {},
                        read_timeout_seconds=10,
                    )
                self.assertEqual(caught.exception.code, "tool_refused")

        asyncio.run(run())


class HttpTransportTests(unittest.TestCase):
    def test_bearer_header_is_sent_and_redacted_from_errors(self):
        config = load_config()
        server = replace(
            config.mcp.server("company"),
            url="http://127.0.0.1:9/mcp",
            enabled=True,
        )
        os.environ["MCP_COMPANY_TOKEN"] = "s3cret"
        self.addCleanup(os.environ.pop, "MCP_COMPANY_TOKEN", None)
        seen = {}

        def explode(url, headers=None, timeout=None, sse_read_timeout=None, httpx_client_factory=None):
            seen["url"] = url
            seen["headers"] = headers
            client = httpx_client_factory(headers=headers, timeout=timeout)
            seen["trust_env"] = client.trust_env
            raise RuntimeError(f"proxy said {headers['Authorization']}")

        async def run():
            with patch("app.mcp.connect.sse_client", explode):
                async with open_session(server, config.mcp):
                    pass

        with self.assertRaises(McpError) as caught:
            asyncio.run(run())
        self.assertEqual(seen["headers"], {"Authorization": "Bearer s3cret"})
        self.assertFalse(seen["trust_env"])
        self.assertEqual(caught.exception.code, "connect_failed")
        self.assertNotIn("s3cret", caught.exception.detail)
        self.assertIn("[redacted]", caught.exception.detail)
        self.assertNotIn("s3cret", format_error(_texts(), caught.exception))

    def test_streamable_http_uses_that_client(self):
        config = load_config()
        server = replace(
            config.mcp.server("spike"),
            transport="streamable_http",
            url="http://127.0.0.1:9/mcp",
            command="",
            args=(),
            cwd="",
            env=(),
            auth=McpAuth(type="none", token_env=""),
            enabled=True,
        )
        seen = {}

        def explode(url, *, http_client=None, terminate_on_close=True, max_sse_event_size=None):
            seen["url"] = url
            seen["trust_env"] = http_client.trust_env
            raise RuntimeError("down")

        async def run():
            with patch("app.mcp.connect.streamable_http_client", explode):
                async with open_session(server, config.mcp):
                    pass

        with self.assertRaises(McpError) as caught:
            asyncio.run(run())
        self.assertEqual(seen["url"], "http://127.0.0.1:9/mcp")
        self.assertFalse(seen["trust_env"])
        self.assertEqual(caught.exception.code, "connect_failed")

    def test_a_disallowed_host_is_rejected_before_connect(self):
        config = load_config()
        server = replace(
            config.mcp.server("spike"),
            transport="streamable_http",
            url="https://evil.example/mcp",
            command="",
            args=(),
            cwd="",
            env=(),
            auth=McpAuth(type="none", token_env=""),
        )

        def explode(*args, **kwargs):
            raise AssertionError("connected")

        async def run():
            with patch("app.mcp.connect.streamable_http_client", explode):
                async with open_session(server, config.mcp):
                    pass

        with self.assertRaises(McpError) as caught:
            asyncio.run(run())
        self.assertEqual(caught.exception.code, "host_rejected")

    def test_remote_http_is_rejected(self):
        config = load_config()
        settings = replace(config.mcp, allowed_hosts=("tools.example",))
        server = replace(
            config.mcp.server("spike"),
            transport="sse",
            url="http://tools.example/mcp",
            command="",
            args=(),
            cwd="",
            env=(),
            auth=McpAuth(type="none", token_env=""),
        )

        async def run():
            async with open_session(server, settings):
                pass

        with self.assertRaises(McpError) as caught:
            asyncio.run(run())
        self.assertEqual(caught.exception.code, "https_required")

    def test_missing_token_names_the_variable_only(self):
        config = load_config()
        server = replace(config.mcp.server("company"), url="http://127.0.0.1:9/mcp")
        os.environ.pop("MCP_COMPANY_TOKEN", None)

        async def run():
            async with open_session(server, config.mcp):
                pass

        with self.assertRaises(McpError) as caught:
            asyncio.run(run())
        self.assertEqual(caught.exception.code, "token_missing")
        message = format_error(_texts(), caught.exception)
        self.assertIn("MCP_COMPANY_TOKEN", message)
        self.assertNotIn("Bearer", message)


class LoopTests(unittest.TestCase):
    def test_runs_the_tool_then_asks_the_model_again(self):
        texts = _texts()
        seen = []

        async def complete(messages, tools):
            seen.append((list(messages), list(tools)))
            if len(seen) == 1:
                return ModelReply(
                    "",
                    (
                        ToolCall(
                            "c1",
                            TOOL_NAME,
                            {"topic": "reception_phone"},
                            '{"topic":"reception_phone"}',
                            False,
                        ),
                    ),
                )
            return ModelReply("555-0148", ())

        async def call_tool(name, arguments):
            self.assertEqual(name, TOOL_NAME)
            self.assertEqual(dict(arguments), {"topic": "reception_phone"})
            return "555-0148", True

        result = asyncio.run(
            run_tool_loop(
                [{"role": "user", "content": "What phone number does reception use?"}],
                [{"type": "function"}],
                allowed=frozenset({TOOL_NAME}),
                max_steps=3,
                complete=complete,
                call_tool=call_tool,
                refusal=texts["mcp_tool_refused"],
                bad_arguments=texts["mcp_bad_arguments"],
                tool_failed=texts["mcp_tool_failed"],
                cap_message=texts["mcp_step_cap"],
            )
        )
        self.assertEqual(result.stop, "answer")
        self.assertEqual(result.text, "555-0148")
        self.assertEqual(result.steps[0].ok, True)
        self.assertEqual(seen[1][0][-1]["role"], "tool")
        self.assertEqual(seen[1][0][-1]["content"], "555-0148")
        self.assertEqual(seen[1][1], [{"type": "function"}])

    def test_stops_after_the_configured_cap(self):
        calls = {"n": 0}

        async def complete(messages, tools):
            if not tools:
                return ModelReply("enough", ())
            return ModelReply(
                "",
                (ToolCall("c", TOOL_NAME, {"topic": "reception_phone"}, "{}", False),),
            )

        async def call_tool(name, arguments):
            calls["n"] += 1
            return "555-0148", True

        result = asyncio.run(
            run_tool_loop(
                [{"role": "user", "content": "again"}],
                [{"type": "function"}],
                allowed=frozenset({TOOL_NAME}),
                max_steps=3,
                complete=complete,
                call_tool=call_tool,
                refusal="no",
                bad_arguments="bad",
                tool_failed="fail",
                cap_message="stopped",
            )
        )
        self.assertEqual(calls["n"], 3)
        self.assertEqual(result.stop, "cap")
        self.assertEqual(result.text, "enough")

    def test_refuses_a_tool_that_is_not_allowlisted(self):
        called = {"yes": False}

        async def complete(messages, tools):
            if any(item.get("role") == "tool" for item in messages):
                return ModelReply("I cannot do that.", ())
            return ModelReply("", (ToolCall("c", "delete_files", {}, "{}", False),))

        async def call_tool(name, arguments):
            called["yes"] = True
            return "gone", True

        result = asyncio.run(
            run_tool_loop(
                [{"role": "user", "content": "delete it"}],
                [{"type": "function"}],
                allowed=frozenset({TOOL_NAME}),
                max_steps=3,
                complete=complete,
                call_tool=call_tool,
                refusal="That tool is not allowed.",
                bad_arguments="bad",
                tool_failed="fail",
                cap_message="stopped",
            )
        )
        self.assertFalse(called["yes"])
        self.assertEqual(result.steps[0].text, "That tool is not allowed.")
        self.assertFalse(result.steps[0].ok)

    def test_invalid_arguments_are_not_sent(self):
        called = {"yes": False}

        async def complete(messages, tools):
            if any(item.get("role") == "tool" for item in messages):
                return ModelReply("retry failed", ())
            return ModelReply("", (ToolCall("c", TOOL_NAME, None, "{", True),))

        async def call_tool(name, arguments):
            called["yes"] = True
            return "x", True

        result = asyncio.run(
            run_tool_loop(
                [{"role": "user", "content": "lookup"}],
                [{"type": "function"}],
                allowed=frozenset({TOOL_NAME}),
                max_steps=3,
                complete=complete,
                call_tool=call_tool,
                refusal="no",
                bad_arguments="Those tool arguments are not valid.",
                tool_failed="fail",
                cap_message="stopped",
            )
        )
        self.assertFalse(called["yes"])
        self.assertFalse(result.steps[0].ok)
        self.assertEqual(result.steps[0].text, "Those tool arguments are not valid.")


class ScoreTests(unittest.TestCase):
    def test_scores_a_correct_call_and_a_miss(self):
        prompt = load_config().mcp.spike.prompts[0]
        good = LoopResult(
            "555-0148",
            (StepRecord(TOOL_NAME, {"topic": "reception_phone"}, True, "555-0148"),),
            "answer",
            1.25,
        )
        row = score_prompt(prompt, good)
        self.assertTrue(row["tool_chosen_correctly"])
        self.assertTrue(row["arguments_valid"])
        self.assertTrue(row["task_completed"])
        self.assertEqual(row["steps"], 1)
        missed = LoopResult("I don't know", (), "answer", 0.4)
        row = score_prompt(prompt, missed)
        self.assertFalse(row["tool_chosen_correctly"])
        self.assertFalse(row["arguments_valid"])
        self.assertFalse(row["task_completed"])

    def test_unsupported_model_skips_the_prompts(self):
        config = load_config()
        texts = _texts()

        async def complete(model_name, messages, tools):
            raise AssertionError("model was called")

        async def call_tool(name, arguments):
            raise AssertionError("tool was called")

        report = asyncio.run(
            evaluate_model(
                config,
                texts,
                model_name="gemma3:4b",
                support="unsupported",
                tools=[{"type": "function"}],
                allowed=frozenset({TOOL_NAME}),
                complete=complete,
                call_tool=call_tool,
            )
        )
        self.assertEqual(report["tool_support"], "unsupported")
        self.assertEqual(len(report["prompts"]), 20)
        self.assertTrue(all(row["stop"] == "skipped" for row in report["prompts"]))
        self.assertEqual(report["prompts"][0]["detail"], texts["tools_unsupported"])
        self.assertEqual(report["summary"]["task_completed"], 0)

    def test_disabled_servers_do_not_run_prompts(self):
        config = load_config()
        texts = _texts()

        async def support(model, *, timeout_seconds):
            return "supported"

        with patch("app.mcp.spike.model_tool_support", support):
            report = asyncio.run(build_report(config, texts, server_name=""))
        self.assertFalse(report["tools_enabled"])
        self.assertIsNone(report["server"])
        self.assertEqual(report["note"], texts["mcp_tools_off"])
        self.assertEqual(report["models"][0]["prompts"][0]["stop"], "skipped")

    def test_script_can_list_a_disabled_stdio_server(self):
        config = load_config()
        texts = _texts()

        async def support(model, *, timeout_seconds):
            return "unsupported"

        with patch("app.mcp.spike.model_tool_support", support):
            report = asyncio.run(build_report(config, texts, server_name="spike"))
        self.assertEqual(report["server"]["name"], "spike")
        self.assertEqual(report["server"]["transport"], "stdio")
        self.assertFalse(report["server"]["enabled_in_config"])
        self.assertEqual(report["server"]["tools"][0]["name"], TOOL_NAME)
        self.assertEqual(report["note"], texts["mcp_script_only"])
        self.assertEqual(report["models"][0]["tool_support"], "unsupported")
        self.assertEqual(len(report["models"][0]["prompts"]), 20)
