"""One model turn, then at most a few allowlisted tool calls.

The cap is the whole safety stop for this spike. A refused or failed call is
handed back as text so the model can answer without that tool.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

from app.mcp.errors import McpError
from app.models.adapter import ModelReply, ToolCall

Complete = Callable[[list[dict], list[dict]], Awaitable[ModelReply]]
Invoke = Callable[[str, Mapping[str, object]], Awaitable[tuple[str, bool]]]


@dataclass(frozen=True)
class StepRecord:
    name: str
    arguments: dict | None
    ok: bool
    text: str


@dataclass(frozen=True)
class LoopResult:
    text: str
    steps: tuple[StepRecord, ...]
    stop: str
    seconds: float


async def run_tool_loop(
    messages: list[dict],
    tools: list[dict],
    *,
    allowed: frozenset[str],
    max_steps: int,
    complete: Complete,
    call_tool: Invoke,
    refusal: str,
    bad_arguments: str,
    tool_failed: str,
    cap_message: str,
) -> LoopResult:
    started = time.perf_counter()
    steps: list[StepRecord] = []
    transcript = list(messages)
    while len(steps) < max_steps:
        reply = await complete(transcript, tools)
        if not reply.tool_calls:
            return _finish(reply.content, steps, "answer", started)
        transcript.append(_assistant_message(reply))
        for call in reply.tool_calls:
            if len(steps) >= max_steps:
                break
            record, content = await _run_call(
                call,
                allowed=allowed,
                call_tool=call_tool,
                refusal=refusal,
                bad_arguments=bad_arguments,
                tool_failed=tool_failed,
            )
            steps.append(record)
            transcript.append(
                {"role": "tool", "tool_call_id": call.id, "content": content}
            )
        if len(steps) >= max_steps:
            # One last reply with the tools removed, so the model can answer
            # without starting another tool call.
            closing = await complete(transcript, [])
            return _finish(closing.content or cap_message, steps, "cap", started)
    return _finish(cap_message, steps, "cap", started)


async def _run_call(
    call: ToolCall,
    *,
    allowed: frozenset[str],
    call_tool: Invoke,
    refusal: str,
    bad_arguments: str,
    tool_failed: str,
) -> tuple[StepRecord, str]:
    if call.invalid or call.arguments is None:
        return StepRecord(call.name, None, False, bad_arguments), bad_arguments
    if call.name not in allowed:
        return StepRecord(call.name, call.arguments, False, refusal), refusal
    try:
        text, ok = await call_tool(call.name, call.arguments)
    except McpError:
        return StepRecord(call.name, call.arguments, False, tool_failed), tool_failed
    except Exception:
        return StepRecord(call.name, call.arguments, False, tool_failed), tool_failed
    return StepRecord(call.name, call.arguments, ok, text), text


def _assistant_message(reply: ModelReply) -> dict:
    return {
        "role": "assistant",
        "content": reply.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.raw or "{}"},
            }
            for call in reply.tool_calls
        ],
    }


def _finish(text: str, steps: list[StepRecord], stop: str, started: float) -> LoopResult:
    elapsed = round(time.perf_counter() - started, 3)
    return LoopResult(text=text, steps=tuple(steps), stop=stop, seconds=elapsed)
