"""Run the configured spike prompts and write a review file.

The app itself does not call tools. This command does, against one server.
Pass --server to use a server that is still disabled in config.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.config import ROOT, AppConfig, McpServerConfig, SpikePrompt, load_config
from app.mcp.connect import (
    call_allowed_tool,
    list_allowed_tools,
    openai_tool,
    open_session,
    select_server,
)
from app.mcp.errors import McpError, format_error
from app.mcp.loop import LoopResult, run_tool_loop
from app.models.adapter import AdapterError, ChatAdapter, model_tool_support

STRINGS_PATH = Path(__file__).resolve().parents[1] / "strings.json"


def load_strings() -> dict[str, str]:
    payload = json.loads(STRINGS_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("strings.json must be an object of strings")
    return {key: value for key, value in payload.items() if isinstance(key, str) and isinstance(value, str)}


def score_prompt(prompt: SpikePrompt, result: LoopResult) -> dict:
    first = result.steps[0] if result.steps else None
    chosen = first is not None and first.name == prompt.tool
    valid = chosen and first.arguments == prompt.argument_map()
    completed = prompt.expect.casefold() in result.text.casefold()
    return {
        "id": prompt.id,
        "tool_chosen_correctly": chosen,
        "arguments_valid": valid,
        "task_completed": completed,
        "steps": len(result.steps),
        "seconds": result.seconds,
        "stop": result.stop,
        "answer": result.text,
    }


def skipped_prompt(prompt: SpikePrompt, reason: str) -> dict:
    return {
        "id": prompt.id,
        "tool_chosen_correctly": False,
        "arguments_valid": False,
        "task_completed": False,
        "steps": 0,
        "seconds": 0,
        "stop": "skipped",
        "detail": reason,
        "answer": "",
    }


async def evaluate_model(
    config: AppConfig,
    texts: dict[str, str],
    *,
    model_name: str,
    support: str,
    tools: list[dict],
    allowed: frozenset[str],
    complete,
    call_tool,
) -> dict:
    prompts = config.mcp.spike.prompts
    if support != "supported":
        reason = texts["tools_unsupported"] if support == "unsupported" else texts["tools_unknown"]
        print(f"{model_name}  tool support: {support}", flush=True)
        rows = [skipped_prompt(prompt, reason) for prompt in prompts]
        return _model_report(model_name, support, rows)
    if not tools:
        rows = [skipped_prompt(prompt, texts["mcp_tools_off"]) for prompt in prompts]
        return _model_report(model_name, support, rows)
    rows = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": config.mcp.spike.system_prompt},
            {"role": "user", "content": prompt.text},
        ]
        try:
            result = await run_tool_loop(
                messages,
                tools,
                allowed=allowed,
                max_steps=config.mcp.max_steps,
                complete=lambda transcript, defined, _model=model_name: complete(
                    _model, transcript, defined
                ),
                call_tool=call_tool,
                refusal=texts["mcp_tool_refused"],
                bad_arguments=texts["mcp_bad_arguments"],
                tool_failed=texts["mcp_tool_failed"],
                cap_message=texts["mcp_step_cap"].replace("{steps}", str(config.mcp.max_steps)),
            )
        except AdapterError as exc:
            row = skipped_prompt(prompt, str(exc))
            row["stop"] = "error"
            rows.append(row)
            print(f"{model_name}  {prompt.id}  error", flush=True)
            continue
        row = score_prompt(prompt, result)
        rows.append(row)
        print(
            f"{model_name}  {row['id']}  tool={row['tool_chosen_correctly']}  "
            f"args={row['arguments_valid']}  done={row['task_completed']}  "
            f"steps={row['steps']}  {row['seconds']}s",
            flush=True,
        )
    return _model_report(model_name, support, rows)


def render_report(report: dict) -> str:
    lines = []
    server = report.get("server")
    if not server:
        lines.append(report.get("note") or "")
    else:
        lines.append(f"{server['name']} ({server['transport']}) tools:")
        for tool in server["tools"]:
            lines.append(f"  {tool['name']}")
        if not server["tools"]:
            lines.append("  (none)")
    for model in report.get("models") or []:
        summary = model["summary"]
        lines.append(
            f"{model['name']}  tool support: {model['tool_support']}  "
            f"tool {summary['tool_chosen_correctly']}/{summary['prompts']}  "
            f"args {summary['arguments_valid']}/{summary['prompts']}  "
            f"done {summary['task_completed']}/{summary['prompts']}"
        )
    return "\n".join(lines)


def write_report(report: dict, output: str) -> Path:
    path = ROOT / output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


async def build_report(config: AppConfig, texts: dict[str, str], *, server_name: str) -> dict:
    try:
        server = select_server(config.mcp, server_name)
    except McpError as exc:
        return _failed(config, texts, format_error(texts, exc))
    if server is None:
        models = await _supports(config, texts, tools=[], allowed=frozenset(), call_tool=_unused)
        return {
            "tools_enabled": False,
            "server": None,
            "note": texts["mcp_tools_off"],
            "models": models,
        }
    try:
        async with open_session(server, config.mcp) as client:
            listed = await list_allowed_tools(client, server)
            tools = [openai_tool(item) for item in listed]
            allowed = frozenset(item.name for item in listed)

            async def call_tool(name, arguments):
                answer = await call_allowed_tool(
                    client,
                    server,
                    name,
                    arguments,
                    read_timeout_seconds=config.mcp.read_timeout_seconds,
                )
                return answer.text, answer.ok

            models = await _supports(
                config,
                texts,
                tools=tools,
                allowed=allowed,
                call_tool=call_tool,
            )
    except McpError as exc:
        return _failed(config, texts, format_error(texts, exc), detail=exc.detail, server=server)
    return {
        "tools_enabled": config.mcp.enabled,
        "server": _server_summary(server, listed),
        "note": "" if server.enabled else texts["mcp_script_only"],
        "models": models,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the MCP tool-calling spike.")
    parser.add_argument("--server", default="", help="Server name. Disabled servers are allowed here.")
    parser.add_argument("--output", default="", help="Review file. Defaults to mcp.spike.output.")
    args = parser.parse_args(argv)
    config = load_config()
    texts = load_strings()
    output = args.output or config.mcp.spike.output
    if not output:
        raise SystemExit("mcp.spike.output is empty")
    report = asyncio.run(build_report(config, texts, server_name=args.server))
    path = write_report(report, output)
    print(render_report(report))
    print(path)
    return 1 if report.get("error") else 0


async def _supports(config, texts, *, tools, allowed, call_tool) -> list[dict]:
    rows = []
    for model in config.models:
        support = await model_tool_support(
            model,
            timeout_seconds=config.mcp.model_probe_timeout_seconds,
        )

        async def complete(model_name, messages, defined, _model=model):
            if model_name != _model.name:
                raise RuntimeError(model_name)
            return await ChatAdapter(_model).complete(messages, tools=defined)

        rows.append(
            await evaluate_model(
                config,
                texts,
                model_name=model.name,
                support=support,
                tools=tools,
                allowed=allowed,
                complete=complete,
                call_tool=call_tool,
            )
        )
    return rows


def _model_report(name: str, support: str, rows: list[dict]) -> dict:
    return {
        "name": name,
        "tool_support": support,
        "prompts": rows,
        "summary": {
            "prompts": len(rows),
            "tool_chosen_correctly": sum(1 for row in rows if row["tool_chosen_correctly"]),
            "arguments_valid": sum(1 for row in rows if row["arguments_valid"]),
            "task_completed": sum(1 for row in rows if row["task_completed"]),
        },
    }


def _server_summary(server: McpServerConfig, tools) -> dict:
    return {
        "name": server.name,
        "transport": server.transport,
        "enabled_in_config": server.enabled,
        "tools": [{"name": tool.name, "description": tool.description} for tool in tools],
    }


def _failed(config, texts, message: str, detail: str = "", server: McpServerConfig | None = None) -> dict:
    return {
        "tools_enabled": config.mcp.enabled,
        "server": None
        if server is None
        else {"name": server.name, "transport": server.transport, "enabled_in_config": server.enabled, "tools": []},
        "note": texts["mcp_tools_off"] if server is None else "",
        "error": message,
        "detail": detail,
        "models": [],
    }


async def _unused(name, arguments):
    raise McpError("tool_refused")


if __name__ == "__main__":
    raise SystemExit(main())
