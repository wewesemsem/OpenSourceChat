"""Open one configured MCP server and call an allowlisted tool.

The SDK owns the protocol. This module only chooses the transport and applies
the auth header. A stdio server is launched with the SDK's own minimal
environment, plus any variable names listed for that server.
"""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx2
from mcp import Client, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from app.config import ROOT, McpConfig, McpServerConfig, check_mcp_url
from app.mcp.errors import McpError, redact


@dataclass(frozen=True)
class ListedTool:
    name: str
    description: str
    parameters: dict


@dataclass(frozen=True)
class ToolAnswer:
    text: str
    ok: bool


def stdio_parameters(server: McpServerConfig) -> StdioServerParameters:
    """Build the subprocess description. `python` means this interpreter."""
    command = server.command
    if command in {"python", "python3"}:
        command = sys.executable
    return StdioServerParameters(
        command=command,
        args=list(server.args),
        env=_stdio_env(server.env),
        cwd=server.cwd or str(ROOT),
    )


def auth_headers(server: McpServerConfig) -> dict[str, str]:
    token = _token(server)
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def openai_tool(tool: ListedTool) -> dict:
    """Model-neutral tool list, shaped for the OpenAI-compatible request."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters or {"type": "object", "properties": {}},
        },
    }


def select_server(settings: McpConfig, name: str) -> McpServerConfig | None:
    """The named server, or the first enabled one. None means tools stay off."""
    if name:
        try:
            return settings.server(name)
        except KeyError as exc:
            raise McpError("connect_failed", detail=name) from exc
    enabled = [item for item in settings.servers if item.enabled]
    if not enabled:
        return None
    return enabled[0]


@asynccontextmanager
async def open_session(
    server: McpServerConfig,
    settings: McpConfig,
    *,
    http_auth=None,
) -> AsyncIterator[Client]:
    secret = _token(server)
    if server.auth.type == "oauth" and http_auth is None:
        raise McpError("sign_in_required")
    entered = False
    try:
        async with _connected(server, settings, secret, http_auth) as client:
            entered = True
            yield client
    except McpError:
        raise
    except Exception as exc:
        if entered:
            raise
        raise McpError("connect_failed", detail=redact(str(exc), secret)) from exc


async def list_allowed_tools(client: Client, server: McpServerConfig) -> list[ListedTool]:
    listed = await client.list_tools()
    allowed = set(server.allowed_tools)
    tools = []
    for tool in listed.tools:
        if tool.name not in allowed:
            continue
        schema = tool.input_schema if isinstance(tool.input_schema, dict) else {}
        tools.append(
            ListedTool(
                name=tool.name,
                description=tool.description or "",
                parameters=schema,
            )
        )
    return tools


async def call_allowed_tool(
    client: Client,
    server: McpServerConfig,
    name: str,
    arguments: Mapping[str, object],
    *,
    read_timeout_seconds: float,
) -> ToolAnswer:
    if name not in server.allowed_tools:
        raise McpError("tool_refused")
    result = await client.call_tool(
        name,
        dict(arguments),
        read_timeout_seconds=read_timeout_seconds,
    )
    parts = []
    for block in result.content or []:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            parts.append(text)
    return ToolAnswer(text="\n".join(parts), ok=not bool(result.is_error))


@asynccontextmanager
async def _connected(
    server: McpServerConfig,
    settings: McpConfig,
    secret: str,
    http_auth,
) -> AsyncIterator[Client]:
    headers = {"Authorization": f"Bearer {secret}"} if secret else None
    if server.transport == "stdio":
        async with Client(stdio_client(stdio_parameters(server))) as client:
            yield client
        return
    try:
        url = check_mcp_url(server.url, settings.allowed_hosts)
    except ValueError as exc:
        text = redact(str(exc), secret)
        if "HTTPS" in text:
            code = "https_required"
        elif "allowed_hosts" in text:
            code = "host_rejected"
        else:
            code = "connect_failed"
        raise McpError(code, detail=text) from exc
    if server.transport == "sse":
        transport = sse_client(
            url,
            headers=headers,
            timeout=settings.connect_timeout_seconds,
            sse_read_timeout=settings.read_timeout_seconds,
            httpx_client_factory=_sse_factory,
        )
        async with Client(transport) as client:
            yield client
        return
    if server.transport == "streamable_http":
        timeout = httpx2.Timeout(settings.connect_timeout_seconds, read=settings.read_timeout_seconds)
        async with httpx2.AsyncClient(
            headers=headers,
            auth=http_auth,
            timeout=timeout,
            trust_env=False,
        ) as http_client:
            async with Client(streamable_http_client(url, http_client=http_client)) as client:
                yield client
        return
    raise McpError("connect_failed", detail=server.transport)


def _sse_factory(headers=None, timeout=None, auth=None):
    """SSE client that does not pick up a proxy from the environment."""
    kwargs = {"trust_env": False}
    if headers is not None:
        kwargs["headers"] = headers
    if timeout is not None:
        kwargs["timeout"] = timeout
    if auth is not None:
        kwargs["auth"] = auth
    return httpx2.AsyncClient(**kwargs)


def _token(server: McpServerConfig) -> str:
    if server.auth.type != "bearer":
        return ""
    value = os.environ.get(server.auth.token_env, "")
    if not isinstance(value, str) or not value.strip():
        raise McpError("token_missing", name=server.auth.token_env)
    return value


def _stdio_env(names: tuple[str, ...]) -> dict[str, str] | None:
    if not names:
        return None
    extra = {}
    for name in names:
        if name in os.environ:
            extra[name] = os.environ[name]
    return extra
