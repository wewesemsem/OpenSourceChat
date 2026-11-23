"""List and save MCP servers for the Customize panel.

The shipped config.yaml stays untouched. Edits go to data/mcp-servers.yaml,
which load_config reads on the next start. A token is only the name of an
environment variable.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

import yaml

from app.config import LOCAL_HOSTS, AppConfig, McpConnector, McpServerConfig, replace_mcp_servers
from app.mcp.connect import open_session
from app.mcp.errors import McpError, format_error
from app.mcp.oauth import (
    build_provider,
    client_notice,
    client_ready,
    flow_state,
    has_token,
)

OVERLAY_NAME = "mcp-servers.yaml"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SECRET_KEYS = frozenset({"token", "secret", "password", "authorization"})


def public_server(server: McpServerConfig, *, status: str, detail: str = "") -> dict:
    return {
        "name": server.name,
        "transport": server.transport,
        "url": server.url,
        "command": server.command,
        "args": list(server.args),
        "auth_type": server.auth.type,
        "token_env": server.auth.token_env,
        "enabled": server.enabled,
        "allowed_tools": list(server.allowed_tools),
        "status": status,
        "detail": detail,
    }


async def connection_status(server: McpServerConfig, config: AppConfig, texts: dict[str, str]) -> tuple[str, str]:
    """Connected only after the existing client finishes its handshake."""
    if not server.enabled:
        return "disconnected", ""
    http_auth = None
    if server.auth.type == "oauth":
        if flow_state(server.name)["pending"]:
            return "disconnected", ""
        if not has_token(config.data_dir, server.name):
            return "disconnected", texts["mcp_sign_in_required"]
        try:
            connector = config.mcp.connector(server.name)
        except KeyError:
            return "disconnected", texts["mcp_sign_in_required"]
        http_auth = build_provider(
            config,
            connector,
            redirect=_refuse_redirect,
            callback=_refuse_callback,
        )
    try:
        async with open_session(server, config.mcp, http_auth=http_auth) as client:
            await client.list_tools()
    except McpError as exc:
        return "disconnected", format_error(texts, exc)
    return "connected", ""


async def _refuse_redirect(_url: str) -> None:
    raise McpError("sign_in_required")


async def _refuse_callback():
    raise McpError("sign_in_required")


def connector_views(config: AppConfig, texts: dict[str, str], listed: list[dict]) -> list[dict]:
    """Catalog entries plus the status already collected for the server list."""
    by_name = {item["name"]: item for item in listed}
    views = []
    for connector in config.mcp.connectors:
        server = by_name.get(connector.id)
        flow = flow_state(connector.id)
        status = server["status"] if server else "disconnected"
        detail = server["detail"] if server else ""
        if status != "connected" and flow["error"]:
            detail = flow["error"]
        elif flow["pending"]:
            detail = texts["mcp_sign_in_wait"]
        notice = ""
        if connector.client_id_env and not client_ready(connector):
            notice = client_notice(connector, config, texts)
        views.append({
            "id": connector.id,
            "title_key": connector.title_key,
            "summary_key": connector.summary_key,
            "service_keys": list(connector.service_keys),
            "url": connector.url,
            "installed": server is not None,
            "enabled": bool(server and server["enabled"]),
            "signed_in": has_token(config.data_dir, connector.id),
            "status": status,
            "detail": detail,
            "pending": flow["pending"],
            "authorization_url": flow["authorization_url"],
            "notice": notice,
            "auth_type": "oauth",
            "token_env": "",
            "transport": connector.transport,
            "command": "",
            "args": [],
            "allowed_tools": list(server["allowed_tools"]) if server else [],
        })
    return views


def install_connector(config: AppConfig, connector: McpConnector) -> AppConfig:
    """Add the official server, still off until sign-in finishes."""
    bodies = [public_server(item, status="disconnected") for item in config.mcp.servers]
    record = _connector_record(connector)
    for index, body in enumerate(bodies):
        if body["name"] != connector.id:
            continue
        record["allowed_tools"] = list(body["allowed_tools"])
        record["enabled"] = bool(body["enabled"] and has_token(config.data_dir, connector.id))
        bodies[index] = record
        break
    else:
        bodies.append(record)
    return save_server_list(config, bodies)


def disconnect_connector(config: AppConfig, name: str) -> AppConfig:
    from app.mcp.oauth import cancel_oauth, clear_tokens

    cancel_oauth(name)
    clear_tokens(config.data_dir, name)
    bodies = [
        public_server(item, status="disconnected")
        for item in config.mcp.servers
        if item.name != name
    ]
    if len(bodies) == len(config.mcp.servers):
        return config
    return save_server_list(config, bodies)


def _connector_record(connector: McpConnector) -> dict:
    return {
        "name": connector.id,
        "transport": connector.transport,
        "url": connector.url,
        "command": "",
        "args": [],
        "auth_type": "oauth",
        "token_env": "",
        "enabled": False,
        "allowed_tools": [],
    }


def save_error(exc: ValueError, texts: dict[str, str]) -> str:
    text = str(exc)
    if text == "secret" or "credential" in text.lower():
        return texts["mcp_no_secret"]
    if text == "sign_in":
        return texts["mcp_sign_in_required"]
    if text == "name":
        return texts["mcp_bad_name"]
    if text == "list":
        return texts["mcp_save_failed"]
    return text or texts["mcp_save_failed"]


def save_server_list(config: AppConfig, bodies: list[dict]) -> AppConfig:
    existing = {item.name: item for item in config.mcp.servers}
    raw_servers = [_raw_server(body, existing.get(str(body.get("name") or ""))) for body in bodies]
    _reject_unsigned_oauth(config, raw_servers)
    hosts = _hosts_for(raw_servers, config.mcp.allowed_hosts)
    updated = replace_mcp_servers(config, raw_servers, hosts)
    path = config.data_dir / OVERLAY_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "allowed_hosts": list(updated.mcp.allowed_hosts),
        "servers": [_stored_server(item) for item in updated.mcp.servers],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return updated


def _raw_server(body: dict, existing: McpServerConfig | None) -> dict:
    if _SECRET_KEYS & set(body):
        raise ValueError("secret")
    name = str(body.get("name") or "").strip()
    if not _NAME.match(name):
        raise ValueError("name")
    transport = str(body.get("transport") or "").strip()
    item = {
        "name": name,
        "transport": transport,
        "enabled": bool(body.get("enabled", False)),
        "allowed_tools": _strings(body.get("allowed_tools")),
        "auth": {"type": str(body.get("auth_type") or "none").strip() or "none"},
    }
    token_env = str(body.get("token_env") or "").strip()
    if token_env:
        item["auth"]["token_env"] = token_env
    if transport == "stdio":
        item["command"] = str(body.get("command") or "").strip()
        item["args"] = _strings(body.get("args"))
        if existing is not None and existing.transport == "stdio":
            if existing.cwd:
                item["cwd"] = existing.cwd
            if existing.env:
                item["env"] = list(existing.env)
    else:
        item["url"] = str(body.get("url") or "").strip()
    return item


def _reject_unsigned_oauth(config: AppConfig, servers: list[dict]) -> None:
    for item in servers:
        auth = item.get("auth") or {}
        if auth.get("type") == "oauth" and item.get("enabled") and not has_token(config.data_dir, str(item.get("name") or "")):
            raise ValueError("sign_in")


def _hosts_for(servers: list[dict], current: tuple[str, ...]) -> tuple[str, ...]:
    hosts = list(current)
    known = {item.lower() for item in hosts}
    for item in servers:
        url = str(item.get("url") or "")
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if not host or host in known:
            continue
        if parsed.scheme == "https" or (parsed.scheme == "http" and host in LOCAL_HOSTS):
            hosts.append(host)
            known.add(host)
    return tuple(hosts)


def _strings(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("list")
    items = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("list")
        text = item.strip()
        if text:
            items.append(text)
    return items


def _stored_server(server: McpServerConfig) -> dict:
    record = {
        "name": server.name,
        "transport": server.transport,
        "enabled": server.enabled,
        "allowed_tools": list(server.allowed_tools),
        "auth": {"type": server.auth.type},
    }
    if server.auth.token_env:
        record["auth"]["token_env"] = server.auth.token_env
    if server.transport == "stdio":
        record["command"] = server.command
        record["args"] = list(server.args)
        if server.cwd:
            record["cwd"] = server.cwd
        if server.env:
            record["env"] = list(server.env)
    else:
        record["url"] = server.url
    return record
