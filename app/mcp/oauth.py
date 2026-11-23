"""OAuth for the official connectors.

Tokens and a dynamically registered client stay in the data directory.
A client secret that comes from the environment is never written there.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from mcp.client.auth import AuthorizationCodeResult, OAuthClientProvider, OAuthFlowError, TokenStorage
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

from app.config import AppConfig, McpConnector
from app.mcp.connect import open_session

TOKEN_DIR = "mcp-oauth"


@dataclass
class _Flow:
    authorization_url: str = ""
    error: str = ""
    done: bool = False
    denied: bool = False
    state: str = ""
    future: asyncio.Future | None = None
    states: list[str] = field(default_factory=list)


_flows: dict[str, _Flow] = {}
_tasks: dict[str, asyncio.Task] = {}
_by_state: dict[str, _Flow] = {}


def redirect_uri(config: AppConfig) -> str:
    return f"http://{config.host}:{config.port}/api/mcp/oauth/callback"


def client_ready(connector: McpConnector) -> bool:
    """Atlassian registers itself. Microsoft needs both environment variables."""
    if not connector.client_id_env:
        return True
    return bool(_env(connector.client_id_env)) and bool(_env(connector.client_secret_env))


def client_notice(connector: McpConnector, config: AppConfig, texts: dict[str, str]) -> str:
    return (
        texts["mcp_client_missing"]
        .replace("{client_id}", connector.client_id_env)
        .replace("{client_secret}", connector.client_secret_env)
        .replace("{redirect}", redirect_uri(config))
    )


def has_token(data_dir: Path, name: str) -> bool:
    raw = _read(token_path(data_dir, name)).get("tokens") or {}
    return isinstance(raw, dict) and bool(raw.get("access_token"))


def clear_tokens(data_dir: Path, name: str) -> None:
    path = token_path(data_dir, name)
    data = _read(path)
    if not data:
        return
    data["tokens"] = None
    _write(path, data)


def flow_state(name: str) -> dict:
    flow = _flows.get(name)
    if flow is None:
        return {"pending": False, "authorization_url": "", "error": ""}
    return {
        "pending": not flow.done,
        "authorization_url": "" if flow.done else flow.authorization_url,
        "error": flow.error,
    }


def cancel_oauth(name: str) -> None:
    flow = _flows.pop(name, None)
    task = _tasks.pop(name, None)
    if flow is not None and flow.future is not None and not flow.future.done():
        flow.future.cancel()
    _drop_states(flow)
    if task is not None and not task.done():
        task.cancel()


def begin_oauth(config: AppConfig, connector: McpConnector, texts: dict[str, str], on_connected) -> None:
    """Start the sign-in handshake. A second click while one is running does nothing."""
    current = _flows.get(connector.id)
    if current is not None and not current.done:
        return
    flow = _Flow()
    _flows[connector.id] = flow
    _tasks[connector.id] = asyncio.create_task(
        _run(flow, config, connector, texts, on_connected),
        name=f"mcp-oauth-{connector.id}",
    )


def callback_html(params, texts: dict[str, str]) -> str:
    """Finish the browser redirect. The original tab keeps polling."""
    error = str(params.get("error") or "")
    state = str(params.get("state") or "")
    code = str(params.get("code") or "")
    iss = params.get("iss")
    flow = _by_state.get(state)
    if flow is None or flow.future is None or flow.future.done():
        return _page(texts["mcp_oauth_done"], texts["mcp_oauth_missing"])
    if error:
        flow.denied = error == "access_denied"
        flow.future.set_exception(OAuthFlowError("denied" if flow.denied else "failed"))
        if flow.denied:
            return _page(texts["mcp_oauth_done"], texts["mcp_oauth_denied"])
        return _page(texts["mcp_oauth_done"], texts["mcp_oauth_failed"])
    if not code:
        return _page(texts["mcp_oauth_done"], texts["mcp_oauth_missing"])
    issuer = str(iss) if isinstance(iss, str) and iss else None
    flow.future.set_result(AuthorizationCodeResult(code=code, state=state, iss=issuer))
    return _page(texts["mcp_oauth_done"], texts["mcp_oauth_done_body"])


def token_path(data_dir: Path, name: str) -> Path:
    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError("name")
    return data_dir / TOKEN_DIR / f"{name}.json"


def build_provider(config: AppConfig, connector: McpConnector, *, redirect, callback) -> OAuthClientProvider:
    metadata = OAuthClientMetadata(
        client_name=config.ui.title,
        redirect_uris=[redirect_uri(config)],  # type: ignore[list-item]
        token_endpoint_auth_method="client_secret_post" if connector.client_secret_env else "none",
        grant_types=["authorization_code", "refresh_token"],
        scope=connector.scope or None,
    )
    return OAuthClientProvider(
        server_url=connector.url,
        client_metadata=metadata,
        storage=_FileStorage(token_path(config.data_dir, connector.id), connector),
        redirect_handler=redirect,
        callback_handler=callback,
    )


class _FileStorage(TokenStorage):
    def __init__(self, path: Path, connector: McpConnector):
        self.path = path
        self.connector = connector

    async def get_tokens(self) -> OAuthToken | None:
        raw = _read(self.path).get("tokens")
        if not isinstance(raw, dict) or not raw.get("access_token"):
            return None
        return OAuthToken.model_validate(raw)

    async def set_tokens(self, tokens: OAuthToken) -> None:
        data = _read(self.path)
        data["tokens"] = tokens.model_dump(mode="json")
        _write(self.path, data)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        preset = _preset_client(self.connector)
        if preset is not None:
            return preset
        raw = _read(self.path).get("client")
        if not isinstance(raw, dict) or not raw.get("client_id"):
            return None
        return OAuthClientInformationFull.model_validate(raw)

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        if _preset_client(self.connector) is not None:
            return
        data = _read(self.path)
        data["client"] = client_info.model_dump(mode="json")
        _write(self.path, data)


def _preset_client(connector: McpConnector) -> OAuthClientInformationFull | None:
    if not connector.client_id_env:
        return None
    client_id = _env(connector.client_id_env)
    secret = _env(connector.client_secret_env)
    if not client_id or not secret:
        return None
    return OAuthClientInformationFull(
        client_id=client_id,
        client_secret=secret,
        token_endpoint_auth_method="client_secret_post",
        grant_types=["authorization_code", "refresh_token"],
    )


async def _run(flow: _Flow, config: AppConfig, connector: McpConnector, texts: dict[str, str], on_connected) -> None:
    try:
        server = config.mcp.server(connector.id)
        flow.future = asyncio.get_running_loop().create_future()

        async def redirect(url: str) -> None:
            query = parse_qs(urlparse(url).query)
            state = (query.get("state") or [""])[0]
            flow.authorization_url = url
            if state:
                flow.state = state
                flow.states.append(state)
                _by_state[state] = flow

        async def callback() -> AuthorizationCodeResult:
            if flow.future is None:
                raise OAuthFlowError("failed")
            return await asyncio.wait_for(flow.future, config.mcp.oauth_timeout_seconds)

        provider = build_provider(config, connector, redirect=redirect, callback=callback)
        async with open_session(server, config.mcp, http_auth=provider) as client:
            await client.list_tools()
        on_connected()
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        flow.error = texts["mcp_oauth_timeout"]
    except Exception:
        flow.error = texts["mcp_oauth_denied"] if flow.denied else texts["mcp_oauth_failed"]
    finally:
        flow.done = True
        flow.authorization_url = ""
        _drop_states(flow)
        _tasks.pop(connector.id, None)


def _drop_states(flow: _Flow | None) -> None:
    if flow is None:
        return
    for state in flow.states:
        if _by_state.get(state) is flow:
            _by_state.pop(state, None)


def _env(name: str) -> str:
    value = os.environ.get(name, "")
    if not isinstance(value, str):
        return ""
    return value.strip()


def _read(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    os.chmod(path, 0o600)


def _page(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>"
        f"{escape(title)}</title></head><body><p>{escape(body)}</p></body></html>"
    )
