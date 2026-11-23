"""Customize panel: server list, save file, and connection status."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.config import load_config
from app.main import app, config
from app.mcp.catalog import connection_status, save_error, save_server_list
from app.mcp.oauth import _FileStorage, callback_html
from app.mcp.spike import load_strings


def _client() -> TestClient:
    return TestClient(
        app,
        base_url=f"http://127.0.0.1:{config.port}",
        headers={config.security.csrf_header: config.security.csrf_token},
    )


def _write_config(directory: Path) -> Path:
    path = directory / "config.yaml"
    path.write_text(
        dedent(
            f"""
            data_dir: {directory}
            models:
              - name: gemma3:12b
                base_url: http://127.0.0.1:11434/v1
            mcp:
              allowed_hosts:
                - 127.0.0.1
                - localhost
              servers:
                - name: company
                  transport: sse
                  url: ""
                  auth:
                    type: bearer
                    token_env: MCP_COMPANY_TOKEN
                  enabled: false
                  allowed_tools: []
            """
        ),
        encoding="utf-8",
    )
    return path


class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = _write_config(Path(self.tmp.name))

    def test_save_round_trips_and_approves_the_https_host(self):
        loaded = load_config(self.path)
        updated = save_server_list(
            loaded,
            [
                {
                    "name": "desk",
                    "transport": "streamable_http",
                    "url": "https://tools.example/mcp",
                    "auth_type": "bearer",
                    "token_env": "MCP_DESK_TOKEN",
                    "enabled": False,
                    "allowed_tools": ["lookup"],
                }
            ],
        )
        self.assertIn("tools.example", updated.mcp.allowed_hosts)
        again = load_config(self.path)
        saved = again.mcp.server("desk")
        self.assertEqual(saved.url, "https://tools.example/mcp")
        self.assertEqual(saved.auth.token_env, "MCP_DESK_TOKEN")
        self.assertFalse(saved.enabled)
        stored = (Path(self.tmp.name) / "mcp-servers.yaml").read_text(encoding="utf-8")
        self.assertNotIn("secret-value", stored)

    def test_rejects_a_secret_and_plain_http(self):
        loaded = load_config(self.path)
        texts = load_strings()
        with self.assertRaises(ValueError) as secret:
            save_server_list(
                loaded,
                [
                    {
                        "name": "desk",
                        "transport": "stdio",
                        "command": "echo",
                        "token": "secret-value",
                        "enabled": False,
                    }
                ],
            )
        self.assertNotIn("secret-value", save_error(secret.exception, texts))
        self.assertEqual(save_error(secret.exception, texts), texts["mcp_no_secret"])
        with self.assertRaises(ValueError):
            save_server_list(
                loaded,
                [
                    {
                        "name": "desk",
                        "transport": "sse",
                        "url": "http://tools.example/mcp",
                        "enabled": True,
                    }
                ],
            )

    def test_disabled_server_does_not_connect(self):
        loaded = load_config(self.path)

        def explode(*args, **kwargs):
            raise AssertionError("connected")

        with patch("app.mcp.catalog.open_session", explode):
            status, detail = asyncio.run(
                connection_status(loaded.mcp.server("company"), loaded, load_strings())
            )
        self.assertEqual(status, "disconnected")
        self.assertEqual(detail, "")


class ApiTests(unittest.TestCase):
    def test_lists_configured_servers_as_disconnected(self):
        payload = _client().get("/api/mcp/servers").json()
        by_name = {item["name"]: item for item in payload["servers"]}
        self.assertIn("company", by_name)
        self.assertIn("spike", by_name)
        self.assertEqual(by_name["company"]["status"], "disconnected")
        self.assertEqual(by_name["company"]["token_env"], "MCP_COMPANY_TOKEN")
        self.assertNotIn("Bearer", json.dumps(payload))

    def test_add_and_remove_use_the_data_directory(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        loaded = load_config(_write_config(Path(tmp.name)))
        body = {
            "name": "desk",
            "transport": "stdio",
            "command": "python",
            "args": ["-m", "app.mcp.spike_server"],
            "auth_type": "none",
            "enabled": False,
            "allowed_tools": ["office_lookup"],
        }
        with patch("app.main.config", loaded):
            client = _client()
            created = client.post("/api/mcp/servers", json=body)
            self.assertEqual(created.status_code, 200, created.text)
            listed = client.get("/api/mcp/servers").json()["servers"]
            self.assertIn("desk", [item["name"] for item in listed])
            removed = client.delete("/api/mcp/servers/desk")
            self.assertEqual(removed.status_code, 200, removed.text)
            names = [item["name"] for item in client.get("/api/mcp/servers").json()["servers"]]
            self.assertNotIn("desk", names)
        self.assertTrue((Path(tmp.name) / "mcp-servers.yaml").is_file())

    def test_catalog_lists_official_connectors_without_secrets(self):
        with patch.dict("os.environ", {"MCP_MICROSOFT365_CLIENT_ID": "", "MCP_MICROSOFT365_CLIENT_SECRET": ""}):
            payload = _client().get("/api/mcp/servers").json()
        by_id = {item["id"]: item for item in payload["connectors"]}
        self.assertEqual(by_id["atlassian"]["service_keys"], ["mcp_jira", "mcp_confluence"])
        self.assertEqual(
            by_id["microsoft365"]["service_keys"],
            ["mcp_outlook", "mcp_teams", "mcp_sharepoint", "mcp_onedrive"],
        )
        self.assertIn("MCP_MICROSOFT365_CLIENT_ID", by_id["microsoft365"]["notice"])
        self.assertIn("127.0.0.1", by_id["microsoft365"]["notice"])
        self.assertFalse(by_id["atlassian"]["installed"])
        self.assertFalse(by_id["microsoft365"]["signed_in"])
        self.assertNotIn("access_token", json.dumps(payload))

    def test_microsoft_connect_asks_for_the_app_registration(self):
        with patch.dict("os.environ", {"MCP_MICROSOFT365_CLIENT_ID": "", "MCP_MICROSOFT365_CLIENT_SECRET": ""}):
            response = _client().post("/api/mcp/connectors/microsoft365/connect")
        self.assertEqual(response.status_code, 400)
        self.assertIn("MCP_MICROSOFT365_CLIENT_SECRET", response.json()["detail"])
        self.assertNotIn("secret-value", response.text)

    def test_oauth_cannot_be_enabled_before_sign_in(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        loaded = load_config(_write_config(Path(tmp.name)))
        body = {
            "name": "atlassian",
            "transport": "streamable_http",
            "url": "https://mcp.atlassian.com/v2/mcp",
            "auth_type": "oauth",
            "enabled": True,
            "allowed_tools": [],
        }
        with self.assertRaises(ValueError) as caught:
            save_server_list(loaded, [body])
        self.assertEqual(save_error(caught.exception, load_strings()), load_strings()["mcp_sign_in_required"])
        self.assertFalse((Path(tmp.name) / "mcp-servers.yaml").exists())

    def test_callback_without_a_sign_in_is_a_page(self):
        response = _client().get("/api/mcp/oauth/callback")
        self.assertEqual(response.status_code, 200)
        self.assertIn(load_strings()["mcp_oauth_missing"], response.text)
        self.assertIn(load_strings()["mcp_oauth_missing"], callback_html({}, load_strings()))

    def test_environment_client_secret_is_not_written(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        connector = load_config().mcp.connector("microsoft365")
        path = Path(tmp.name) / "microsoft365.json"
        storage = _FileStorage(path, connector)
        with patch.dict(
            "os.environ",
            {"MCP_MICROSOFT365_CLIENT_ID": "app-id", "MCP_MICROSOFT365_CLIENT_SECRET": "secret-value"},
        ):
            info = asyncio.run(storage.get_client_info())
            self.assertEqual(info.client_id, "app-id")
            asyncio.run(storage.set_client_info(info))
        self.assertFalse(path.exists())
