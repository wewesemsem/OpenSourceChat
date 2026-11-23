"""Config defaults: localhost bind, localhost model URLs."""

import tempfile
import unittest
from pathlib import Path
from textwrap import dedent

from app.config import ROOT, load_config, require_api_base, require_localhost_url


class LocalUrlTests(unittest.TestCase):
    def test_accepts_localhost_names(self):
        self.assertEqual(
            require_localhost_url("http://127.0.0.1:11434/v1/"),
            "http://127.0.0.1:11434/v1",
        )
        self.assertEqual(
            require_localhost_url("http://localhost:11434/v1"),
            "http://localhost:11434/v1",
        )
        self.assertEqual(
            require_localhost_url("http://[::1]:11434/v1"),
            "http://[::1]:11434/v1",
        )

    def test_rejects_remote_and_credentialed_urls(self):
        for url in (
            "http://example.com/v1",
            "http://127.0.0.1@example.com/v1",
            "http://user:pass@127.0.0.1:11434/v1",
            "ftp://127.0.0.1/v1",
        ):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    require_localhost_url(url)


class LoadConfigTests(unittest.TestCase):
    def test_omitted_host_defaults_to_loopback(self):
        path = self._write(
            """
            models:
              - name: gemma3:12b
                base_url: http://127.0.0.1:11434/v1
            """
        )
        config = load_config(path)
        self.assertEqual(config.host, "127.0.0.1")
        self.assertEqual(config.port, 8000)
        self.assertEqual(config.default_model, "gemma3:12b")
        self.assertEqual(config.model().context_size, 8192)
        self.assertEqual(config.model().label, "gemma3:12b")
        self.assertEqual(config.model().keep_alive, "30m")
        self.assertFalse(config.model().think)
        self.assertFalse(config.model().fast)
        self.assertEqual(config.ui.footer, "Runs locally. Your chats stay on this machine.")
        self.assertEqual(config.data_dir, (ROOT / "data").resolve())
        self.assertFalse(config.security.cors_enabled)
        self.assertFalse(config.security.allow_remote_models)
        self.assertEqual(config.security.csrf_header, "X-Local-Chat")
        self.assertIn("http://127.0.0.1:8000", config.security.allowed_origins)
        self.assertEqual(config.files.allowed_folders, ())
        self.assertTrue(config.files.allow_upload)
        self.assertFalse(config.index.enabled)
        self.assertEqual(config.index.retrieval_mode, "keyword")
        self.assertEqual(config.index.chunk_tokens, 400)
        self.assertEqual(config.index.chunk_overlap_tokens, 40)
        self.assertEqual(config.index.top_k, 4)
        self.assertEqual(config.index.min_score, 0)
        self.assertEqual(config.index.max_files, 2000)
        self.assertEqual(config.index.followup_query_mode, "heuristic")
        self.assertIsNone(config.index.embedding_model)
        self.assertFalse(config.mcp.enabled)
        self.assertEqual(config.mcp.max_steps, 3)
        self.assertEqual(config.mcp.servers, ())

    def test_explicit_bind_host_is_kept(self):
        path = self._write(
            """
            server:
              host: 0.0.0.0
              port: 9000
            models:
              - name: gemma3:12b
                base_url: http://localhost:11434/v1
            """
        )
        config = load_config(path)
        self.assertEqual(config.host, "0.0.0.0")
        self.assertEqual(config.port, 9000)

    def test_relative_data_dir_cannot_leave_the_project(self):
        path = self._write(
            """
            data_dir: ../outside
            models:
              - name: gemma3:12b
                base_url: http://127.0.0.1:11434/v1
            """
        )
        with self.assertRaises(ValueError):
            load_config(path)

    def test_api_base_stays_local(self):
        self.assertEqual(require_api_base(""), "")
        self.assertEqual(require_api_base("/chat"), "/chat")
        with self.assertRaises(ValueError):
            require_api_base("https://example.com")

    def test_rejects_remote_model_url(self):
        path = self._write(
            """
            models:
              - name: remote
                base_url: http://example.com/v1
            """
        )
        with self.assertRaises(ValueError):
            load_config(path)

    def test_allows_a_remote_model_only_when_configured(self):
        path = self._write(
            """
            security:
              allow_remote_models: true
            models:
              - name: remote
                base_url: https://example.com/v1
            """
        )
        config = load_config(path)
        self.assertEqual(config.model().base_url, "https://example.com/v1")

    def test_rejects_credentials_even_when_remote_models_are_allowed(self):
        path = self._write(
            """
            security:
              allow_remote_models: true
            models:
              - name: remote
                base_url: https://user:secret@example.com/v1
            """
        )
        with self.assertRaises(ValueError):
            load_config(path)

    def test_rejects_cors_with_no_origins(self):
        path = self._write(
            """
            security:
              cors_enabled: true
              cors_origins: []
            models:
              - name: gemma3:12b
                base_url: http://127.0.0.1:11434/v1
            """
        )
        with self.assertRaises(ValueError):
            load_config(path)

    def test_rejects_a_relative_allowed_folder(self):
        path = self._write(
            """
            files:
              allowed_folders:
                - docs
            models:
              - name: gemma3:12b
                base_url: http://127.0.0.1:11434/v1
            """
        )
        with self.assertRaises(ValueError):
            load_config(path)

    def test_rejects_a_wildcard_origin(self):
        path = self._write(
            """
            security:
              cors_enabled: true
              cors_origins:
                - "*"
            models:
              - name: gemma3:12b
                base_url: http://127.0.0.1:11434/v1
            """
        )
        with self.assertRaises(ValueError):
            load_config(path)

    def test_keep_alive_can_be_forever(self):
        path = self._write(
            """
            keep_alive: -1
            models:
              - name: gemma3:12b
                base_url: http://127.0.0.1:11434/v1
                think: true
            """
        )
        config = load_config(path)
        self.assertEqual(config.keep_alive, -1)
        self.assertEqual(config.model().keep_alive, -1)
        self.assertTrue(config.model().think)

    def test_shipped_config_binds_loopback(self):
        config = load_config()
        self.assertEqual(config.host, "127.0.0.1")
        self.assertNotEqual(config.host, "0.0.0.0")
        self.assertEqual(config.default_model, "gemma3:12b")
        self.assertEqual(
            [item.name for item in config.models],
            ["gemma3:12b", "llama3.1:8b", "gemma3:4b"],
        )
        self.assertEqual(config.keep_alive, "30m")
        self.assertEqual(config.model().keep_alive, "30m")
        self.assertFalse(config.model().think)
        fast = config.model("gemma3:4b")
        self.assertTrue(fast.fast)
        self.assertEqual(fast.context_size, 4096)
        self.assertEqual(fast.reserved_output_tokens, 512)
        self.assertTrue(config.model().base_url.startswith("http://127.0.0.1:"))
        self.assertFalse(config.security.cors_enabled)
        self.assertFalse(config.security.allow_remote_models)
        self.assertIn("127.0.0.1", config.security.allowed_hosts)
        self.assertEqual(config.files.allowed_folders, ())
        self.assertTrue(config.files.allow_upload)
        self.assertEqual(config.files.allowed_extensions, ("txt", "md", "pdf", "docx", "csv"))
        self.assertFalse(config.index.enabled)
        self.assertEqual(config.index.retrieval_mode, "keyword")
        self.assertEqual(config.index.chunk_tokens, 400)
        self.assertEqual(config.index.chunk_overlap_tokens, 40)
        self.assertEqual(config.index.top_k, 4)
        self.assertEqual(config.index.min_score, 0.0)
        self.assertEqual(config.index.max_files, 2000)
        self.assertEqual(config.index.max_total_size_bytes, 200 * 1024 * 1024)
        self.assertEqual(config.index.max_chunks_per_folder, 20000)
        self.assertEqual(config.index.max_context_tokens_share, 0.25)
        self.assertEqual(config.index.index_throttle, 0)
        self.assertTrue(config.index.pause_indexing_during_chat)
        self.assertEqual(config.index.followup_query_mode, "heuristic")
        self.assertIsNone(config.index.embedding_model)
        self.assertFalse(config.mcp.enabled)
        self.assertEqual(config.mcp.max_steps, 3)
        self.assertEqual([item.name for item in config.mcp.servers], ["company", "spike"])
        self.assertTrue(all(not item.enabled for item in config.mcp.servers))
        self.assertEqual(config.mcp.server("company").transport, "sse")
        self.assertEqual(config.mcp.server("company").auth.type, "bearer")
        self.assertEqual(config.mcp.server("company").auth.token_env, "MCP_COMPANY_TOKEN")
        self.assertEqual(len(config.mcp.spike.prompts), 20)
        self.assertEqual([item.id for item in config.mcp.connectors], ["atlassian", "microsoft365"])
        atlassian = config.mcp.connector("atlassian")
        self.assertEqual(atlassian.url, "https://mcp.atlassian.com/v2/mcp")
        self.assertEqual(atlassian.service_keys, ("mcp_jira", "mcp_confluence"))
        self.assertEqual(atlassian.client_id_env, "")
        microsoft = config.mcp.connector("microsoft365")
        self.assertEqual(microsoft.url, "https://workiq.svc.cloud.microsoft/mcp")
        self.assertEqual(
            microsoft.service_keys,
            ("mcp_outlook", "mcp_teams", "mcp_sharepoint", "mcp_onedrive"),
        )
        self.assertEqual(microsoft.client_id_env, "MCP_MICROSOFT365_CLIENT_ID")
        self.assertEqual(microsoft.client_secret_env, "MCP_MICROSOFT365_CLIENT_SECRET")
        self.assertIn("mcp.atlassian.com", config.mcp.allowed_hosts)
        self.assertIn("workiq.svc.cloud.microsoft", config.mcp.allowed_hosts)

    def test_rejects_a_secret_written_into_auth(self):
        path = self._write(
            """
            mcp:
              servers:
                - name: company
                  transport: sse
                  url: https://tools.example/mcp
                  auth:
                    type: bearer
                    token: secret-value
                  enabled: true
            models:
              - name: gemma3:12b
                base_url: http://127.0.0.1:11434/v1
            """
        )
        with self.assertRaises(ValueError) as caught:
            load_config(path)
        self.assertNotIn("secret-value", str(caught.exception))

    def test_connector_rejects_a_secret(self):
        path = self._write(
            """
            mcp:
              allowed_hosts:
                - mcp.atlassian.com
              connectors:
                - id: atlassian
                  transport: streamable_http
                  url: https://mcp.atlassian.com/v2/mcp
                  title_key: mcp_atlassian
                  summary_key: mcp_atlassian_summary
                  service_keys:
                    - mcp_jira
                  client_secret: secret-value
            models:
              - name: gemma3:12b
                base_url: http://127.0.0.1:11434/v1
            """
        )
        with self.assertRaises(ValueError) as caught:
            load_config(path)
        self.assertNotIn("secret-value", str(caught.exception))

    def test_rejects_plain_http_for_a_remote_mcp_server(self):
        path = self._write(
            """
            mcp:
              allowed_hosts:
                - tools.example
              servers:
                - name: company
                  transport: sse
                  url: http://tools.example/mcp
                  enabled: true
            models:
              - name: gemma3:12b
                base_url: http://127.0.0.1:11434/v1
            """
        )
        with self.assertRaises(ValueError):
            load_config(path)

    def test_rejects_an_mcp_host_that_is_not_allowed(self):
        path = self._write(
            """
            mcp:
              servers:
                - name: company
                  transport: streamable_http
                  url: https://tools.example/mcp
                  enabled: true
            models:
              - name: gemma3:12b
                base_url: http://127.0.0.1:11434/v1
            """
        )
        with self.assertRaises(ValueError):
            load_config(path)

    def test_accepts_https_when_the_host_is_allowed(self):
        path = self._write(
            """
            mcp:
              allowed_hosts:
                - tools.example
              servers:
                - name: company
                  transport: streamable_http
                  url: https://tools.example/mcp
                  auth:
                    type: bearer
                    token_env: MCP_COMPANY_TOKEN
                  enabled: true
                  allowed_tools:
                    - lookup
            models:
              - name: gemma3:12b
                base_url: http://127.0.0.1:11434/v1
            """
        )
        config = load_config(path)
        self.assertTrue(config.mcp.enabled)
        self.assertEqual(config.mcp.server("company").url, "https://tools.example/mcp")

    def test_requires_an_explicit_transport(self):
        path = self._write(
            """
            mcp:
              servers:
                - name: company
                  url: http://127.0.0.1:9/mcp
                  enabled: false
            models:
              - name: gemma3:12b
                base_url: http://127.0.0.1:11434/v1
            """
        )
        with self.assertRaises(ValueError):
            load_config(path)

    def test_index_defaults_when_the_block_is_partial(self):
        path = self._write(
            """
            index:
              enabled: true
            models:
              - name: gemma3:4b
                base_url: http://127.0.0.1:11434/v1
            """
        )
        config = load_config(path)
        self.assertTrue(config.index.enabled)
        self.assertEqual(config.index.chunk_tokens, 400)
        self.assertEqual(config.index.retrieval_mode, "keyword")
        self.assertIsNone(config.index.embedding_model)

    def test_rejects_an_unknown_index_mode_and_embedding_model(self):
        path = self._write(
            """
            index:
              retrieval_mode: semantic
            models:
              - name: gemma3:4b
                base_url: http://127.0.0.1:11434/v1
            """
        )
        with self.assertRaises(ValueError):
            load_config(path)
        path = self._write(
            """
            index:
              chunk_overlap_tokens: 400
            models:
              - name: gemma3:4b
                base_url: http://127.0.0.1:11434/v1
            """
        )
        with self.assertRaises(ValueError):
            load_config(path)
        path = self._write(
            """
            index:
              embedding_model: missing-model
            models:
              - name: gemma3:4b
                base_url: http://127.0.0.1:11434/v1
            """
        )
        with self.assertRaises(ValueError):
            load_config(path)

    def _write(self, text: str) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "config.yaml"
        path.write_text(dedent(text), encoding="utf-8")
        return path
