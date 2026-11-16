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
        self.assertEqual(config.ui.footer, "Runs locally. Your chats stay on this machine.")
        self.assertEqual(config.data_dir, (ROOT / "data").resolve())
        self.assertFalse(config.security.cors_enabled)
        self.assertFalse(config.security.allow_remote_models)
        self.assertEqual(config.security.csrf_header, "X-Local-Chat")
        self.assertIn("http://127.0.0.1:8000", config.security.allowed_origins)
        self.assertEqual(config.files.allowed_folders, ())
        self.assertTrue(config.files.allow_upload)

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

    def test_shipped_config_binds_loopback(self):
        config = load_config()
        self.assertEqual(config.host, "127.0.0.1")
        self.assertNotEqual(config.host, "0.0.0.0")
        self.assertEqual(config.default_model, "gemma3:12b")
        self.assertEqual([item.name for item in config.models], ["gemma3:12b", "llama3.1:8b"])
        self.assertTrue(config.model().base_url.startswith("http://127.0.0.1:"))
        self.assertFalse(config.security.cors_enabled)
        self.assertFalse(config.security.allow_remote_models)
        self.assertIn("127.0.0.1", config.security.allowed_hosts)
        self.assertEqual(config.files.allowed_folders, ())
        self.assertTrue(config.files.allow_upload)
        self.assertEqual(config.files.allowed_extensions, ("txt", "md", "pdf", "docx", "csv"))

    def _write(self, text: str) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "config.yaml"
        path.write_text(dedent(text), encoding="utf-8")
        return path
