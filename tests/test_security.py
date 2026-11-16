"""Host, origin, CSRF, auth hook, and security headers."""

import unittest
from dataclasses import replace

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.testclient import TestClient
from starlette.requests import Request

from app.config import load_config
from app.main import app, config, strings
from app.security import GuardMiddleware, cors_kwargs, guard_request, warn_if_public_bind


def _request(method: str, path: str, headers: dict[str, str] | None = None) -> Request:
    raw = []
    for key, value in (headers or {}).items():
        raw.append((key.lower().encode(), value.encode()))
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": raw,
            "client": ("127.0.0.1", 40000),
            "server": ("127.0.0.1", 8000),
        }
    )


def _messages() -> dict[str, str]:
    return {
        "host_rejected": strings["host_rejected"],
        "origin_rejected": strings["origin_rejected"],
        "csrf_rejected": strings["csrf_rejected"],
        "auth_rejected": strings["auth_rejected"],
    }


class GuardTests(unittest.TestCase):
    def test_rejects_a_rebound_host_and_a_foreign_origin(self):
        security = config.security
        rebound = guard_request(
            _request("GET", "/api/meta", {"host": "evil.example:8000"}),
            security,
            _messages(),
            config.port,
        )
        self.assertEqual(rebound.status, 403)
        self.assertEqual(rebound.message, strings["host_rejected"])
        foreign = guard_request(
            _request(
                "POST",
                "/api/chat",
                {
                    "host": "127.0.0.1:8000",
                    "origin": "http://evil.example",
                    security.csrf_header: security.csrf_token,
                },
            ),
            security,
            _messages(),
            config.port,
        )
        self.assertEqual(foreign.status, 403)
        self.assertEqual(foreign.message, strings["origin_rejected"])

    def test_requires_the_csrf_header_on_writes(self):
        security = config.security
        missing = guard_request(
            _request("DELETE", "/api/conversations/1", {"host": "127.0.0.1:8000"}),
            security,
            _messages(),
            config.port,
        )
        self.assertEqual(missing.message, strings["csrf_rejected"])
        allowed = guard_request(
            _request(
                "POST",
                "/api/chat",
                {"host": "localhost:8000", security.csrf_header: security.csrf_token},
            ),
            security,
            _messages(),
            config.port,
        )
        self.assertIsNone(allowed)

    def test_missing_origin_is_allowed_and_a_null_origin_is_not(self):
        security = config.security
        self.assertIsNone(
            guard_request(
                _request("GET", "/api/meta", {"host": "127.0.0.1:8000"}),
                security,
                _messages(),
                config.port,
            )
        )
        null_origin = guard_request(
            _request("GET", "/api/meta", {"host": "127.0.0.1:8000", "origin": "null"}),
            security,
            _messages(),
            config.port,
        )
        self.assertEqual(null_origin.message, strings["origin_rejected"])

    def test_auth_token_is_a_no_op_until_configured(self):
        open_page = guard_request(
            _request("GET", "/api/meta", {"host": "127.0.0.1:8000"}),
            config.security,
            _messages(),
            config.port,
        )
        self.assertIsNone(open_page)
        locked = replace(config.security, auth_token="secret")
        page = guard_request(
            _request("GET", "/", {"host": "127.0.0.1:8000"}),
            locked,
            _messages(),
            config.port,
        )
        self.assertIsNone(page)
        denied = guard_request(
            _request("GET", "/api/meta", {"host": "127.0.0.1:8000"}),
            locked,
            _messages(),
            config.port,
        )
        self.assertEqual(denied.status, 401)
        self.assertEqual(denied.message, strings["auth_rejected"])
        accepted = guard_request(
            _request(
                "GET",
                "/api/meta",
                {"host": "127.0.0.1:8000", "authorization": "Bearer secret"},
            ),
            locked,
            _messages(),
            config.port,
        )
        self.assertIsNone(accepted)

    def test_warns_only_when_the_bind_address_is_not_loopback(self):
        with self.assertNoLogs("app.security", level="WARNING"):
            warn_if_public_bind("127.0.0.1", strings["bind_warning"])
        with self.assertLogs("app.security", level="WARNING") as captured:
            warn_if_public_bind("0.0.0.0", strings["bind_warning"])
        self.assertIn("0.0.0.0", captured.output[0])
        self.assertIn("127.0.0.1", captured.output[0])


class AppGuardTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app, base_url=f"http://127.0.0.1:{config.port}")

    def test_sets_security_headers_and_hides_the_token_slot(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("default-src 'self'", response.headers["content-security-policy"])
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(response.headers["referrer-policy"], "no-referrer")
        self.assertEqual(response.headers["x-frame-options"], "DENY")
        self.assertIn('name="csrf-header"', response.text)
        self.assertIn('name="auth-token" content=""', response.text)
        self.assertNotIn("access-control-allow-origin", response.headers)

    def test_rejects_a_write_without_the_csrf_header(self):
        response = self.client.post("/api/chat", json={"content": "Hi"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"], strings["csrf_rejected"])
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")

    def test_rejects_a_foreign_origin_on_the_live_app(self):
        response = self.client.get("/api/meta", headers={"Origin": "http://evil.example"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"], strings["origin_rejected"])

    def test_static_assets_carry_the_same_headers(self):
        response = self.client.get("/static/app.js")
        self.assertEqual(response.status_code, 200)
        self.assertIn("default-src 'self'", response.headers["content-security-policy"])


class CorsTests(unittest.TestCase):
    def test_lists_one_origin_and_rejects_the_rest(self):
        security = replace(
            load_config().security,
            cors_enabled=True,
            cors_origins=("http://allowed.example",),
            allowed_origins=("http://allowed.example",),
        )
        guarded = FastAPI()

        @guarded.get("/ping")
        def ping() -> dict:
            return {"ok": True}

        guarded.add_middleware(CORSMiddleware, **cors_kwargs(security))
        guarded.add_middleware(
            GuardMiddleware,
            security=security,
            messages=_messages(),
            bound_port=8000,
        )
        client = TestClient(guarded, base_url="http://127.0.0.1:8000")
        allowed = client.get("/ping", headers={"Origin": "http://allowed.example"})
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(allowed.headers["access-control-allow-origin"], "http://allowed.example")
        blocked = client.get("/ping", headers={"Origin": "https://other.example"})
        self.assertEqual(blocked.status_code, 403)
        self.assertNotIn("access-control-allow-origin", blocked.headers)

    def test_auth_middleware_lets_the_page_load_and_locks_the_api(self):
        security = replace(load_config().security, auth_token="secret")
        guarded = FastAPI()

        @guarded.get("/")
        def index() -> HTMLResponse:
            return HTMLResponse("page")

        @guarded.get("/api/meta")
        def meta() -> dict:
            return {"ok": True}

        guarded.add_middleware(
            GuardMiddleware,
            security=security,
            messages=_messages(),
            bound_port=8000,
        )
        client = TestClient(guarded, base_url="http://127.0.0.1:8000")
        self.assertEqual(client.get("/").status_code, 200)
        self.assertEqual(client.get("/api/meta").status_code, 401)
        allowed = client.get("/api/meta", headers={"Authorization": "Bearer secret"})
        self.assertEqual(allowed.status_code, 200)
