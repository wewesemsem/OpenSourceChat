"""Host, origin, CSRF, and optional token checks.

The custom request header is a fixed value from config, not a cookie session.
A browser will not attach that header to a cross-site form post, and a page on
another origin cannot read it. That is the CSRF defense for this app.
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from html import escape
from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import SecurityConfig

UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
LOG = logging.getLogger("app.security")


@dataclass(frozen=True)
class Rejection:
    status: int
    message: str


def warn_if_public_bind(host: str, template: str) -> None:
    """Log when the process is not bound to 127.0.0.1."""
    if host == "127.0.0.1":
        return
    LOG.warning(template.replace("{host}", host))


def render_index(html: str, security: SecurityConfig) -> str:
    """Insert the header names and the optional token into the page."""
    tags = (
        f'<meta name="csrf-header" content="{escape(security.csrf_header, quote=True)}">',
        f'<meta name="csrf-token" content="{escape(security.csrf_token, quote=True)}">',
        f'<meta name="auth-header" content="{escape(security.auth_header, quote=True)}">',
        f'<meta name="auth-scheme" content="{escape(security.auth_scheme, quote=True)}">',
        f'<meta name="auth-token" content="{escape(security.auth_token, quote=True)}">',
    )
    needle = '<meta charset="utf-8">'
    if needle not in html:
        raise ValueError("index.html is missing the charset meta")
    return html.replace(needle, needle + "".join(tags), 1)


def guard_request(
    request: Request,
    security: SecurityConfig,
    messages: dict[str, str],
    bound_port: int,
) -> Rejection | None:
    host_error = _check_host(request.headers.get("host"), security, messages, bound_port)
    if host_error:
        return host_error
    origin_error = _check_origin(request.headers.get("origin"), security, messages)
    if origin_error:
        return origin_error
    if request.method in UNSAFE_METHODS and not _same(
        request.headers.get(security.csrf_header, ""),
        security.csrf_token,
    ):
        return Rejection(403, messages["csrf_rejected"])
    if security.auth_token and not _public_page(request) and not _same(
        request.headers.get(security.auth_header, ""),
        _auth_value(security),
    ):
        return Rejection(401, messages["auth_rejected"])
    return None


def apply_security_headers(headers: MutableHeaders, security: SecurityConfig) -> None:
    headers["Content-Security-Policy"] = security.content_security_policy
    headers["X-Content-Type-Options"] = security.x_content_type_options
    headers["Referrer-Policy"] = security.referrer_policy
    headers["X-Frame-Options"] = security.x_frame_options


class GuardMiddleware:
    """Reject a bad Host, Origin, CSRF header, or auth token without buffering the body."""

    def __init__(self, app, security: SecurityConfig, messages: dict[str, str], bound_port: int):
        self.app = app
        self.security = security
        self.messages = messages
        self.bound_port = bound_port

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        rejection = guard_request(request, self.security, self.messages, self.bound_port)
        if rejection is not None:
            response = JSONResponse({"detail": rejection.message}, status_code=rejection.status)
            apply_security_headers(response.headers, self.security)
            await response(scope, receive, send)
            return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                apply_security_headers(MutableHeaders(scope=message), self.security)
            await send(message)

        await self.app(scope, receive, send_with_headers)


def _public_page(request: Request) -> bool:
    path = request.url.path
    return request.method == "GET" and (path == "/" or path.startswith("/static/"))


def _auth_value(security: SecurityConfig) -> str:
    if security.auth_scheme:
        return f"{security.auth_scheme} {security.auth_token}"
    return security.auth_token


def _check_host(
    raw: str | None,
    security: SecurityConfig,
    messages: dict[str, str],
    bound_port: int,
) -> Rejection | None:
    if not raw:
        return Rejection(403, messages["host_rejected"])
    try:
        hostname, port = _split_host(raw)
    except ValueError:
        return Rejection(403, messages["host_rejected"])
    if hostname not in {item.lower() for item in security.allowed_hosts}:
        return Rejection(403, messages["host_rejected"])
    if port is not None and port != bound_port:
        return Rejection(403, messages["host_rejected"])
    return None


def _check_origin(
    raw: str | None,
    security: SecurityConfig,
    messages: dict[str, str],
) -> Rejection | None:
    if raw is None:
        return None
    origin = raw.strip().rstrip("/").lower()
    allowed = {item.lower() for item in security.allowed_origins}
    if origin not in allowed:
        return Rejection(403, messages["origin_rejected"])
    return None


def _split_host(value: str) -> tuple[str, int | None]:
    text = value.strip()
    if text.startswith("["):
        end = text.find("]")
        if end <= 1:
            raise ValueError("bad host")
        hostname = text[1:end].lower()
        rest = text[end + 1 :]
        if not rest:
            return hostname, None
        if not rest.startswith(":"):
            raise ValueError("bad host")
        return hostname, int(rest[1:])
    if text.count(":") > 1:
        raise ValueError("bad host")
    if ":" not in text:
        return text.lower(), None
    hostname, port_text = text.rsplit(":", 1)
    if not hostname:
        raise ValueError("bad host")
    return hostname.lower(), int(port_text)


def _same(left: str, right: str) -> bool:
    a = left.encode()
    b = right.encode()
    if len(a) != len(b):
        return False
    return secrets.compare_digest(a, b)


def cors_kwargs(security: SecurityConfig) -> dict:
    """Origins and headers for CORSMiddleware. Wildcards are already rejected."""
    headers = list(dict.fromkeys([*security.cors_headers, security.csrf_header, security.auth_header]))
    return {
        "allow_origins": list(security.cors_origins),
        "allow_methods": list(security.cors_methods),
        "allow_headers": headers,
        "allow_credentials": security.cors_allow_credentials,
    }
