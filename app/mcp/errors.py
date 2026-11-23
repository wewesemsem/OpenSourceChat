"""MCP failures. The code is stable; the wording lives in strings.json."""

from __future__ import annotations


class McpError(Exception):
    def __init__(self, code: str, detail: str = "", **fields: str):
        super().__init__(code)
        self.code = code
        self.detail = detail
        self.fields = fields


def format_error(texts: dict[str, str], error: McpError) -> str:
    template = texts.get(f"mcp_{error.code}", error.code)
    text = template
    for key, value in error.fields.items():
        text = text.replace("{" + key + "}", value)
    return text


def redact(text: str, secret: str) -> str:
    if not secret or secret not in text:
        return text
    return text.replace(secret, "[redacted]")
