"""OpenAI-compatible chat client. Streams visible content only."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass

import httpx

from app.config import ModelConfig


class AdapterError(Exception):
    """The model server could not complete the stream."""

    def __init__(self, message: str, code: str = "model_error"):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ParsedLine:
    done: bool = False
    delta: str | None = None


def parse_sse_line(line: str) -> ParsedLine | None:
    """Parse one server-sent line.

    Returns None when the line carries nothing to show. Reasoning fields are
    ignored so they never enter the transcript.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith(":"):
        return None
    if not stripped.startswith("data:"):
        return None
    data = stripped[len("data:") :].strip()
    if data == "[DONE]":
        return ParsedLine(done=True)
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise AdapterError("the model stream sent invalid JSON") from exc
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if error:
        if isinstance(error, dict):
            error = error.get("message") or error
        raise AdapterError(str(error))
    return _content_line(payload)


def _content_line(payload: dict) -> ParsedLine | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, dict):
        return None
    delta = choice.get("delta") or {}
    if not isinstance(delta, dict):
        return None
    content = delta.get("content")
    if not isinstance(content, str) or content == "":
        return None
    return ParsedLine(delta=content)


def _plain_model_error(body: str, status: int) -> str:
    text = " ".join(body.split())
    if not text or text.startswith("{") or text.startswith("[") or len(text) > 180:
        return f"The model request failed ({status})."
    return text


def chat_payload(model: ModelConfig, messages: Iterable[dict[str, str]]) -> dict:
    """Body for an OpenAI-compatible chat completion.

    max_tokens keeps the reply inside the reserved output budget.
    options.num_ctx asks Ollama to use the same context size that budget used.
    """
    return {
        "model": model.name,
        "messages": list(messages),
        "stream": True,
        "temperature": model.temperature,
        "max_tokens": model.reserved_output_tokens,
        "options": {"num_ctx": model.context_size},
    }


def direct_client(timeout: httpx.Timeout) -> httpx.AsyncClient:
    """HTTP client that does not consult proxy environment variables."""
    return httpx.AsyncClient(timeout=timeout, trust_env=False)


class ChatAdapter:
    def __init__(self, model: ModelConfig):
        self.model = model

    async def stream(self, messages: Iterable[dict[str, str]]) -> AsyncIterator[str]:
        payload = chat_payload(self.model, messages)
        url = f"{self.model.base_url}/chat/completions"
        timeout = httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=5.0)
        try:
            async with direct_client(timeout) as client:
                async for delta in _iter_deltas(client, url, payload):
                    yield delta
        except httpx.HTTPError as exc:
            raise AdapterError(str(exc), code="unreachable") from exc


async def probe_model(model: ModelConfig) -> str:
    """Return ready when the configured model is listed, otherwise offline."""
    url = f"{model.base_url}/models"
    timeout = httpx.Timeout(connect=1.5, read=1.5, write=1.5, pool=1.5)
    try:
        async with direct_client(timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError):
        return "offline"
    if not isinstance(payload, dict):
        return "offline"
    names = set()
    for item in payload.get("data") or []:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            names.add(item["id"])
    return "ready" if model.name in names else "offline"


async def _iter_deltas(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
) -> AsyncIterator[str]:
    async with client.stream("POST", url, json=payload) as response:
        if response.status_code != 200:
            body = (await response.aread()).decode(errors="replace")[:500]
            raise AdapterError(_plain_model_error(body, response.status_code))
        async for line in response.aiter_lines():
            parsed = parse_sse_line(line)
            if parsed is None:
                continue
            if parsed.done:
                return
            if parsed.delta:
                yield parsed.delta
