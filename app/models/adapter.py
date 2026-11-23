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
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class ParsedLine:
    done: bool = False
    delta: str | None = None
    usage: Usage | None = None


@dataclass
class StreamStats:
    """Token counts reported by the model server, when it sends them."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class ToolCall:
    """One tool request from a model, in a shape that is not tied to a vendor."""

    id: str
    name: str
    arguments: dict | None
    raw: str
    invalid: bool


@dataclass(frozen=True)
class ModelReply:
    content: str
    tool_calls: tuple[ToolCall, ...] = ()


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
    usage = _usage(payload)
    content = _content_line(payload)
    if content is None and usage is None:
        return None
    if content is None:
        return ParsedLine(usage=usage)
    return ParsedLine(done=content.done, delta=content.delta, usage=usage)


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


def chat_payload(
    model: ModelConfig,
    messages: Iterable[dict],
    *,
    tools: list[dict] | None = None,
    stream: bool = True,
) -> dict:
    """Body for an OpenAI-compatible chat completion.

    max_tokens keeps the reply inside the reserved output budget.
    options.num_ctx asks Ollama to allocate that context, not a larger default.
    think is false so a reasoning model does not spend the wait on hidden tokens.
    keep_alive stops Ollama from unloading the model after its 5 minute default.
    tools is omitted unless the caller has a list to send.
    """
    payload = {
        "model": model.name,
        "messages": list(messages),
        "stream": stream,
        "temperature": model.temperature,
        "max_tokens": model.reserved_output_tokens,
        "think": model.think,
        "keep_alive": model.keep_alive,
        "options": {"num_ctx": model.context_size},
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    if tools:
        payload["tools"] = list(tools)
    return payload


def generation_rate(tokens: int, seconds: float) -> float | None:
    """Tokens per second after the first token. A single burst has no rate."""
    if tokens <= 0 or seconds < 0.05:
        return None
    return round(tokens / seconds, 1)


def native_base(base_url: str) -> str:
    """Ollama's own API sits beside the OpenAI /v1 routes."""
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/v1"):
        return trimmed[: -len("/v1")]
    return trimmed


def same_model(reported: str, wanted: str) -> bool:
    """True when Ollama's loaded-model name is the configured model."""
    if reported == wanted:
        return True
    head, sep, _tail = reported.rpartition(":")
    return bool(sep) and head == wanted


def warmup_payload(model: ModelConfig) -> dict:
    """Load a model without asking it to write a reply."""
    return {"model": model.name, "keep_alive": model.keep_alive, "stream": False}


def direct_client(timeout: httpx.Timeout) -> httpx.AsyncClient:
    """HTTP client that does not consult proxy environment variables."""
    return httpx.AsyncClient(timeout=timeout, trust_env=False)


class ChatAdapter:
    def __init__(self, model: ModelConfig):
        self.model = model
        self.stats = StreamStats()

    async def stream(self, messages: Iterable[dict[str, str]]) -> AsyncIterator[str]:
        payload = chat_payload(self.model, messages)
        url = f"{self.model.base_url}/chat/completions"
        timeout = httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=5.0)
        try:
            async with direct_client(timeout) as client:
                async for parsed in _iter_events(client, url, payload):
                    _remember_usage(self.stats, parsed.usage)
                    if parsed.done:
                        return
                    if parsed.delta:
                        yield parsed.delta
        except httpx.HTTPError as exc:
            raise AdapterError(str(exc), code="unreachable") from exc

    async def complete(self, messages: Iterable[dict], tools: list[dict] | None = None) -> ModelReply:
        """One non-streaming reply, including any tool calls the model requested."""
        payload = chat_payload(self.model, messages, tools=tools, stream=False)
        url = f"{self.model.base_url}/chat/completions"
        timeout = httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=5.0)
        try:
            async with direct_client(timeout) as client:
                response = await client.post(url, json=payload)
        except httpx.HTTPError as exc:
            raise AdapterError(str(exc), code="unreachable") from exc
        if response.status_code != 200:
            raise AdapterError(_plain_model_error(response.text[:500], response.status_code))
        try:
            body = response.json()
        except ValueError as exc:
            raise AdapterError("the model returned an invalid response") from exc
        return parse_completion(body)


def parse_completion(payload: object) -> ModelReply:
    """Read a finished chat completion, including tool calls."""
    if not isinstance(payload, dict):
        raise AdapterError("the model returned an invalid response")
    error = payload.get("error")
    if error:
        if isinstance(error, dict):
            error = error.get("message") or error
        raise AdapterError(str(error))
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise AdapterError("the model returned an invalid response")
    message = choices[0].get("message") or {}
    if not isinstance(message, dict):
        raise AdapterError("the model returned an invalid response")
    content = message.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise AdapterError("the model returned an invalid response")
    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        raise AdapterError("the model returned an invalid response")
    calls = tuple(_parse_tool_call(item, index) for index, item in enumerate(raw_calls))
    return ModelReply(content=content, tool_calls=calls)


def tool_support_from_show(payload: object) -> str:
    """Map an Ollama /api/show body to supported, unsupported, or unknown."""
    if not isinstance(payload, dict):
        return "unknown"
    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, list):
        return "unknown"
    if any(item == "tools" for item in capabilities):
        return "supported"
    return "unsupported"


async def model_tool_support(model: ModelConfig, *, timeout_seconds: float) -> str:
    """Ask the model server whether this model can call tools.

    A missing answer disables tools. Guessing would turn them on for a model
    that cannot use them.
    """
    url = f"{native_base(model.base_url)}/api/show"
    timeout = httpx.Timeout(timeout_seconds)
    try:
        async with direct_client(timeout) as client:
            response = await client.post(url, json={"model": model.name})
            if response.status_code != 200:
                return "unknown"
            return tool_support_from_show(response.json())
    except (httpx.HTTPError, ValueError):
        return "unknown"


def _parse_tool_call(item: object, index: int) -> ToolCall:
    call_id = f"call-{index}"
    if not isinstance(item, dict):
        return ToolCall(call_id, "", None, "", True)
    raw_id = item.get("id")
    if isinstance(raw_id, str) and raw_id.strip():
        call_id = raw_id.strip()
    function = item.get("function")
    if not isinstance(function, dict):
        return ToolCall(call_id, "", None, "", True)
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        return ToolCall(call_id, "", None, "", True)
    name = name.strip()
    raw = function.get("arguments")
    if raw is None or raw == "":
        return ToolCall(call_id, name, {}, "", False)
    if isinstance(raw, dict):
        return ToolCall(call_id, name, raw, json.dumps(raw, ensure_ascii=False), False)
    if not isinstance(raw, str):
        return ToolCall(call_id, name, None, "", True)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return ToolCall(call_id, name, None, raw, True)
    if not isinstance(parsed, dict):
        return ToolCall(call_id, name, None, raw, True)
    return ToolCall(call_id, name, parsed, raw, False)


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


async def warm_model(model: ModelConfig) -> bool:
    """Ask Ollama to load this model and keep it resident. Failures are quiet."""
    url = f"{native_base(model.base_url)}/api/generate"
    timeout = httpx.Timeout(connect=2.0, read=180.0, write=30.0, pool=2.0)
    try:
        async with direct_client(timeout) as client:
            response = await client.post(url, json=warmup_payload(model))
            return response.status_code == 200
    except httpx.HTTPError:
        return False


async def model_is_loaded(model: ModelConfig) -> bool | None:
    """True when the weights are already in memory. None if Ollama did not answer."""
    url = f"{native_base(model.base_url)}/api/ps"
    timeout = httpx.Timeout(0.3)
    try:
        async with direct_client(timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    for item in payload.get("models") or []:
        if not isinstance(item, dict):
            continue
        for key in ("name", "model"):
            reported = item.get(key)
            if isinstance(reported, str) and same_model(reported.strip(), model.name):
                return True
    return False


def _remember_usage(stats: StreamStats, usage: Usage | None) -> None:
    if usage is None:
        return
    if usage.prompt_tokens is not None:
        stats.prompt_tokens = usage.prompt_tokens
    if usage.completion_tokens is not None:
        stats.completion_tokens = usage.completion_tokens


def _usage(payload: dict) -> Usage | None:
    raw = payload.get("usage")
    prompt = None
    completion = None
    if isinstance(raw, dict):
        prompt = _count(raw.get("prompt_tokens"))
        completion = _count(raw.get("completion_tokens"))
    timings = payload.get("timings")
    if isinstance(timings, dict):
        if prompt is None:
            prompt = _count(timings.get("prompt_n"))
        if completion is None:
            completion = _count(timings.get("predicted_n"))
    if prompt is None and completion is None:
        return None
    return Usage(prompt_tokens=prompt, completion_tokens=completion)


def _count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0:
        return None
    return value


async def _iter_events(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
) -> AsyncIterator[ParsedLine]:
    async with client.stream("POST", url, json=payload) as response:
        if response.status_code != 200:
            body = (await response.aread()).decode(errors="replace")[:500]
            raise AdapterError(_plain_model_error(body, response.status_code))
        async for line in response.aiter_lines():
            parsed = parse_sse_line(line)
            if parsed is None:
                continue
            yield parsed
            if parsed.done:
                return
