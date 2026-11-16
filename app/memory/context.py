"""Assemble one stateless prompt from named slots.

The budget is the model's context size minus the tokens reserved for the
reply. Slots are charged in order:

    system_prompt | pinned_notes | retrieved_context | history

History receives whatever remains. The oldest turns are dropped first.
The system prompt is never dropped. Retrieved file text is its own user
message, not part of the system prompt.

Tokens are estimated at about four characters each. Llama and Gemma use
different tokenizers, and a tokenizer package would match only one of
them, so the estimate stays in this file and rounds up.
"""

from __future__ import annotations

from dataclasses import dataclass

# Role markers around a message. Charged so the window stays under the budget.
MESSAGE_OVERHEAD = 4


@dataclass(frozen=True)
class BuiltPrompt:
    messages: list[dict[str, str]]
    trimmed: bool
    tokens: int


def estimate_tokens(text: str) -> int:
    """Round a string up to a token count. Empty text is zero."""
    if not text:
        return 0
    return (len(text) + 3) // 4


def message_tokens(content: str) -> int:
    counted = estimate_tokens(content)
    if counted == 0:
        return 0
    return counted + MESSAGE_OVERHEAD


def build_prompt(
    history: list[dict],
    *,
    system_prompt: str,
    context_size: int,
    reserved_output_tokens: int,
    pinned_notes: str = "",
    retrieved_context: str = "",
    context_instruction: str = "",
) -> BuiltPrompt:
    """Build the messages for one model call.

    `history` is the stored transcript. Only user and assistant text is
    used, so a reasoning trace cannot be fed back in.
    """
    if context_size < 1:
        raise ValueError("context_size must be positive")
    if reserved_output_tokens < 0 or reserved_output_tokens >= context_size:
        raise ValueError("reserved_output_tokens must be smaller than context_size")
    system = system_prompt.strip()
    if not system:
        raise ValueError("system_prompt is empty")

    budget = context_size - reserved_output_tokens
    prefix = _prefix(system, pinned_notes, retrieved_context, context_instruction)
    turns = _turns(_visible(history))
    if not turns:
        raise ValueError("messages is empty")

    remaining = budget - sum(message_tokens(item["content"]) for item in prefix)
    kept, dropped = _fit_history(turns, remaining)
    messages = [*prefix, *[item for turn in kept for item in turn]]
    return BuiltPrompt(
        messages=messages,
        trimmed=dropped,
        tokens=sum(message_tokens(item["content"]) for item in messages),
    )


def _prefix(
    system: str,
    pinned_notes: str,
    retrieved_context: str,
    context_instruction: str,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system}]
    if pinned_notes.strip():
        messages.append({"role": "system", "content": pinned_notes.strip()})
    retrieved = retrieved_context.strip()
    if not retrieved:
        return messages
    instruction = context_instruction.strip()
    if instruction:
        messages.append({"role": "system", "content": instruction})
    messages.append({"role": "user", "content": retrieved})
    return messages


def _visible(history: list[dict]) -> list[dict[str, str]]:
    visible = []
    for message in history:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        text = content.strip()
        if not text:
            continue
        visible.append({"role": role, "content": text})
    return visible


def _turns(messages: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    """Group the transcript into turns. A user message starts a new turn."""
    turns: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    for message in messages:
        if message["role"] == "user" and current:
            turns.append(current)
            current = []
        current.append(message)
    if current:
        turns.append(current)
    return turns


def _fit_history(
    turns: list[list[dict[str, str]]],
    remaining: int,
) -> tuple[list[list[dict[str, str]]], bool]:
    """Keep a suffix of turns. Truncate only when the newest turn alone is too big."""
    kept: list[list[dict[str, str]]] = []
    used = 0
    skipped = 0
    for turn in reversed(turns):
        cost = sum(message_tokens(item["content"]) for item in turn)
        if used + cost <= remaining:
            kept.append(turn)
            used += cost
            continue
        skipped = len(turns) - len(kept)
        break
    else:
        skipped = 0
    kept.reverse()
    if kept:
        return kept, skipped > 0

    fitted = _fit_turn(turns[-1], remaining)
    changed = fitted != turns[-1]
    return ([fitted] if fitted else []), skipped > 0 or changed or not fitted


def _fit_turn(turn: list[dict[str, str]], remaining: int) -> list[dict[str, str]]:
    """Keep the newest messages in one turn. The oldest kept message may be cut."""
    chosen: list[dict[str, str]] = []
    used = 0
    for message in reversed(turn):
        cost = message_tokens(message["content"])
        if used + cost <= remaining:
            chosen.append(message)
            used += cost
            continue
        tail = _tail(message["content"], remaining - used - MESSAGE_OVERHEAD)
        if tail:
            chosen.append({"role": message["role"], "content": tail})
        break
    chosen.reverse()
    return chosen


def _tail(text: str, token_room: int) -> str:
    """Return the newest slice of text that fits in token_room."""
    if token_room <= 0 or not text:
        return ""
    limit = token_room * 4 - 3
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[-limit:]
