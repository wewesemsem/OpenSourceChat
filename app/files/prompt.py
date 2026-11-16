"""Fit one file into the retrieved_context slot.

The file body is wrapped in delimiters from config. A separate instruction,
also from config, tells the model the body is reference data. The body is
never copied into that instruction.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from pathlib import Path

from app.config import FilesConfig, ModelConfig
from app.files.extract import read_opened, read_text
from app.files.safe import safe_path
from app.files.store import stored_file
from app.memory.context import message_tokens


@dataclass(frozen=True)
class SlotFit:
    retrieved: str
    instruction: str
    file_tokens: int
    budget_tokens: int
    use_prefix: bool
    over_budget: bool
    size: int
    digest: str
    name: str


def slot_budget(files: FilesConfig, model: ModelConfig, system_prompt: str) -> int:
    """How many tokens the attached file may use for this model."""
    available = model.context_size - model.reserved_output_tokens - message_tokens(system_prompt)
    if available < 1:
        return 0
    return int(available * files.context_share)


def fit_file(
    path,
    files: FilesConfig,
    model: ModelConfig,
    system_prompt: str,
    *,
    use_prefix: bool,
    root: Path | None = None,
) -> SlotFit:
    """Read a file and either fit it, mark it over budget, or take an explicit prefix.

    `root` is set for a copy this app saved. Allowlisted files leave it empty
    and go through safe_path().
    """
    if root is None:
        real = safe_path(path, files)
        extracted = read_text(real, files)
    else:
        real = stored_file(path, root, files)
        extracted = read_opened(real, files)
    payload = real.read_bytes()
    budget = slot_budget(files, model, system_prompt)
    name = real.name
    base = SlotFit(
        retrieved="",
        instruction="",
        file_tokens=0,
        budget_tokens=budget,
        use_prefix=False,
        over_budget=True,
        size=real.stat().st_size,
        digest=hashlib.sha256(payload).hexdigest(),
        name=name,
    )
    if not extracted.ok:
        raise SlotError(extracted.error_code)
    if not extracted.text.strip():
        raise SlotError("empty")
    full = _wrap(extracted.text, name, files, truncated=False)
    full_tokens = _slot_tokens(full, files)
    if full_tokens <= budget:
        return SlotFit(
            retrieved=full,
            instruction=files.context_instruction,
            file_tokens=full_tokens,
            budget_tokens=budget,
            use_prefix=False,
            over_budget=False,
            size=base.size,
            digest=base.digest,
            name=name,
        )
    if not use_prefix:
        return SlotFit(
            retrieved="",
            instruction="",
            file_tokens=full_tokens,
            budget_tokens=budget,
            use_prefix=False,
            over_budget=True,
            size=base.size,
            digest=base.digest,
            name=name,
        )
    body = _prefix_body(extracted.text, name, files, budget)
    if not body:
        return SlotFit(
            retrieved="",
            instruction="",
            file_tokens=full_tokens,
            budget_tokens=budget,
            use_prefix=True,
            over_budget=True,
            size=base.size,
            digest=base.digest,
            name=name,
        )
    wrapped = _wrap(body, name, files, truncated=True)
    return SlotFit(
        retrieved=wrapped,
        instruction=files.context_instruction,
        file_tokens=_slot_tokens(wrapped, files),
        budget_tokens=budget,
        use_prefix=True,
        over_budget=False,
        size=base.size,
        digest=base.digest,
        name=name,
    )


class SlotError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _slot_tokens(wrapped: str, files: FilesConfig) -> int:
    return message_tokens(wrapped) + message_tokens(files.context_instruction)


def _wrap(body: str, name: str, files: FilesConfig, *, truncated: bool) -> str:
    parts = [
        files.context_begin.replace("{name}", name),
        body,
        files.context_end.replace("{name}", name),
    ]
    if truncated:
        parts.append(files.truncated_note)
    return "\n".join(parts)


def _prefix_body(text: str, name: str, files: FilesConfig, budget: int) -> str:
    """Longest leading slice whose wrapped form fits the slot."""
    best = ""
    low = 0
    high = len(text)
    while low <= high:
        mid = (low + high) // 2
        candidate = text[:mid].rstrip()
        if candidate and _slot_tokens(_wrap(candidate, name, files, truncated=True), files) <= budget:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best
