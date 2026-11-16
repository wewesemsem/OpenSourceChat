"""FastAPI app: streaming chat with local conversation storage."""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator, model_validator

from app import db
from app.config import ModelConfig, load_config
from app.files.browse import file_id, list_entries, resolve_file
from app.files.extract import message_for
from app.files.prompt import SlotError, fit_file
from app.files.safe import FileAccessError, screen_name
from app.files.store import (
    discard_upload,
    find_upload,
    save_upload,
    upload_id_of,
    uploads_root,
)
from app.memory.context import BuiltPrompt, build_prompt
from app.models.adapter import AdapterError, ChatAdapter, probe_model
from app.security import GuardMiddleware, cors_kwargs, render_index, warn_if_public_bind

STATIC_DIR = Path(__file__).resolve().parent / "static"
STRINGS_PATH = Path(__file__).resolve().parent / "strings.json"
config = load_config()
strings = json.loads(STRINGS_PATH.read_text(encoding="utf-8"))
if not isinstance(strings, dict) or any(not isinstance(value, str) for value in strings.values()):
    raise ValueError("strings.json must be an object of strings")
for key in (
    "untitled",
    "host_rejected",
    "origin_rejected",
    "csrf_rejected",
    "auth_rejected",
    "bind_warning",
    "file_disabled",
    "file_rejected",
    "file_excluded",
    "file_too_large",
    "file_unsupported",
    "file_corrupt",
    "file_timeout",
    "file_too_many_pages",
    "file_too_long",
    "file_not_found",
    "file_not_a_file",
    "file_empty",
    "file_over_budget",
    "file_use_prefix",
    "file_index",
    "file_index_unavailable",
    "file_prefixed",
    "file_empty_list",
    "file_list_truncated",
    "file_back",
    "attach",
    "attach_title",
    "file_remove",
):
    if key not in strings:
        raise ValueError(f"strings.json missing {key}")
db.configure(config.data_dir / "chat.sqlite")


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    warn_if_public_bind(config.host, strings["bind_warning"])
    yield


app = FastAPI(title=config.ui.title, lifespan=_lifespan)
if config.security.cors_enabled:
    app.add_middleware(CORSMiddleware, **cors_kwargs(config.security))
app.add_middleware(
    GuardMiddleware,
    security=config.security,
    messages=strings,
    bound_port=config.port,
)


class ChatInputError(Exception):
    def __init__(self, message: str, status: int = 400, body: dict | None = None):
        super().__init__(message)
        self.status = status
        self.body = body


class ChatRequest(BaseModel):
    content: str | None = None
    model: str | None = None
    conversation_id: str | None = None
    regenerate: bool = False

    @field_validator("content", "model")
    @classmethod
    def strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class RenameRequest(BaseModel):
    title: str

    @field_validator("title")
    @classmethod
    def strip_title(cls, value: str) -> str:
        cleaned = " ".join(value.strip().split())
        if not cleaned:
            raise ValueError("title is empty")
        return cleaned


@dataclass
class Prepared:
    conversation: dict
    model: ModelConfig
    history: list[dict[str, str]]
    user_message_id: str
    trimmed: bool


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def public_conversation(row: dict) -> dict:
    return {"id": row["id"], "title": row["title"], "updated_at": row["updated_at"]}


def public_message(row: dict) -> dict:
    return {
        "id": row["id"],
        "role": row["role"],
        "content": row["content"],
        "model": row["model"],
        "latency_ms": row["latency_ms"],
        "created_at": row["created_at"],
    }


def find_model(name: str | None) -> ModelConfig:
    try:
        return config.model(name)
    except KeyError as exc:
        raise ChatInputError(f"unknown model {exc.args[0]}") from exc


def prepare_turn(request: ChatRequest) -> Prepared:
    if request.regenerate and request.content:
        raise ChatInputError("send either a message or a retry")
    model = find_model(request.model)
    if request.regenerate:
        return _prepare_regenerate(request, model)
    if not request.content:
        raise ChatInputError("message is empty")
    return _prepare_message(request, model)


def _prepare_message(request: ChatRequest, model: ModelConfig) -> Prepared:
    if request.conversation_id:
        conversation = db.get_conversation(request.conversation_id)
        if conversation is None:
            raise ChatInputError("conversation not found", 404)
    else:
        conversation = db.create_conversation(strings["untitled"], config.system_prompt)
    if conversation["title"] == strings["untitled"]:
        renamed = db.rename_conversation(conversation["id"], db.title_from_text(request.content or ""))
        if renamed:
            conversation = renamed
    slot = file_slot(conversation, model)
    user = db.add_message(
        conversation_id=conversation["id"],
        role="user",
        content=request.content or "",
        model=model.name,
    )
    built = assemble(
        db.list_messages(conversation["id"]),
        conversation["system_prompt"],
        model,
        retrieved=slot[0],
        instruction=slot[1],
    )
    return Prepared(conversation, model, built.messages, user["id"], built.trimmed)


def _prepare_regenerate(request: ChatRequest, model: ModelConfig) -> Prepared:
    if not request.conversation_id:
        raise ChatInputError("missing conversation")
    conversation = db.get_conversation(request.conversation_id)
    if conversation is None:
        raise ChatInputError("conversation not found", 404)
    db.delete_last_assistant(conversation["id"])
    messages = db.list_messages(conversation["id"])
    if not messages or messages[-1]["role"] != "user":
        raise ChatInputError("nothing to retry")
    user = messages[-1]
    if user["model"] != model.name:
        db.set_message_model(user["id"], model.name)
        user["model"] = model.name
    slot = file_slot(conversation, model)
    built = assemble(
        messages,
        conversation["system_prompt"],
        model,
        retrieved=slot[0],
        instruction=slot[1],
    )
    return Prepared(conversation, model, built.messages, user["id"], built.trimmed)


def assemble(
    messages: list[dict],
    system_prompt: str,
    model: ModelConfig,
    *,
    retrieved: str = "",
    instruction: str = "",
) -> BuiltPrompt:
    try:
        return build_prompt(
            messages,
            system_prompt=system_prompt,
            context_size=model.context_size,
            reserved_output_tokens=model.reserved_output_tokens,
            retrieved_context=retrieved,
            context_instruction=instruction,
        )
    except ValueError as exc:
        raise ChatInputError(str(exc)) from exc


def file_slot(conversation: dict, model: ModelConfig) -> tuple[str, str]:
    row = db.get_attachment(conversation["id"])
    if row is None:
        return "", ""
    stored_id = upload_id_of(row["path"], config.data_dir)
    try:
        fitted = fit_file(
            row["path"],
            config.files,
            model,
            conversation["system_prompt"],
            use_prefix=bool(row["use_prefix"]),
            root=uploads_root(config.data_dir) if stored_id else None,
        )
    except FileAccessError as exc:
        raise ChatInputError(message_for(exc.code, strings)) from exc
    except SlotError as exc:
        key = exc.code if exc.code.startswith("file_") else f"file_{exc.code}"
        raise ChatInputError(strings.get(key, strings["file_corrupt"])) from exc
    if fitted.over_budget:
        raise ChatInputError(
            strings["file_over_budget"],
            status=409,
            body=_over_budget_body(
                fitted,
                None if stored_id else file_id(Path(row["path"])),
                stored_id,
            ),
        )
    if fitted.size != row["size"] or fitted.digest != row["hash"]:
        db.update_attachment_stat(row["id"], fitted.size, fitted.digest)
    return fitted.retrieved, fitted.instruction


def _over_budget_body(fitted, listed_id: str | None = None, stored_id: str | None = None) -> dict:
    detail = (
        strings["file_over_budget"]
        .replace("{name}", fitted.name)
        .replace("{tokens}", str(fitted.file_tokens))
        .replace("{budget}", str(fitted.budget_tokens))
    )
    body = {
        "detail": detail,
        "code": "file_over_budget",
        "name": fitted.name,
        "file_tokens": fitted.file_tokens,
        "budget_tokens": fitted.budget_tokens,
    }
    if listed_id:
        body["file_id"] = listed_id
    if stored_id:
        body["upload_id"] = stored_id
    return body


class FreshStatic(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/static", FreshStatic(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> HTMLResponse:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(render_index(html, config.security), headers={"Cache-Control": "no-cache"})


@app.get("/api/meta")
def meta() -> dict:
    return {
        "title": config.ui.title,
        "footer": config.ui.footer,
        "examples": list(config.ui.examples),
        "default_model": config.default_model,
        "api_base": config.ui.api_base,
        "files_enabled": config.files.allow_upload or bool(config.files.allowed_folders),
        "upload_enabled": config.files.allow_upload,
        "file_extensions": list(config.files.allowed_extensions),
        "strings": strings,
    }


@app.get("/api/models")
async def models() -> dict:
    listed = []
    for model in config.models:
        listed.append(
            {
                "name": model.name,
                "label": model.label,
                "context_size": model.context_size,
                "status": await probe_model(model),
            }
        )
    return {"models": listed}


@app.get("/api/conversations")
def conversations() -> dict:
    return {"conversations": db.list_conversations()}


@app.get("/api/conversations/{conversation_id}")
def conversation_detail(conversation_id: str) -> dict:
    conversation = db.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    attachment = db.get_attachment(conversation_id)
    return {
        "conversation": public_conversation(conversation),
        "messages": [public_message(item) for item in db.list_messages(conversation_id)],
        "attachment": None if attachment is None else db.public_attachment(attachment),
    }


@app.patch("/api/conversations/{conversation_id}")
def rename(conversation_id: str, request: RenameRequest) -> dict:
    conversation = db.rename_conversation(conversation_id, request.title)
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return public_conversation(conversation)


@app.delete("/api/conversations/{conversation_id}")
def remove_conversation(conversation_id: str) -> dict:
    attachment = db.get_attachment(conversation_id)
    if not db.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="conversation not found")
    if attachment is not None:
        stored_id = upload_id_of(attachment["path"], config.data_dir)
        if stored_id:
            discard_upload(config.data_dir, stored_id)
    return {"ok": True}


@app.delete("/api/messages/{message_id}")
def remove_message(message_id: str) -> dict:
    conversation_id = db.delete_message(message_id)
    if conversation_id is None:
        raise HTTPException(status_code=404, detail="message not found")
    return {"ok": True, "conversation_id": conversation_id}


@app.get("/api/files")
def files_root() -> dict:
    return list_entries(config.files)


@app.get("/api/files/{folder_id}")
def files_folder(folder_id: str) -> dict:
    try:
        return list_entries(config.files, folder_id)
    except FileAccessError as exc:
        raise HTTPException(status_code=404, detail=message_for(exc.code, strings)) from exc


class AttachRequest(BaseModel):
    model_config = {"extra": "forbid"}
    file_id: str | None = None
    upload_id: str | None = None
    conversation_id: str | None = None
    model: str | None = None
    choice: str | None = None

    @field_validator("file_id", "upload_id")
    @classmethod
    def stripped(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("choice")
    @classmethod
    def known_choice(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value not in {"prefix", "index"}:
            raise ValueError("unknown choice")
        return value

    @model_validator(mode="after")
    def one_target(self):
        if bool(self.file_id) == bool(self.upload_id):
            raise ValueError("choose one file")
        return self


@app.post("/api/attachments/upload")
async def upload_attachment(request: Request):
    if not config.files.allow_upload:
        raise HTTPException(status_code=404, detail=strings["file_disabled"])
    choice = request.query_params.get("choice") or None
    if choice not in {None, "prefix", "index"}:
        raise HTTPException(status_code=400, detail="unknown choice")
    if choice == "index":
        return JSONResponse(
            status_code=409,
            content={"detail": strings["file_index_unavailable"], "code": "file_index_unavailable"},
        )
    filename = request.query_params.get("filename") or ""
    try:
        screen = screen_name(filename, config.files)
    except FileAccessError as exc:
        raise HTTPException(status_code=400, detail=message_for(exc.code, strings)) from exc
    length = request.headers.get("content-length")
    if length is not None:
        try:
            declared = int(length)
        except ValueError:
            raise HTTPException(status_code=400, detail=strings["file_rejected"]) from None
        if declared > config.files.max_file_size_bytes:
            raise HTTPException(status_code=400, detail=strings["file_too_large"])
    payload = await request.body()
    if len(payload) > config.files.max_file_size_bytes:
        raise HTTPException(status_code=400, detail=strings["file_too_large"])
    try:
        upload_id, path = save_upload(config.data_dir, screen, payload, config.files)
    except FileAccessError as exc:
        raise HTTPException(status_code=400, detail=message_for(exc.code, strings)) from exc
    return _attach_saved(
        path,
        request.query_params.get("conversation_id") or None,
        request.query_params.get("model") or None,
        choice,
        upload_id=upload_id,
    )


@app.delete("/api/uploads/{upload_id}")
def remove_upload(upload_id: str) -> dict:
    path = find_upload(config.data_dir, upload_id)
    if path is not None and _attachment_uses(path):
        raise HTTPException(status_code=409, detail=strings["file_rejected"])
    if not discard_upload(config.data_dir, upload_id):
        raise HTTPException(status_code=404, detail=strings["file_not_found"])
    return {"ok": True}


@app.post("/api/attachments")
def attach(request: AttachRequest):
    if request.upload_id:
        target = find_upload(config.data_dir, request.upload_id)
        upload_id = request.upload_id
    else:
        target = resolve_file(config.files, request.file_id or "")
        upload_id = None
    if target is None:
        raise HTTPException(status_code=404, detail=strings["file_rejected"])
    if request.choice == "index":
        if upload_id:
            discard_upload(config.data_dir, upload_id)
        return JSONResponse(
            status_code=409,
            content={"detail": strings["file_index_unavailable"], "code": "file_index_unavailable"},
        )
    return _attach_saved(
        target,
        request.conversation_id,
        request.model,
        request.choice,
        upload_id=upload_id,
        listed_id=request.file_id,
    )


def _attach_saved(
    target: Path,
    conversation_id: str | None,
    model_name: str | None,
    choice: str | None,
    *,
    upload_id: str | None,
    listed_id: str | None = None,
):
    if conversation_id:
        conversation = db.get_conversation(conversation_id)
        if conversation is None:
            if upload_id:
                discard_upload(config.data_dir, upload_id)
            raise HTTPException(status_code=404, detail="conversation not found")
        system_prompt = conversation["system_prompt"]
    else:
        conversation = None
        system_prompt = config.system_prompt
    try:
        model = find_model(model_name)
    except ChatInputError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    root = uploads_root(config.data_dir) if upload_id else None
    try:
        fitted = fit_file(
            target,
            config.files,
            model,
            system_prompt,
            use_prefix=choice == "prefix",
            root=root,
        )
    except FileAccessError as exc:
        if upload_id:
            discard_upload(config.data_dir, upload_id)
        raise HTTPException(status_code=400, detail=message_for(exc.code, strings)) from exc
    except SlotError as exc:
        if upload_id:
            discard_upload(config.data_dir, upload_id)
        key = exc.code if exc.code.startswith("file_") else f"file_{exc.code}"
        raise HTTPException(status_code=400, detail=strings.get(key, strings["file_corrupt"])) from exc
    if fitted.over_budget:
        return JSONResponse(
            status_code=409,
            content=_over_budget_body(fitted, None if upload_id else listed_id, upload_id),
        )
    if conversation is None:
        conversation = db.create_conversation(strings["untitled"], config.system_prompt)
    previous = db.get_attachment(conversation["id"])
    stored = db.set_attachment(
        conversation_id=conversation["id"],
        path=str(target),
        name=fitted.name,
        size=fitted.size,
        digest=fitted.digest,
        use_prefix=fitted.use_prefix,
    )
    if previous is not None and previous["path"] != str(target):
        previous_id = upload_id_of(previous["path"], config.data_dir)
        if previous_id:
            discard_upload(config.data_dir, previous_id)
    return {
        "conversation": public_conversation(conversation),
        "attachment": db.public_attachment(stored),
    }


def _attachment_uses(path: Path) -> bool:
    with db.connection() as conn:
        row = conn.execute("SELECT 1 FROM attachments WHERE path = ?", (str(path),)).fetchone()
    return row is not None


@app.delete("/api/attachments/{attachment_id}")
def remove_attachment(attachment_id: str) -> dict:
    row = db.delete_attachment(attachment_id)
    if row is None:
        raise HTTPException(status_code=404, detail=strings["file_not_found"])
    stored_id = upload_id_of(row["path"], config.data_dir)
    if stored_id:
        discard_upload(config.data_dir, stored_id)
    return {"ok": True, "conversation_id": row["conversation_id"]}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        prepared = prepare_turn(request)
    except ChatInputError as exc:
        if exc.body is not None:
            return JSONResponse(status_code=exc.status, content=exc.body)
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc

    async def events():
        parts: list[str] = []
        started = time.perf_counter()
        error_payload = None
        row = None
        latency_ms = 0
        try:
            yield sse(
                {
                    "conversation": public_conversation(prepared.conversation),
                    "user_message_id": prepared.user_message_id,
                    "trimmed": prepared.trimmed,
                }
            )
            async for delta in ChatAdapter(prepared.model).stream(prepared.history):
                parts.append(delta)
                yield sse({"delta": delta})
        except AdapterError as exc:
            error_payload = {"code": exc.code, "message": str(exc)}
        finally:
            latency_ms = int((time.perf_counter() - started) * 1000)
            text = "".join(parts)
            if text:
                row = db.add_message(
                    conversation_id=prepared.conversation["id"],
                    role="assistant",
                    content=text,
                    model=prepared.model.name,
                    latency_ms=latency_ms,
                )
        if error_payload:
            yield sse(
                {
                    "error": error_payload,
                    "message_id": None if row is None else row["id"],
                    "latency_ms": latency_ms,
                }
            )
            return
        yield sse(
            {
                "done": True,
                "message_id": None if row is None else row["id"],
                "latency_ms": latency_ms,
                "model": prepared.model.name,
            }
        )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
