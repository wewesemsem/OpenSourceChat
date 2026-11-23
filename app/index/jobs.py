"""One folder index at a time, off the request thread.

A large folder can still keep SQLite busy between files. Pausing while a chat
reply is in progress is not built yet.
"""

from __future__ import annotations

import threading
from pathlib import Path

from app.config import AppConfig
from app.index.scan import IndexResult, index_folder

_lock = threading.Lock()
_state: dict = {"running": False, "name": "", "result": None}


def start_job(root: Path, config: AppConfig) -> bool:
    """Return False when a folder is already being indexed."""
    with _lock:
        if _state["running"]:
            return False
        _state["running"] = True
        _state["name"] = root.name
        _state["result"] = None
    threading.Thread(
        target=_execute,
        args=(root, config),
        name="index-folder",
        daemon=True,
    ).start()
    return True


def public_status(texts: dict[str, str]) -> dict:
    """Status the page can show. Paths stay on the server."""
    with _lock:
        running = _state["running"]
        name = _state["name"]
        result = _state["result"]
    if running:
        return {
            "running": True,
            "name": name,
            "message": texts["index_running"].replace("{name}", name),
        }
    if result is None:
        return {"running": False, "name": "", "message": ""}
    from app.index.__main__ import render_index as render_result

    return {
        "running": False,
        "name": result.folder_name,
        "message": "\n".join(render_result(result, texts)),
    }


def reset_job() -> None:
    """Clear the in-memory job. Tests call this after the thread has finished."""
    with _lock:
        _state["running"] = False
        _state["name"] = ""
        _state["result"] = None


def _execute(root: Path, config: AppConfig) -> None:
    try:
        result = index_folder(root, config)
    except Exception:
        result = IndexResult(
            indexed=False,
            reason="failed",
            folder_name=root.name,
            retrieval_mode=config.index.retrieval_mode,
        )
    with _lock:
        _state["running"] = False
        _state["name"] = result.folder_name
        _state["result"] = result
