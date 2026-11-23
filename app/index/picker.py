"""Open the system folder window and remember the folder the user picked.

The page receives the path only after this dialog returns it. A request cannot
name a different directory. The remembered folder is the root the scanner walks.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from app.files.browse import file_id

_lock = threading.Lock()
_dialog = threading.Lock()
_roots: dict[str, Path] = {}
_last_id: str | None = None

_WINDOWS_SCRIPT = """
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = $env:LOCAL_CHAT_FOLDER_PROMPT
$dialog.ShowNewFolderButton = $false
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
  Write-Output $dialog.SelectedPath
}
""".strip()


class PickError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def dialog_command(prompt: str) -> list[str] | None:
    """The native folder dialog for this operating system, or None when it has none."""
    if sys.platform == "darwin":
        return [
            "osascript",
            "-e",
            "on run argv\nPOSIX path of (choose folder with prompt (item 1 of argv))\nend run",
            prompt,
        ]
    if sys.platform == "win32":
        return ["powershell", "-NoProfile", "-STA", "-Command", _WINDOWS_SCRIPT]
    if shutil.which("zenity"):
        return ["zenity", "--file-selection", "--directory", "--title", prompt]
    if shutil.which("kdialog"):
        return ["kdialog", "--getexistingdirectory", str(Path.home()), "--title", prompt]
    return None


def choose_directory(prompt: str) -> Path | None:
    """Return the chosen directory. None means the user cancelled."""
    command = dialog_command(prompt)
    if command is None:
        raise PickError("unavailable")
    if not _dialog.acquire(blocking=False):
        raise PickError("busy")
    try:
        environment = os.environ.copy()
        environment["LOCAL_CHAT_FOLDER_PROMPT"] = prompt
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=600,
                env=environment,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            raise PickError("unavailable") from None
    finally:
        _dialog.release()
    if completed.returncode == 0:
        text = completed.stdout.strip()
        if not text:
            return None
        return validate_directory(text)
    if _cancelled(completed):
        return None
    raise PickError("unavailable")


def validate_directory(raw: str | Path) -> Path:
    """The real directory, or an error when it is missing or cannot be listed."""
    text = raw if isinstance(raw, str) else str(raw)
    if not text or "\x00" in text:
        raise PickError("rejected")
    try:
        real = Path(os.path.realpath(text))
    except (OSError, ValueError):
        raise PickError("rejected") from None
    if not real.exists():
        raise PickError("not_found")
    if not real.is_dir():
        raise PickError("not_a_folder")
    try:
        with os.scandir(real) as entries:
            next(entries, None)
    except PermissionError:
        raise PickError("permission") from None
    except OSError:
        raise PickError("permission") from None
    return real


def remember_root(root: Path) -> str:
    """Keep a validated directory so a later request can index it by id."""
    global _last_id
    ident = file_id(root)
    with _lock:
        _roots[ident] = root
        _last_id = ident
    return ident


def remembered_root(wanted: str) -> Path | None:
    with _lock:
        return _roots.get(wanted)


def current_selection() -> Path | None:
    with _lock:
        if _last_id is None:
            return None
        return _roots.get(_last_id)


def reset_selection() -> None:
    """Forget picked folders. Tests call this."""
    global _last_id
    with _lock:
        _roots.clear()
        _last_id = None


def _cancelled(completed: subprocess.CompletedProcess[str]) -> bool:
    text = f"{completed.stderr}\n{completed.stdout}".lower()
    return "user canceled" in text or "user cancelled" in text
