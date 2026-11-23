"""Select Folder uses the system folder window. The page cannot name a different path."""

import os
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from app import db
from app.index.jobs import public_status, reset_job
from app.index.picker import PickError, reset_selection
from app.main import config, strings
from tests.test_chat import open_client


class IndexFolderApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "shelf"
        self.root.mkdir()
        (self.root / "note.txt").write_text("zephyr lantern on the shelf\n", encoding="utf-8")
        db.configure(Path(self.tmp.name) / "chat.sqlite")
        self.release = threading.Event()
        self.release.set()
        reset_job()
        reset_selection()
        self.addCleanup(self._finish)

    def _finish(self):
        self.release.set()
        deadline = time.time() + 5
        while time.time() < deadline and public_status(strings)["running"]:
            time.sleep(0.05)
        reset_job()
        reset_selection()

    def _config(self, *, enabled=True):
        files = replace(config.files, allowed_folders=())
        index = replace(config.index, enabled=enabled)
        return replace(config, files=files, index=index)

    def _wait(self, client):
        deadline = time.time() + 5
        payload = {"running": True, "message": ""}
        while time.time() < deadline:
            payload = client.get("/api/index/job").json()
            if not payload["running"]:
                return payload
            time.sleep(0.02)
        self.fail(f"index job did not finish: {payload}")

    def test_a_picked_folder_is_indexed_and_a_typed_path_is_refused(self):
        client = open_client()
        real = os.path.realpath(self.root)
        with (
            patch("app.main.config", self._config(enabled=False)),
            patch("app.main.choose_directory", return_value=Path(real)),
        ):
            picked = client.post("/api/index/pick")
            self.assertEqual(picked.status_code, 200)
            body = picked.json()
            self.assertEqual(body["cancelled"], False)
            self.assertEqual(body["name"], "shelf")
            self.assertEqual(body["path"], real)
            refused = client.post("/api/index/folders", json={"path": real})
            self.assertEqual(refused.status_code, 422)
            unknown = client.post("/api/index/folders", json={"folder_id": real})
            self.assertEqual(unknown.status_code, 404)
            self.assertEqual(unknown.json()["detail"], strings["index_not_selected"])
            started = client.post("/api/index/folders", json={"folder_id": body["id"]})
            self.assertEqual(started.status_code, 200)
            self.assertEqual(started.json()["name"], "shelf")
            job = self._wait(client)
            selection = client.get("/api/index/selection").json()
        self.assertFalse(job["running"])
        self.assertIn("shelf", job["message"])
        self.assertIn("1 files", job["message"])
        self.assertEqual(selection["path"], real)

    def test_cancel_permission_and_a_second_start(self):
        client = open_client()
        real = os.path.realpath(self.root)
        entered = threading.Event()

        def hold(root, cfg):
            entered.set()
            if not self.release.wait(3):
                raise TimeoutError("release")
            raise RuntimeError(real)

        self.release.clear()
        with patch("app.main.config", self._config(enabled=False)):
            with patch("app.main.choose_directory", return_value=None):
                cancelled = client.post("/api/index/pick")
            self.assertEqual(cancelled.json(), {"cancelled": True})
            with patch("app.main.choose_directory", side_effect=PickError("permission")):
                denied = client.post("/api/index/pick")
            self.assertEqual(denied.status_code, 400)
            self.assertEqual(denied.json()["detail"], strings["index_permission"])
            with (
                patch("app.main.choose_directory", return_value=Path(real)),
                patch("app.index.jobs.index_folder", hold),
            ):
                picked = client.post("/api/index/pick").json()
                first = client.post("/api/index/folders", json={"folder_id": picked["id"]})
                self.assertEqual(first.status_code, 200)
                self.assertTrue(entered.wait(2))
                second = client.post("/api/index/folders", json={"folder_id": picked["id"]})
                self.assertEqual(second.status_code, 409)
                self.assertEqual(second.json()["detail"], strings["index_busy"])
                self.release.set()
                job = self._wait(client)
        self.assertEqual(job["message"], strings["index_failed_run"])
        self.assertNotIn(real, job["message"])
