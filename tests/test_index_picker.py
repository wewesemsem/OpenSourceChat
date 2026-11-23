"""The folder window returns a real directory, a cancel, or a readable error."""

import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from app.index.picker import PickError, choose_directory, dialog_command, validate_directory


class PickerTests(unittest.TestCase):
    def test_each_platform_asks_for_a_directory(self):
        with patch("app.index.picker.sys.platform", "darwin"):
            command = dialog_command("Select Folder")
        self.assertEqual(command[0], "osascript")
        self.assertIn("choose folder", command[2])
        self.assertEqual(command[3], "Select Folder")

        with patch("app.index.picker.sys.platform", "win32"):
            command = dialog_command("Select Folder")
        self.assertEqual(command[0], "powershell")
        self.assertIn("FolderBrowserDialog", command[-1])

        with (
            patch("app.index.picker.sys.platform", "linux"),
            patch("app.index.picker.shutil.which", side_effect=lambda name: name == "zenity"),
        ):
            command = dialog_command("Select Folder")
        self.assertEqual(command[:3], ["zenity", "--file-selection", "--directory"])

    def test_cancel_is_empty_and_a_missing_dialog_is_an_error(self):
        cancelled = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="User canceled.")
        with (
            patch("app.index.picker.sys.platform", "darwin"),
            patch("app.index.picker.subprocess.run", return_value=cancelled),
        ):
            self.assertIsNone(choose_directory("Select Folder"))
        with (
            patch("app.index.picker.sys.platform", "linux"),
            patch("app.index.picker.shutil.which", return_value=None),
        ):
            with self.assertRaises(PickError) as caught:
                choose_directory("Select Folder")
        self.assertEqual(caught.exception.code, "unavailable")

    def test_a_chosen_path_must_be_a_readable_directory(self):
        root = Path(self.id().replace(".", "-"))
        missing = Path("/no/such/local-chat-folder")
        with self.assertRaises(PickError) as missing_error:
            validate_directory(missing)
        self.assertEqual(missing_error.exception.code, "not_found")
        with self.assertRaises(PickError) as file_error:
            validate_directory(Path(__file__))
        self.assertEqual(file_error.exception.code, "not_a_folder")
        with patch("app.index.picker.os.scandir", side_effect=PermissionError):
            with self.assertRaises(PickError) as denied:
                validate_directory(Path(__file__).resolve().parent)
        self.assertEqual(denied.exception.code, "permission")
        self.assertFalse(root.exists())
