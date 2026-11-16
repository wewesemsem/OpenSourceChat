"""Allowlist, symlink escapes, and per-file extraction errors."""

import json
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfWriter

from app.config import ROOT
from app.files import ERROR_CODES, message_for, read_text, safe_path
from app.files.safe import FileAccessError


def _settings(folder: Path, **overrides):
    from app.config import FilesConfig

    values = dict(
        allowed_folders=(Path(folder).resolve(),),
        allowed_extensions=("txt", "md", "pdf", "docx", "csv"),
        max_file_size_bytes=1024,
        max_pages=2,
        exclude_patterns=(".*", ".env", ".git", "*.key", "*.pem", "id_rsa"),
        extract_timeout_seconds=2,
        max_extract_chars=1000,
        context_share=0.5,
        context_instruction="Reference only.",
        context_begin="----- BEGIN FILE {name} -----",
        context_end="----- END FILE {name} -----",
        truncated_note="Only the beginning of the file is included.",
        max_list_entries=50,
        max_list_depth=4,
    )
    values.update(overrides)
    return FilesConfig(**values)


class SafePathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "allowed"
        self.outside = Path(self.tmp.name) / "outside"
        self.root.mkdir()
        self.outside.mkdir()
        self.files = _settings(self.root)

    def test_reads_a_file_inside_the_folder(self):
        target = self.root / "notes.txt"
        target.write_text("hello", encoding="utf-8")
        self.assertEqual(safe_path(target, self.files), target.resolve())
        result = read_text(self.root / "sub" / ".." / "notes.txt", self.files)
        self.assertTrue(result.ok)
        self.assertEqual(result.text, "hello")
        self.assertEqual(result.name, "notes.txt")

    def test_rejects_a_path_that_leaves_the_folder(self):
        secret = self.outside / "secret.txt"
        secret.write_text("nope", encoding="utf-8")
        escaped = self.root / ".." / "outside" / "secret.txt"
        with self.assertRaises(FileAccessError) as caught:
            safe_path(escaped, self.files)
        self.assertEqual(caught.exception.code, "rejected")
        result = read_text(escaped, self.files)
        self.assertEqual(result.error_code, "rejected")
        self.assertNotIn("nope", result.text)

    def test_rejects_a_relative_path(self):
        with self.assertRaises(FileAccessError) as caught:
            safe_path("notes.txt", self.files)
        self.assertEqual(caught.exception.code, "rejected")

    def test_rejects_a_symlink_that_leaves_the_folder(self):
        secret = self.outside / "secret.txt"
        secret.write_text("hidden", encoding="utf-8")
        link = self.root / "link.txt"
        link.symlink_to(secret)
        with self.assertRaises(FileAccessError) as caught:
            safe_path(link, self.files)
        self.assertEqual(caught.exception.code, "rejected")
        result = read_text(link, self.files)
        self.assertEqual(result.text, "")
        self.assertEqual(result.error_code, "rejected")

    def test_follows_a_symlink_that_stays_inside(self):
        target = self.root / "real.txt"
        target.write_text("inside", encoding="utf-8")
        link = self.root / "alias.txt"
        link.symlink_to(target)
        self.assertEqual(read_text(link, self.files).text, "inside")

    def test_rejects_excluded_names(self):
        (self.root / ".env").write_text("TOKEN=1", encoding="utf-8")
        (self.root / ".secret.txt").write_text("hidden", encoding="utf-8")
        (self.root / "private.key").write_text("key", encoding="utf-8")
        nested = self.root / ".git"
        nested.mkdir()
        (nested / "notes.txt").write_text("git", encoding="utf-8")
        for path in (
            self.root / ".env",
            self.root / ".secret.txt",
            self.root / "private.key",
            nested / "notes.txt",
        ):
            with self.subTest(path=path.name):
                with self.assertRaises(FileAccessError) as caught:
                    safe_path(path, self.files)
                self.assertEqual(caught.exception.code, "excluded")

    def test_rejects_an_oversized_file_without_returning_its_text(self):
        target = self.root / "big.txt"
        target.write_bytes(b"x" * (self.files.max_file_size_bytes + 1))
        result = read_text(target, self.files)
        self.assertEqual(result.error_code, "too_large")
        self.assertEqual(result.text, "")

    def test_disabled_when_no_folder_is_configured(self):
        files = _settings(self.root, allowed_folders=())
        target = self.root / "notes.txt"
        target.write_text("hello", encoding="utf-8")
        result = read_text(target, files)
        self.assertEqual(result.error_code, "disabled")


class ExtractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "allowed"
        self.root.mkdir()
        self.files = _settings(self.root, max_file_size_bytes=1024 * 1024)

    def test_corrupt_pdf_and_docx_are_reported(self):
        pdf = self.root / "bad.pdf"
        pdf.write_bytes(b"not a pdf")
        docx = self.root / "bad.docx"
        docx.write_bytes(b"not a zip")
        self.assertEqual(read_text(pdf, self.files).error_code, "corrupt")
        self.assertEqual(read_text(docx, self.files).error_code, "corrupt")

    def test_reads_csv_and_docx_text(self):
        csv_path = self.root / "rows.csv"
        csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
        self.assertEqual(read_text(csv_path, self.files).text, "a\tb\n1\t2")
        docx_path = self.root / "note.docx"
        xml = (
            "<?xml version='1.0' encoding='UTF-8'?>"
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body><w:p><w:r><w:t>Hello doc</w:t></w:r></w:p></w:body>"
            "</w:document>"
        )
        with zipfile.ZipFile(docx_path, "w") as package:
            package.writestr("word/document.xml", xml)
        self.assertEqual(read_text(docx_path, self.files).text, "Hello doc")

    def test_rejects_a_pdf_with_too_many_pages(self):
        path = self.root / "pages.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.add_blank_page(width=72, height=72)
        writer.add_blank_page(width=72, height=72)
        with path.open("wb") as handle:
            writer.write(handle)
        result = read_text(path, self.files)
        self.assertEqual(result.error_code, "too_many_pages")
        self.assertEqual(result.text, "")

    def test_does_not_silently_truncate_long_text(self):
        path = self.root / "long.txt"
        path.write_text("hello world", encoding="utf-8")
        files = _settings(self.root, max_extract_chars=5, max_file_size_bytes=1024)
        result = read_text(path, files)
        self.assertEqual(result.error_code, "too_long")
        self.assertEqual(result.text, "")

    def test_timeout_is_reported(self):
        path = self.root / "slow.txt"
        path.write_text("hi", encoding="utf-8")
        files = _settings(self.root, extract_timeout_seconds=0.05, max_file_size_bytes=1024)

        def _slow(real, settings):
            time.sleep(0.4)
            return "late"

        with patch("app.files.extract._extract_real", _slow):
            result = read_text(path, files)
        self.assertEqual(result.error_code, "timeout")
        self.assertEqual(result.text, "")

    def test_every_error_code_has_a_string(self):
        strings = json.loads((ROOT / "app" / "strings.json").read_text(encoding="utf-8"))
        for code in ERROR_CODES:
            self.assertTrue(message_for(code, strings))


if __name__ == "__main__":
    unittest.main()
