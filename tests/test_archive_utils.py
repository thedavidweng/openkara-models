"""Tests for scripts/archive_utils.py — safe tar/zip extraction.

Covers the malicious-archive rejection rules:
  - absolute paths
  - drive-prefixed paths
  - .. traversal
  - symlink and hardlink escapes
  - duplicate normalized output paths
  - excessive member counts
  - excessive extracted size

Malicious archives are generated in-memory by the tests (no binary fixtures
committed to the repo) so the fixtures always match the current archive_utils
checks.
"""

from __future__ import annotations

import io
import sys
import tarfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import archive_utils  # noqa: E402


def _make_tar(members: list[tuple[str, bytes, dict]], path: Path) -> None:
    """members: list of (name, data, extra) where extra can set linkname/type."""
    with tarfile.open(path, "w:gz") as tar:
        for name, data, extra in members:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            for k, v in extra.items():
                setattr(info, k, v)
            tar.addfile(info, io.BytesIO(data))


def _make_zip(members: list[tuple[str, bytes]], path: Path) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members:
            zf.writestr(name, data)


class SafeReadArchiveTests(unittest.TestCase):
    def test_benign_tar(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ok.tar.gz"
            _make_tar([("a.txt", b"hello", {}), ("sub/b.txt", b"world", {})], p)
            files = archive_utils.safe_read_archive(p)
            self.assertEqual(files["a.txt"], b"hello")
            self.assertEqual(files["sub/b.txt"], b"world")

    def test_benign_zip(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ok.zip"
            _make_zip([("a.txt", b"hello"), ("sub/b.txt", b"world")], p)
            files = archive_utils.safe_read_archive(p)
            self.assertEqual(files["a.txt"], b"hello")

    def test_rejects_absolute_path_tar(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.tar.gz"
            _make_tar([("/etc/passwd", b"x", {})], p)
            with self.assertRaises(archive_utils.UnsafeArchiveError) as ctx:
                archive_utils.safe_read_archive(p)
            self.assertIn("absolute", str(ctx.exception).lower())

    def test_rejects_drive_prefixed_path_zip(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.zip"
            _make_zip([("C:/Windows/system32/evil.txt", b"x")], p)
            with self.assertRaises(archive_utils.UnsafeArchiveError):
                archive_utils.safe_read_archive(p)

    def test_rejects_traversal_tar(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.tar.gz"
            _make_tar([("../../etc/passwd", b"x", {})], p)
            with self.assertRaises(archive_utils.UnsafeArchiveError) as ctx:
                archive_utils.safe_read_archive(p)
            self.assertIn("traversal", str(ctx.exception).lower())

    def test_rejects_traversal_zip(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.zip"
            _make_zip([("../escape.txt", b"x")], p)
            with self.assertRaises(archive_utils.UnsafeArchiveError):
                archive_utils.safe_read_archive(p)

    def test_rejects_symlink_escape_tar(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.tar.gz"
            _make_tar([("link", b"", {"type": tarfile.SYMTYPE, "linkname": "../../etc/passwd"})], p)
            with self.assertRaises(archive_utils.UnsafeArchiveError) as ctx:
                archive_utils.safe_read_archive(p)
            self.assertIn("link", str(ctx.exception).lower())

    def test_rejects_hardlink_escape_tar(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.tar.gz"
            _make_tar([("link", b"", {"type": tarfile.LNKTYPE, "linkname": "../../etc/passwd"})], p)
            with self.assertRaises(archive_utils.UnsafeArchiveError):
                archive_utils.safe_read_archive(p)

    def test_rejects_duplicate_normalized_path_tar(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.tar.gz"
            _make_tar([
                ("./a.txt", b"safe", {}),
                ("a.txt", b"evil", {}),
            ], p)
            with self.assertRaises(archive_utils.UnsafeArchiveError) as ctx:
                archive_utils.safe_read_archive(p)
            self.assertIn("duplicate", str(ctx.exception).lower())

    def test_rejects_excessive_member_count(self):
        import tempfile
        # Patch the limit down to make the test fast.
        orig = archive_utils.MAX_MEMBER_COUNT
        archive_utils.MAX_MEMBER_COUNT = 5
        try:
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / "bad.tar.gz"
                _make_tar([(f"f{i}.txt", b"x", {}) for i in range(10)], p)
                with self.assertRaises(archive_utils.UnsafeArchiveError) as ctx:
                    archive_utils.safe_read_archive(p)
                self.assertIn("member count", str(ctx.exception).lower())
        finally:
            archive_utils.MAX_MEMBER_COUNT = orig

    def test_rejects_excessive_extracted_size(self):
        import tempfile
        orig = archive_utils.MAX_EXTRACTED_SIZE
        archive_utils.MAX_EXTRACTED_SIZE = 100
        try:
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / "bad.tar.gz"
                _make_tar([("big.txt", b"x" * 200, {})], p)
                with self.assertRaises(archive_utils.UnsafeArchiveError) as ctx:
                    archive_utils.safe_read_archive(p)
                self.assertIn("size limit", str(ctx.exception).lower())
        finally:
            archive_utils.MAX_EXTRACTED_SIZE = orig


class SafeExtractTests(unittest.TestCase):
    def test_benign_extract(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            p = td / "ok.tar.gz"
            _make_tar([("a.txt", b"hello", {}), ("sub/b.txt", b"world", {})], p)
            dest = td / "out"
            archive_utils.safe_extract(p, dest)
            self.assertEqual((dest / "a.txt").read_bytes(), b"hello")
            self.assertEqual((dest / "sub" / "b.txt").read_bytes(), b"world")

    def test_extract_rejects_traversal(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            p = td / "bad.tar.gz"
            _make_tar([("../escape.txt", b"x", {})], p)
            dest = td / "out"
            with self.assertRaises(archive_utils.UnsafeArchiveError):
                archive_utils.safe_extract(p, dest)
            # Nothing should have been written outside dest.
            self.assertFalse((td / "escape.txt").exists())

    def test_installed_size(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            p = td / "ok.tar.gz"
            _make_tar([("a.txt", b"hello", {}), ("b.txt", b"world!", {})], p)
            self.assertEqual(archive_utils.installed_size(p), len(b"hello") + len(b"world!"))


class UnknownFormatTests(unittest.TestCase):
    def test_unknown_format_raises(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.xyz"
            p.write_bytes(b"not an archive")
            with self.assertRaises(ValueError):
                archive_utils.safe_read_archive(p)


if __name__ == "__main__":
    unittest.main()
