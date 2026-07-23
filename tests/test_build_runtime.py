"""Tests for build_runtime.py argument parsing and reproducibility features.

These tests do not run the actual build (which requires a full ORT source
checkout and CMake toolchain). They verify the argument parser, the --locked
flag, and the archive creation functions.
"""

from __future__ import annotations

import io
import json
import sys
import tarfile
import tempfile
import unittest
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import build_runtime as br  # noqa: E402


class LockedFlagTests(unittest.TestCase):
    def test_locked_flag_is_parsed(self):
        """The --locked flag must be accepted by the argument parser."""
        with mock.patch.object(sys, "argv", [
            "build_runtime.py", "--target", "aarch64-apple-darwin", "--locked",
        ]):
            with mock.patch.object(br, "_load_lock", return_value={
                "upstream": {"commit_sha": "a" * 40, "tag": "v1.27.1", "repo": "microsoft/onnxruntime"},
                "submodules": {},
                "targets": {"aarch64-apple-darwin": {}},
            }):
                with mock.patch.object(br, "_clone_source"):
                    with mock.patch.object(br, "_verify_upstream_commit", return_value=True):
                        with mock.patch.object(br, "_verify_submodules"):
                            with mock.patch.object(br, "_build_target", return_value={}):
                                with mock.patch.object(br, "_package_target", return_value=Path("/tmp/fake.tar.gz")):
                                    with mock.patch.object(br, "_sha256_file", return_value=(0, "0" * 64)):
                                        with redirect_stdout(io.StringIO()):
                                            code = br.main()
        self.assertEqual(code, 0)

    def test_locked_flag_verifies_upstream_commit(self):
        """--locked must call _verify_upstream_commit and fail if it returns False."""
        fake_source = Path("/fake/source")
        with mock.patch.object(sys, "argv", [
            "build_runtime.py", "--target", "aarch64-apple-darwin",
            "--locked", "--skip-clone", "--source-dir", str(fake_source),
        ]):
            with mock.patch.object(br, "_load_lock", return_value={
                "upstream": {"commit_sha": "a" * 40, "tag": "v1.27.1", "repo": "microsoft/onnxruntime"},
                "submodules": {},
                "targets": {"aarch64-apple-darwin": {}},
            }):
                with mock.patch.object(Path, "exists", return_value=True):
                    with mock.patch.object(br, "_verify_upstream_commit", return_value=False) as mock_verify:
                        with mock.patch.object(br, "_verify_submodules") as mock_sub:
                            out = io.StringIO()
                            err = io.StringIO()
                            with redirect_stdout(out), redirect_stderr(err):
                                code = br.main()
        self.assertNotEqual(code, 0)
        mock_verify.assert_called_once()
        mock_sub.assert_not_called()

    def test_no_locked_flag_skips_verification(self):
        """Without --locked, verification is skipped."""
        fake_source = Path("/fake/source")
        with mock.patch.object(sys, "argv", [
            "build_runtime.py", "--target", "aarch64-apple-darwin",
            "--skip-clone", "--source-dir", str(fake_source),
        ]):
            with mock.patch.object(br, "_load_lock", return_value={
                "upstream": {"commit_sha": "a" * 40, "tag": "v1.27.1", "repo": "microsoft/onnxruntime"},
                "submodules": {},
                "targets": {"aarch64-apple-darwin": {}},
            }):
                with mock.patch.object(Path, "exists", return_value=True):
                    with mock.patch.object(br, "_verify_upstream_commit", return_value=True) as mock_verify:
                        with mock.patch.object(br, "_build_target", return_value={}):
                            with mock.patch.object(br, "_package_target", return_value=Path("/tmp/fake.tar.gz")):
                                with mock.patch.object(br, "_sha256_file", return_value=(0, "0" * 64)):
                                    with redirect_stdout(io.StringIO()):
                                        code = br.main()
        self.assertEqual(code, 0)
        mock_verify.assert_not_called()


class ReproducibilityTests(unittest.TestCase):
    def test_build_metadata_has_no_build_host(self):
        """_build_target must not include build_host in the returned metadata."""
        lock = {
            "upstream": {"commit_sha": "a" * 40, "tag": "v1.27.1", "repo": "microsoft/onnxruntime"},
            "submodules": {},
            "targets": {"aarch64-apple-darwin": {"cmake_args": []}},
            "toolchain": {"python_version": "3.11", "cmake_minimum_version": "3.28",
                          "xcode_version": "16.0", "gcc_version": "13",
                          "visual_studio_version": "2022"},
        }
        with mock.patch.object(br, "_run"):
            with mock.patch.object(br, "_assert_toolchain"):
                with mock.patch.object(br.ort_api_version, "assert_api_version", return_value=27):
                    with mock.patch.object(br, "_get_submodule_state", return_value={}):
                        with mock.patch.object(br.os, "cpu_count", return_value=4):
                            platform_mock = mock.MagicMock()
                            platform_mock.system = mock.Mock(return_value="Darwin")
                            platform_mock.release = mock.Mock(return_value="24.0")
                            platform_mock.machine = mock.Mock(return_value="arm64")
                            with mock.patch.object(br, "platform", platform_mock):
                                with tempfile.TemporaryDirectory() as td:
                                    source_dir = Path(td) / "source"
                                    source_dir.mkdir()
                                    build_dir = Path(td) / "build"
                                    meta = br._build_target(lock, "aarch64-apple-darwin",
                                                            source_dir, build_dir)
        self.assertNotIn("build_host", meta,
                         "build_host must not be in build metadata (reproducibility)")
        self.assertIn("build_os", meta)
        self.assertIn("build_arch", meta)
        self.assertEqual(meta["ort_api_version"], 27)

    def test_tar_archive_has_deterministic_timestamps(self):
        """_make_tar must produce archives with deterministic mtime for all entries."""
        with tempfile.TemporaryDirectory() as td:
            staging = Path(td) / "staging"
            staging.mkdir()
            (staging / "file1.txt").write_bytes(b"hello")
            (staging / "file2.txt").write_bytes(b"world")
            archive = Path(td) / "test.tar.gz"
            br._make_tar(archive, staging, 315532800)
            with tarfile.open(archive, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        self.assertEqual(member.mtime, 315532800,
                                         f"{member.name}: mtime must be deterministic")
                        self.assertEqual(member.uid, 0)
                        self.assertEqual(member.gid, 0)
                        self.assertEqual(member.uname, "")
                        self.assertEqual(member.gname, "")

    def test_zip_archive_has_deterministic_timestamps(self):
        """_make_zip must produce archives with fixed date_time for all entries."""
        with tempfile.TemporaryDirectory() as td:
            staging = Path(td) / "staging"
            staging.mkdir()
            (staging / "file1.txt").write_bytes(b"hello")
            (staging / "file2.txt").write_bytes(b"world")
            archive = Path(td) / "test.zip"
            br._make_zip(archive, staging, 315532800)
            with zipfile.ZipFile(archive, "r") as zf:
                for info in zf.infolist():
                    if not info.is_dir():
                        self.assertEqual(info.date_time, (1980, 1, 1, 0, 0, 0),
                                         f"{info.filename}: date_time must be deterministic")

    def test_tar_archives_are_byte_identical(self):
        """Regenerating a tar archive with the same content must produce
        byte-identical output (deterministic ordering + timestamps + gzip)."""
        with tempfile.TemporaryDirectory() as td:
            staging = Path(td) / "staging"
            staging.mkdir()
            (staging / "file1.txt").write_bytes(b"hello")
            (staging / "file2.txt").write_bytes(b"world")
            archive1 = Path(td) / "test1.tar.gz"
            archive2 = Path(td) / "test2.tar.gz"
            br._make_tar(archive1, staging, 315532800)
            br._make_tar(archive2, staging, 315532800)
            self.assertEqual(archive1.read_bytes(), archive2.read_bytes())

    def test_zip_archives_are_byte_identical(self):
        """Regenerating a zip archive with the same content must produce
        byte-identical output."""
        with tempfile.TemporaryDirectory() as td:
            staging = Path(td) / "staging"
            staging.mkdir()
            (staging / "file1.txt").write_bytes(b"hello")
            (staging / "file2.txt").write_bytes(b"world")
            archive1 = Path(td) / "test1.zip"
            archive2 = Path(td) / "test2.zip"
            br._make_zip(archive1, staging, 315532800)
            br._make_zip(archive2, staging, 315532800)
            self.assertEqual(archive1.read_bytes(), archive2.read_bytes())


class SubmoduleVerificationTests(unittest.TestCase):
    def test_verify_submodules_passes_on_match(self):
        lock = {"submodules": {"cmake/external/onnx": {"expected_sha": "a" * 40}}}
        with mock.patch.object(br, "_get_submodule_state", return_value={
            "cmake/external/onnx": "a" * 40,
        }):
            br._verify_submodules(lock, Path("/fake"))  # should not raise

    def test_verify_submodules_fails_on_mismatch(self):
        lock = {"submodules": {"cmake/external/onnx": {"expected_sha": "a" * 40}}}
        with mock.patch.object(br, "_get_submodule_state", return_value={
            "cmake/external/onnx": "b" * 40,
        }):
            with self.assertRaises(RuntimeError) as ctx:
                br._verify_submodules(lock, Path("/fake"))
            self.assertIn("submodule verification failed", str(ctx.exception))

    def test_verify_submodules_fails_on_missing(self):
        lock = {"submodules": {"cmake/external/onnx": {"expected_sha": "a" * 40}}}
        with mock.patch.object(br, "_get_submodule_state", return_value={}):
            with self.assertRaises(RuntimeError) as ctx:
                br._verify_submodules(lock, Path("/fake"))
            self.assertIn("submodule verification failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
