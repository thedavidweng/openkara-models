"""Tests for the ORT source lock and build infrastructure (issue #19 PR 1)."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

LOCK_PATH = ROOT_DIR / "ort" / "source-lock.json"

REQUIRED_TARGETS = {
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "x86_64-pc-windows-msvc",
}


def _load_lock() -> dict:
    with LOCK_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


class SourceLockStructureTests(unittest.TestCase):
    def test_lock_file_exists(self):
        self.assertTrue(LOCK_PATH.is_file(), "ort/source-lock.json must exist")

    def test_lock_version(self):
        lock = _load_lock()
        self.assertEqual(lock["lock_version"], "openkara.ort-source-lock/v1")

    def test_all_five_targets_present(self):
        lock = _load_lock()
        self.assertEqual(set(lock["targets"].keys()), REQUIRED_TARGETS)

    def test_upstream_commit_sha_is_40_hex(self):
        lock = _load_lock()
        sha = lock["upstream"]["commit_sha"]
        self.assertEqual(len(sha), 40)
        int(sha, 16)  # must be valid hex

    def test_upstream_tag_present(self):
        lock = _load_lock()
        self.assertTrue(lock["upstream"]["tag"].startswith("v"))

    def test_c_api_level_declared(self):
        lock = _load_lock()
        self.assertIn("ort_api_version", lock["c_api_level"])
        self.assertIsInstance(lock["c_api_level"]["ort_api_version"], int)

    def test_toolchain_versions_declared(self):
        lock = _load_lock()
        for field in ("cmake_minimum_version", "python_version"):
            self.assertIn(field, lock["toolchain"])


class TargetConfigTests(unittest.TestCase):
    def test_each_target_has_shared_lib_enabled(self):
        lock = _load_lock()
        for target, config in lock["targets"].items():
            self.assertIn("-DBUILD_SHARED_LIB=ON", config["cmake_args"],
                          f"{target} must build shared lib")

    def test_each_target_has_cpu_provider(self):
        lock = _load_lock()
        for target, config in lock["targets"].items():
            self.assertIn("cpu", config["execution_providers"],
                          f"{target} must include CPU provider")

    def test_artifact_name_matches_os(self):
        lock = _load_lock()
        for target, config in lock["targets"].items():
            art = config["artifact_name"]
            if "darwin" in target:
                self.assertTrue(art.endswith(".dylib"), f"{target}: {art}")
            elif "linux" in target:
                self.assertTrue(art.endswith(".so"), f"{target}: {art}")
            elif "windows" in target:
                self.assertTrue(art.endswith(".dll"), f"{target}: {art}")

    def test_apple_silicon_has_coreml(self):
        lock = _load_lock()
        eps = lock["targets"]["aarch64-apple-darwin"]["execution_providers"]
        self.assertIn("coreml", eps)

    def test_windows_has_directml(self):
        lock = _load_lock()
        eps = lock["targets"]["x86_64-pc-windows-msvc"]["execution_providers"]
        self.assertIn("directml", eps)

    def test_linux_targets_have_xnnpack(self):
        lock = _load_lock()
        for target in ("x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"):
            eps = lock["targets"][target]["execution_providers"]
            self.assertIn("xnnpack", eps)

    def test_intel_macos_has_xnnpack(self):
        lock = _load_lock()
        eps = lock["targets"]["x86_64-apple-darwin"]["execution_providers"]
        self.assertIn("xnnpack", eps)

    def test_training_disabled_for_all_targets(self):
        lock = _load_lock()
        for target, config in lock["targets"].items():
            self.assertIn("-DORT_ENABLE_TRAINING=OFF", config["cmake_args"],
                          f"{target} must disable training")

    def test_unit_tests_disabled(self):
        lock = _load_lock()
        for target, config in lock["targets"].items():
            self.assertIn("-Donnxruntime_BUILD_UNIT_TESTS=OFF", config["cmake_args"],
                          f"{target} must disable unit tests")

    def test_macos_deployment_target_set(self):
        lock = _load_lock()
        for target in ("aarch64-apple-darwin", "x86_64-apple-darwin"):
            dt = lock["targets"][target]["deployment_target"]
            self.assertTrue(dt, f"{target} must set deployment_target")


class ValidateSourceLockTests(unittest.TestCase):
    def test_validate_source_lock_passes(self):
        r = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "validate_source_lock.py")],
            capture_output=True, text=True, cwd=str(ROOT_DIR),
        )
        self.assertEqual(r.returncode, 0, r.stderr)


class BuildScriptSyntaxTests(unittest.TestCase):
    def test_build_runtime_compiles(self):
        r = subprocess.run(
            [sys.executable, "-m", "compileall", "-q",
             str(SCRIPTS_DIR / "build_runtime.py"),
             str(SCRIPTS_DIR / "verify_runtime_package.py"),
             str(SCRIPTS_DIR / "validate_source_lock.py")],
            capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
