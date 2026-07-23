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

    def test_c_api_level_not_manually_maintained(self):
        """The lock must NOT carry a manually maintained ort_api_version; it
        is derived from the pinned ORT header at build time."""
        lock = _load_lock()
        self.assertNotIn("ort_api_version", lock.get("c_api_level", {}),
                         "ort_api_version must not be in the source lock; it is "
                         "parsed from the pinned header at build time")
        self.assertNotIn("rust_ort_crate_version", lock.get("c_api_level", {}),
                         "rust_ort_crate_version must not be in the infrastructure "
                         "source lock; the app owns its Rust binding version")

    def test_toolchain_versions_declared(self):
        lock = _load_lock()
        for field in ("cmake_minimum_version", "python_version"):
            self.assertIn(field, lock["toolchain"])


class TargetConfigTests(unittest.TestCase):
    def test_each_target_has_shared_lib_enabled(self):
        lock = _load_lock()
        for target, config in lock["targets"].items():
            self.assertIn("-Donnxruntime_BUILD_SHARED_LIB=ON", config["cmake_args"],
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
        self.assertIn("cpu", eps)
        self.assertIn("directml", eps)

    def test_windows_uses_dynamic_crt(self):
        """DirectML's NuGet package is built with MultiThreadedDLL; the ORT
        build must use the same CRT to link correctly."""
        lock = _load_lock()
        cmake_args = lock["targets"]["x86_64-pc-windows-msvc"]["cmake_args"]
        self.assertIn("-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL", cmake_args)
        self.assertIn("-Donnxruntime_USE_DML=ON", cmake_args)

    def test_windows_companion_libraries_include_directml(self):
        """DirectML.dll must be shipped as a companion library."""
        lock = _load_lock()
        companions = lock["targets"]["x86_64-pc-windows-msvc"]["companion_libraries"]
        self.assertIn("DirectML.dll", companions)

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


class SubmodulePinningTests(unittest.TestCase):
    def test_submodules_have_expected_sha(self):
        """Every submodule must have an expected_sha (40-char hex), not a
        mutable ref like 'main'."""
        lock = _load_lock()
        for path, info in lock.get("submodules", {}).items():
            self.assertIn("expected_sha", info, f"{path} must have expected_sha")
            sha = info["expected_sha"]
            self.assertEqual(len(sha), 40, f"{path}: expected_sha must be 40 chars")
            int(sha, 16)  # must be valid hex

    def test_no_submodule_uses_expected_ref(self):
        """Submodules must not use mutable expected_ref fields."""
        lock = _load_lock()
        for path, info in lock.get("submodules", {}).items():
            self.assertNotIn("expected_ref", info,
                             f"{path}: use expected_sha (pinned commit), not "
                             f"expected_ref (mutable branch/tag)")

    def test_deps_entries_have_sha1(self):
        """Every deps entry must have a sha1 hash for archive verification."""
        lock = _load_lock()
        deps = lock.get("deps", {}).get("entries", {})
        self.assertTrue(deps, "deps.entries must not be empty")
        for name, info in deps.items():
            self.assertIn("sha1", info, f"deps.{name} must have sha1")
            self.assertIn("url", info, f"deps.{name} must have url")


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
