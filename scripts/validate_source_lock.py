#!/usr/bin/env python3
"""Validate the ORT source lock file structure and consistency.

Checks that:
  - All 5 required targets are present with correct target triples.
  - Each target has required fields (runner, cmake_args, artifact_name, etc.).
  - The upstream commit SHA is a 40-char hex string.
  - The C API level is declared.
  - Toolchain versions are declared.

Usage::

    python scripts/validate_source_lock.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "ort" / "source-lock.json"

REQUIRED_TARGETS = {
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "x86_64-pc-windows-msvc",
}

TARGET_REQUIRED_FIELDS = {
    "runner", "cmake_args", "artifact_name", "execution_providers",
    "archive_format", "companion_libraries", "deployment_target",
}


def validate_lock(lock: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    # Check lock version.
    if lock.get("lock_version") != "openkara.ort-source-lock/v1":
        errors.append(f"unexpected lock_version: {lock.get('lock_version')}")

    # Check upstream.
    upstream = lock.get("upstream", {})
    if not upstream.get("tag"):
        errors.append("upstream.tag missing")
    if not upstream.get("commit_sha"):
        errors.append("upstream.commit_sha missing")
    elif len(upstream["commit_sha"]) != 40:
        errors.append(f"upstream.commit_sha must be 40 hex chars, got {len(upstream['commit_sha'])}")
    if not upstream.get("repo"):
        errors.append("upstream.repo missing")

    # Check targets.
    targets = lock.get("targets", {})
    missing = REQUIRED_TARGETS - set(targets.keys())
    if missing:
        errors.append(f"missing targets: {missing}")
    extra = set(targets.keys()) - REQUIRED_TARGETS
    if extra:
        errors.append(f"unknown targets: {extra}")

    for target_name, target_config in targets.items():
        missing_fields = TARGET_REQUIRED_FIELDS - set(target_config.keys())
        if missing_fields:
            errors.append(f"{target_name}: missing fields: {missing_fields}")

        # Check artifact_name matches target OS.
        art = target_config.get("artifact_name", "")
        if "darwin" in target_name and not art.endswith(".dylib"):
            errors.append(f"{target_name}: artifact_name should end with .dylib, got {art}")
        elif "linux" in target_name and not art.endswith(".so"):
            errors.append(f"{target_name}: artifact_name should end with .so, got {art}")
        elif "windows" in target_name and not art.endswith(".dll"):
            errors.append(f"{target_name}: artifact_name should end with .dll, got {art}")

        # Check cmake_args has BUILD_SHARED_LIB=ON.
        cmake_args = target_config.get("cmake_args", [])
        if "-Donnxruntime_BUILD_SHARED_LIB=ON" not in cmake_args:
            errors.append(f"{target_name}: cmake_args must include -Donnxruntime_BUILD_SHARED_LIB=ON")

        # Check execution_providers includes cpu.
        eps = target_config.get("execution_providers", [])
        if "cpu" not in eps:
            errors.append(f"{target_name}: execution_providers must include 'cpu'")

    # Check C API level.
    c_api = lock.get("c_api_level", {})
    if "ort_api_version" not in c_api:
        errors.append("c_api_level.ort_api_version missing")

    # Check toolchain.
    toolchain = lock.get("toolchain", {})
    for field in ("cmake_minimum_version", "python_version"):
        if field not in toolchain:
            errors.append(f"toolchain.{field} missing")

    return errors


def main() -> int:
    if not LOCK_PATH.is_file():
        print(f"ERROR: source lock not found: {LOCK_PATH}", file=sys.stderr)
        return 1

    with LOCK_PATH.open("r", encoding="utf-8") as fh:
        lock = json.load(fh)

    errors = validate_lock(lock)
    if errors:
        print("ERROR: source lock validation failed:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1

    print(f"OK: source lock valid ({len(lock['targets'])} targets, ORT {lock['upstream']['tag']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
