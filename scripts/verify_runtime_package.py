#!/usr/bin/env python3
"""Verify a built ORT runtime package against the source lock.

Checks that:
  - The archive contains the expected runtime library and companion libraries.
  - Every file in the build manifest has the correct size and SHA-256.
  - The build manifest references the correct upstream commit and tag.
  - License and notice files are present.
  - The C API level matches the source lock.

Usage::

    python scripts/verify_runtime_package.py ort/packages/<archive>
    python scripts/verify_runtime_package.py --all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "ort" / "source-lock.json"
PACKAGES_DIR = ROOT / "ort" / "packages"


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _load_lock() -> dict[str, Any]:
    with LOCK_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _extract_archive(archive: Path) -> dict[str, bytes]:
    """Extract all files from a tar.gz or zip archive into memory."""
    files: dict[str, bytes] = {}
    if archive.suffix == ".gz" or archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    f = tar.extractfile(member)
                    if f:
                        # Strip leading ./ from tar member names
                        name = member.name
                        if name.startswith("./"):
                            name = name[2:]
                        files[name] = f.read()
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            for info in zf.infolist():
                if not info.is_dir():
                    files[info.filename] = zf.read(info)
    else:
        raise ValueError(f"unknown archive format: {archive.suffix}")
    return files


def _target_from_archive_name(name: str) -> str:
    """Extract target triple from archive name like onnxruntime-1.27.1-openkara-aarch64-apple-darwin.tar.gz"""
    parts = name.rsplit(".", 2)[0]  # strip .tar.gz or .zip
    segments = parts.split("-openkara-")
    if len(segments) != 2:
        raise ValueError(f"cannot parse target from archive name: {name}")
    return segments[1]


def verify_archive(archive: Path, lock: dict[str, Any]) -> list[str]:
    """Returns a list of error strings (empty = OK)."""
    errors: list[str] = []

    try:
        target = _target_from_archive_name(archive.name)
    except ValueError as e:
        return [str(e)]

    if target not in lock["targets"]:
        return [f"unknown target: {target}"]

    target_config = lock["targets"][target]
    expected_lib = target_config["artifact_name"]
    expected_companions = target_config.get("companion_libraries", [])

    try:
        files = _extract_archive(archive)
    except Exception as e:
        return [f"failed to extract archive: {e}"]

    # Check build manifest exists.
    if "build-manifest.json" not in files:
        errors.append("missing build-manifest.json")
        return errors

    manifest = json.loads(files["build-manifest.json"])

    # Verify upstream commit matches.
    if manifest.get("build", {}).get("commit_sha") != lock["upstream"]["commit_sha"]:
        errors.append(
            f"commit_sha mismatch: manifest={manifest.get('build', {}).get('commit_sha')} "
            f"lock={lock['upstream']['commit_sha']}"
        )

    if manifest.get("build", {}).get("tag") != lock["upstream"]["tag"]:
        errors.append(
            f"tag mismatch: manifest={manifest.get('build', {}).get('tag')} "
            f"lock={lock['upstream']['tag']}"
        )

    # Verify C API level.
    if manifest.get("c_api_level", {}).get("ort_api_version") != lock["c_api_level"]["ort_api_version"]:
        errors.append("C API level mismatch")

    # Verify all files in manifest have correct digests.
    for rel, expected in manifest.get("files", {}).items():
        if rel not in files:
            errors.append(f"manifest references missing file: {rel}")
            continue
        actual_sha = _sha256_bytes(files[rel])
        if actual_sha != expected["sha256"]:
            errors.append(f"sha256 mismatch for {rel}")
        if len(files[rel]) != expected["size"]:
            errors.append(f"size mismatch for {rel}")

    # Check expected library is present.
    lib_found = any(f.endswith(expected_lib) for f in files)
    if not lib_found:
        errors.append(f"missing runtime library: {expected_lib}")

    # Check companion libraries.
    for comp in expected_companions:
        comp_found = any(f.endswith(comp) for f in files)
        if not comp_found:
            errors.append(f"missing companion library: {comp}")

    # Check license/notice presence.
    has_license = any("LICENSE" in f.upper() for f in files)
    if not has_license:
        errors.append("missing LICENSE file")
    has_notice = any("NOTICE" in f.upper() or "THIRDPARTY" in f.upper() for f in files)
    if not has_notice:
        errors.append("missing NOTICE file")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify ORT runtime packages.")
    parser.add_argument("archive", nargs="?", type=Path, help="Archive to verify.")
    parser.add_argument("--all", action="store_true", help="Verify all archives in ort/packages/.")
    args = parser.parse_args()

    if not args.all and not args.archive:
        parser.error("provide an archive path or --all")

    lock = _load_lock()
    all_errors: list[tuple[str, list[str]]] = []

    if args.all:
        archives = sorted(PACKAGES_DIR.glob("*.tar.gz")) + sorted(PACKAGES_DIR.glob("*.zip"))
        if not archives:
            print("No archives found in ort/packages/", file=sys.stderr)
            return 1
        for archive in archives:
            errs = verify_archive(archive, lock)
            if errs:
                all_errors.append((str(archive), errs))
    else:
        if not args.archive.is_file():
            print(f"ERROR: archive not found: {args.archive}", file=sys.stderr)
            return 1
        errs = verify_archive(args.archive, lock)
        if errs:
            all_errors.append((str(args.archive), errs))

    if all_errors:
        for path, errs in all_errors:
            for e in errs:
                print(f"FAIL: {path}: {e}", file=sys.stderr)
        print(f"runtime-package: {sum(len(e) for _, e in all_errors)} error(s)", file=sys.stderr)
        return 1

    count = len(list(PACKAGES_DIR.glob("*.tar.gz")) + list(PACKAGES_DIR.glob("*.zip"))) if args.all else 1
    print(f"runtime-package: {count} archive(s) verified, 0 errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
