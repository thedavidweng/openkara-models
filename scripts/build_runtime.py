#!/usr/bin/env python3
"""Build reproducible target-specific ONNX Runtime distributions from source lock.

Reads ``ort/source-lock.json``, clones ORT at the pinned commit, configures CMake
with target-specific flags, builds the shared library, and packages a normalized
archive with licenses, notices, and a build manifest.

Usage::

    python scripts/build_runtime.py --target aarch64-apple-darwin
    python scripts/build_runtime.py --target x86_64-pc-windows-msvc
    python scripts/build_runtime.py --target aarch64-apple-darwin --skip-clone

The build is reproducible: the same source lock + toolchain produces the same
archive (modulo timestamps, which are fixed to the ORT commit date).

This script is the entry point for both local and CI builds. The CI workflow
calls it on native runners for each target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "ort" / "source-lock.json"
BUILD_DIR = ROOT / "ort" / "build"
SOURCE_DIR = ROOT / "ort" / "source"
PACKAGES_DIR = ROOT / "ort" / "packages"


def _load_lock() -> dict[str, Any]:
    with LOCK_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None,
         check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=check,
                          capture_output=False)


def _sha256_file(path: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return size, h.hexdigest()


def _clone_source(lock: dict[str, Any], target_dir: Path) -> None:
    upstream = lock["upstream"]
    repo = upstream["repo"]
    commit = upstream["commit_sha"]

    if target_dir.exists() and (target_dir / ".git").is_dir():
        print(f"  source dir exists, fetching {commit}...")
        _run(["git", "fetch", "origin", commit], cwd=target_dir)
        _run(["git", "checkout", commit], cwd=target_dir)
    else:
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", f"https://github.com/{repo}.git", str(target_dir)])
        _run(["git", "checkout", commit], cwd=target_dir)

    # Initialize and update submodules.
    _run(["git", "submodule", "update", "--init", "--recursive", "--depth", "1"],
         cwd=target_dir)


def _get_submodule_state(source_dir: Path) -> dict[str, str]:
    """Record the exact SHA of each submodule."""
    result = subprocess.run(
        ["git", "submodule", "status", "--recursive"],
        cwd=str(source_dir),
        capture_output=True,
        text=True,
        check=True,
    )
    submodules: dict[str, str] = {}
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split()
        sha = parts[0].lstrip("+-")
        path = parts[1]
        submodules[path] = sha
    return submodules


def _build_target(lock: dict[str, Any], target: str, source_dir: Path,
                  build_dir: Path) -> dict[str, Any]:
    target_config = lock["targets"][target]
    cmake_args = target_config["cmake_args"]

    # Configure.
    build_dir.mkdir(parents=True, exist_ok=True)
    configure_cmd = ["cmake", "-S", str(source_dir / "cmake"), "-B", str(build_dir),
                     *cmake_args]
    _run(configure_cmd)

    # Build.
    nproc = os.cpu_count() or 4
    build_cmd = ["cmake", "--build", str(build_dir), "--config", "Release",
                 "--parallel", str(nproc)]
    _run(build_cmd)

    # Collect build metadata.
    submodules = _get_submodule_state(source_dir)
    return {
        "target": target,
        "commit_sha": lock["upstream"]["commit_sha"],
        "tag": lock["upstream"]["tag"],
        "submodule_state": submodules,
        "cmake_args": cmake_args,
        "build_host": platform.node(),
        "build_os": f"{platform.system()} {platform.release()}",
        "build_arch": platform.machine(),
    }


def _package_target(lock: dict[str, Any], target: str, source_dir: Path,
                    build_dir: Path, build_meta: dict[str, Any]) -> Path:
    target_config = lock["targets"][target]
    artifact_name = target_config["artifact_name"]
    companion_libs = target_config.get("companion_libraries", [])
    archive_format = target_config["archive_format"]

    # Find the built library.
    lib_path = _find_library(build_dir, artifact_name)
    if lib_path is None:
        raise FileNotFoundError(f"built library {artifact_name} not found under {build_dir}")

    # Create staging directory.
    staging = BUILD_DIR / "staging" / target
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    # Copy runtime library.
    shutil.copy2(lib_path, staging / lib_path.name)

    # Copy companion libraries.
    for comp in companion_libs:
        comp_path = _find_library(build_dir, comp)
        if comp_path:
            shutil.copy2(comp_path, staging / comp_path.name)

    # Copy licenses and notices from source.
    license_file = source_dir / "LICENSE"
    if license_file.is_file():
        shutil.copy2(license_file, staging / "LICENSE.onnxruntime")
    # ORT uses ThirdPartyNotices.txt, not NOTICE
    notice_file = source_dir / "NOTICE"
    if not notice_file.is_file():
        notice_file = source_dir / "ThirdPartyNotices.txt"
    if notice_file.is_file():
        shutil.copy2(notice_file, staging / "NOTICE.onnxruntime")
    # Third-party licenses
    tp_licenses = source_dir / "cmake" / "external" / "LICENSES"
    if tp_licenses.is_dir():
        shutil.copytree(tp_licenses, staging / "third-party-licenses", dirs_exist_ok=True)

    # Write build manifest.
    manifest = {
        "schema_version": "openkara.ort-build-manifest/v1",
        "target": target,
        "upstream": lock["upstream"],
        "build": build_meta,
        "toolchain": lock["toolchain"],
        "c_api_level": lock["c_api_level"],
        "files": {},
    }

    # Compute digests for all staged files.
    for f in sorted(staging.rglob("*")):
        if f.is_file():
            rel = f.relative_to(staging).as_posix()
            size, sha = _sha256_file(f)
            manifest["files"][rel] = {"size": size, "sha256": sha}

    with (staging / "build-manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")

    # Create archive.
    PACKAGES_DIR.mkdir(parents=True, exist_ok=True)
    version = lock["upstream"]["tag"].lstrip("v")
    archive_base = f"onnxruntime-{version}-openkara-{target}"

    if archive_format == "tar.gz":
        archive_path = PACKAGES_DIR / f"{archive_base}.tar.gz"
        _run(["tar", "czf", str(archive_path), "-C", str(staging), "."])
    elif archive_format == "zip":
        archive_path = PACKAGES_DIR / f"{archive_base}.zip"
        _run(["zip", "-r", str(archive_path), "."], cwd=staging)
    else:
        raise ValueError(f"unknown archive format: {archive_format}")

    return archive_path


def _find_library(build_dir: Path, name: str) -> Path | None:
    """Search for a library file in the build directory."""
    for pattern in [f"**/{name}", f"**/Release/{name}", f"**/{name}.dylib",
                    f"**/{name}.so", f"**/{name}.dll"]:
        matches = list(build_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ORT from source lock.")
    parser.add_argument("--target", required=True,
                        choices=["aarch64-apple-darwin", "x86_64-apple-darwin",
                                 "x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu",
                                 "x86_64-pc-windows-msvc"],
                        help="Target triple to build.")
    parser.add_argument("--skip-clone", action="store_true",
                        help="Skip cloning; use existing ort/source/ checkout.")
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR,
                        help="Source directory (default: ort/source/).")
    parser.add_argument("--build-dir", type=Path, default=None,
                        help="Build directory (default: ort/build/<target>/).")
    args = parser.parse_args()

    lock = _load_lock()
    source_dir = args.source_dir
    build_dir = args.build_dir or (BUILD_DIR / args.target)

    if not args.skip_clone:
        print(f"Cloning ORT {lock['upstream']['tag']} ({lock['upstream']['commit_sha'][:12]})...")
        _clone_source(lock, source_dir)
    elif not source_dir.exists():
        print(f"ERROR: source dir not found: {source_dir}", file=sys.stderr)
        return 1

    print(f"Building for {args.target}...")
    build_meta = _build_target(lock, args.target, source_dir, build_dir)

    print(f"Packaging {args.target}...")
    archive_path = _package_target(lock, args.target, source_dir, build_dir, build_meta)

    archive_size, archive_sha = _sha256_file(archive_path)
    print(f"OK: {args.target}")
    print(f"  archive: {archive_path}")
    print(f"  size:    {archive_size} bytes")
    print(f"  sha256:  {archive_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
