#!/usr/bin/env python3
"""Repackage Microsoft's signed ONNX Runtime NuGet package into catalog archives.

Downloads the package pinned in ``ort/nuget-lock.json`` from nuget.org,
requires its pinned size and SHA-256, runs ``dotnet nuget verify --all``
(Microsoft Corporation author signature plus the nuget.org repository
countersignature), checks every used member against the lock, and writes one
deterministic archive per target in the canonical layout::

    onnxruntime-<platform>-<version>/
        LICENSE
        ThirdPartyNotices.txt
        build-manifest.json
        include/<C API headers>
        lib/<runtime library, companions, and version symlinks>

The x86_64-apple-darwin runtime is not in the NuGet package; the catalog keeps
the archive this repository built from source.

Usage::

    python scripts/repackage_nuget_runtime.py --output packages
    python scripts/repackage_nuget_runtime.py --output packages --target x86_64-unknown-linux-gnu
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nuget_package  # noqa: E402
import ort_api_version  # noqa: E402
from build_runtime import _make_tar, _make_zip, _ort_commit_timestamp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "ort" / "nuget-lock.json"


def load_lock(path: Path = LOCK_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def archive_name(lock: dict, target: str) -> str:
    version = lock["package"]["version"]
    extension = lock["targets"][target]["archive_format"]
    return f"onnxruntime-{version}-openkara-{target}.{extension}"


def read_member(package: zipfile.ZipFile, lock: dict, name: str) -> bytes:
    """Read one package member and require its locked size and SHA-256."""
    expected = lock["members"].get(name)
    if expected is None:
        raise ValueError(f"{name}: not pinned in ort/nuget-lock.json")
    data = package.read(name)
    actual = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    if actual != {"size": expected["size"], "sha256": expected["sha256"]}:
        raise ValueError(f"{name}: expected {expected['size']} bytes sha256 {expected['sha256']}, got {actual['size']} bytes sha256 {actual['sha256']}")
    team = expected.get("apple_team_id")
    if team and f"Developer ID Application: Microsoft Corporation ({team})".encode() not in data:
        raise ValueError(f"{name}: missing Microsoft Developer ID signature ({team})")
    return data


def _headers(package: zipfile.ZipFile, lock: dict) -> dict[str, bytes]:
    include = lock["common"]["include"]
    cache: dict[str, bytes] = {}

    def read(name: str) -> bytes:
        if name not in cache:
            cache[name] = read_member(package, lock, f"{include}/{name}")
        return cache[name]

    names = ort_api_version.required_headers(read("onnxruntime_c_api.h"), read)
    return {name: read(name) for name in names}


def stage_target(package: zipfile.ZipFile, lock: dict, target: str, staging: Path) -> Path:
    """Write the canonical archive tree for ``target`` under ``staging``."""
    config = lock["targets"][target]
    payload = staging / config["root"]
    files: dict[str, tuple[bytes, str]] = {}
    for dest, source in lock["common"].items():
        if dest != "include":
            files[dest] = (read_member(package, lock, source), source)
    include = lock["common"]["include"]
    for name, data in _headers(package, lock).items():
        files[f"include/{name}"] = (data, f"{include}/{name}")
    for dest, source in config["files"].items():
        files[dest] = (read_member(package, lock, source), source)

    for rel, (data, _source) in files.items():
        path = payload / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    for rel, link in config["symlinks"].items():
        (payload / rel).symlink_to(link)

    manifest = {
        "schema_version": "openkara.ort-nuget-repackage/v1",
        "target": target,
        "upstream": lock["upstream"],
        "nuget": {key: lock["package"][key] for key in ("id", "version", "url", "size", "sha256", "signers")},
        "files": {
            rel: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "source": source}
            for rel, (data, source) in sorted(files.items())
        },
        "symlinks": dict(sorted(config["symlinks"].items())),
    }
    (payload / "build-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def repackage(nupkg: Path, lock: dict, targets: list[str], output: Path) -> dict[str, Path]:
    output.mkdir(parents=True, exist_ok=True)
    archives = {}
    with zipfile.ZipFile(nupkg) as package, tempfile.TemporaryDirectory() as tmp:
        for target in targets:
            staging = Path(tmp) / target
            stage_target(package, lock, target, staging)
            archive = output / archive_name(lock, target)
            if lock["targets"][target]["archive_format"] == "zip":
                _make_zip(archive, staging, _ort_commit_timestamp(lock))
            else:
                _make_tar(archive, staging, _ort_commit_timestamp(lock))
            archives[target] = archive
    return archives


def fetch_verified_package(lock: dict, cache: Path, verify_signatures: bool = True) -> Path:
    pkg = lock["package"]
    nupkg = nuget_package.download(pkg["url"], cache / f"{pkg['id']}.{pkg['version']}.nupkg", pkg["size"], pkg["sha256"])
    if verify_signatures:
        nuget_package.verify_signatures(nupkg, pkg["id"], pkg["version"])
    return nupkg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="Directory for the repackaged archives.")
    parser.add_argument("--target", action="append", help="Target triple (repeatable; default: every locked target).")
    parser.add_argument("--cache", type=Path, default=ROOT / "ort" / "nuget-cache", help="Download cache directory.")
    parser.add_argument("--headers", type=Path, help="Also write the C API headers as <dir>/include/onnxruntime/core/session/*.h.")
    args = parser.parse_args()

    lock = load_lock()
    targets = args.target or sorted(lock["targets"])
    unknown = sorted(set(targets) - set(lock["targets"]))
    if unknown:
        parser.error(f"unknown target(s): {', '.join(unknown)}")

    nupkg = fetch_verified_package(lock, args.cache)
    print(f"Verified {lock['package']['id']} {lock['package']['version']} (Microsoft author + nuget.org repository signatures)")
    for target, archive in repackage(nupkg, lock, targets, args.output).items():
        size, sha256 = nuget_package.sha256_file(archive)
        print(f"{target}: {archive.name} {size} {sha256}")
    if args.headers:
        session = args.headers / Path(ort_api_version.HEADER_REL).parent
        session.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(nupkg) as package:
            for name, data in _headers(package, lock).items():
                (session / name).write_bytes(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
