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
the previous verified archive from stable-v2.

Usage::

    python scripts/repackage_nuget_runtime.py --output packages
    python scripts/repackage_nuget_runtime.py --output packages --target x86_64-unknown-linux-gnu
    python scripts/repackage_nuget_runtime.py --validate-lock
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nuget_package  # noqa: E402
import ort_api_version  # noqa: E402
from build_runtime import _make_tar, _make_zip, _ort_commit_timestamp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "ort" / "nuget-lock.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FINGERPRINT_RE = re.compile(r"^[0-9A-Fa-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def load_lock(path: Path = LOCK_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_lock(lock: dict[str, Any]) -> list[str]:
    """Validate ort/nuget-lock.json structure, digests, signers, and targets."""
    errors: list[str] = []
    if lock.get("schema_version") != "openkara.ort-nuget-lock/v1":
        errors.append(f"unexpected schema_version: {lock.get('schema_version')!r}")

    upstream = lock.get("upstream", {})
    tag = upstream.get("tag", "")
    commit = upstream.get("commit_sha", "")
    if not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
        errors.append(f"upstream.tag must be vX.Y.Z, got {tag!r}")
    if not _COMMIT_RE.fullmatch(commit):
        errors.append(f"upstream.commit_sha must be 40 lowercase hex chars, got {commit!r}")
    if not upstream.get("project"):
        errors.append("upstream.project missing")

    pkg = lock.get("package", {})
    pkg_id = pkg.get("id", "")
    version = pkg.get("version", "")
    if pkg_id != "Microsoft.ML.OnnxRuntime":
        errors.append(f"unexpected package.id: {pkg_id!r}")
    if not _VERSION_RE.fullmatch(version):
        errors.append(f"package.version must be X.Y.Z, got {version!r}")
    elif tag and tag != f"v{version}":
        errors.append(f"package.version {version!r} disagrees with upstream.tag {tag!r}")
    expected_url = nuget_package.package_url(pkg_id, version) if pkg_id and version else ""
    if pkg.get("url") != expected_url:
        errors.append(f"package.url must be {expected_url!r}, got {pkg.get('url')!r}")
    if not isinstance(pkg.get("size"), int) or pkg["size"] <= 0:
        errors.append("package.size must be a positive integer")
    if not _SHA256_RE.fullmatch(str(pkg.get("sha256", ""))):
        errors.append("package.sha256 must be 64 lowercase hex chars")

    signers = pkg.get("signers", {})
    for key, expected_subject in (
        ("author", nuget_package.AUTHOR_SUBJECT),
        ("repository", nuget_package.REPOSITORY_SUBJECT),
    ):
        signer = signers.get(key)
        if not isinstance(signer, dict):
            errors.append(f"package.signers.{key} missing")
            continue
        if signer.get("subject") != expected_subject:
            errors.append(f"package.signers.{key}.subject mismatch")
        if not _FINGERPRINT_RE.fullmatch(str(signer.get("sha256_fingerprint", ""))):
            errors.append(f"package.signers.{key}.sha256_fingerprint must be 64 hex chars")

    members = lock.get("members", {})
    if not isinstance(members, dict) or not members:
        errors.append("members missing or empty")
    else:
        for name, info in members.items():
            if not isinstance(info, dict):
                errors.append(f"members.{name} must be an object")
                continue
            if not isinstance(info.get("size"), int) or info["size"] <= 0:
                errors.append(f"members.{name}.size must be a positive integer")
            if not _SHA256_RE.fullmatch(str(info.get("sha256", ""))):
                errors.append(f"members.{name}.sha256 must be 64 lowercase hex chars")

    common = lock.get("common", {})
    for key in ("LICENSE", "ThirdPartyNotices.txt"):
        source = common.get(key)
        if not source or source not in members:
            errors.append(f"common.{key} missing from members")
    include_prefix = common.get("include")
    if not include_prefix or f"{include_prefix}/onnxruntime_c_api.h" not in members:
        errors.append("common.include must point to pinned headers including onnxruntime_c_api.h")

    targets = lock.get("targets", {})
    required_targets = {
        "aarch64-apple-darwin",
        "aarch64-unknown-linux-gnu",
        "x86_64-pc-windows-msvc",
        "x86_64-unknown-linux-gnu",
    }
    missing_targets = required_targets - set(targets)
    if missing_targets:
        errors.append(f"missing NuGet targets: {sorted(missing_targets)}")
    for target, config in targets.items():
        if config.get("archive_format") not in ("tar.gz", "zip"):
            errors.append(f"targets.{target}.archive_format invalid")
        if not config.get("root"):
            errors.append(f"targets.{target}.root missing")
        files = config.get("files", {})
        if not files:
            errors.append(f"targets.{target}.files empty")
        for source in files.values():
            if source not in members:
                errors.append(f"targets.{target} member {source!r} missing from members")
    return errors


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
    headers = _headers(package, lock)
    api_version = ort_api_version.parse_header(headers["onnxruntime_c_api.h"].decode("utf-8"))
    for name, data in headers.items():
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
        "c_api_level": {"ort_api_version": api_version},
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
    parser.add_argument("--output", type=Path, help="Directory for the repackaged archives.")
    parser.add_argument("--target", action="append", help="Target triple (repeatable; default: every locked target).")
    parser.add_argument("--cache", type=Path, default=ROOT / "ort" / "nuget-cache", help="Download cache directory.")
    parser.add_argument("--headers", type=Path, help="Also write the C API headers as <dir>/include/onnxruntime/core/session/*.h.")
    parser.add_argument("--validate-lock", action="store_true", help="Validate ort/nuget-lock.json and exit.")
    args = parser.parse_args()

    lock = load_lock()
    if args.validate_lock:
        errors = validate_lock(lock)
        if errors:
            for e in errors:
                print(f"ERROR: {e}", file=sys.stderr)
            return 1
        print(f"OK: nuget lock valid ({len(lock['targets'])} targets, {lock['package']['id']} {lock['package']['version']})")
        return 0

    if not args.output:
        parser.error("--output is required unless --validate-lock is passed")

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
