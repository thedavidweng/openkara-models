#!/usr/bin/env python3
"""Generate runtime artifact catalog spec entries from the source lock + built archives.

For each runtime archive in ``ort/packages/``, produces a catalog spec entry
conforming to the ``runtime_artifact`` definition in
``catalog/schema-release-v1.json``. The entry records:

  - target_triple, arch, os (parsed from the archive name)
  - runtime.version (ORT tag), ort_c_api_level (from source lock)
  - execution_providers (from source lock target config)
  - companion_files (from source lock target config)
  - deployment_target (from source lock target config)
  - supported_model_artifact_ids (from the catalog manifest's compatibility
    matrix, or explicit --compatible-models)
  - toolchain (image, compiler, cmake_flags, submodule_state, deployment_target)
  - _digest (archive SHA-256 + size + per-file digests from the build manifest)
  - supply_chain (refs to the in-archive SBOM + provenance)

Outputs a JSON fragment that can be merged into a catalog release spec's
``artifacts.runtimes`` array by ``generate_catalog_release.py``.

Usage::

    python scripts/generate_runtime_catalog_entries.py \\
        --release 2026-07-20-001 \\
        --output catalog/specs/runtime-entries.json

    # With explicit compatible models:
    python scripts/generate_runtime_catalog_entries.py \\
        --compatible-models htdemucs.balanced.fp32.onnx htdemucs_ft.balanced.fp32.onnx \\
        --output runtime-entries.json
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
SPECS_DIR = ROOT / "catalog" / "specs"
RELEASES_DIR = ROOT / "catalog" / "releases"
GENERATOR_VERSION = "openkara.runtime-catalog-entries/v1"


def _sha256_file(path: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return size, h.hexdigest()


def _read_archive(archive: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    if archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    f = tar.extractfile(member)
                    if f:
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
        raise ValueError(f"unknown archive format: {archive.name}")
    return files


def _parse_target(archive_name: str) -> tuple[str, str, str, bool]:
    """Return (target_triple, arch, os, is_reduced) from an archive name."""
    base = archive_name
    if base.endswith(".tar.gz"):
        base = base[:-7]
    elif base.endswith(".zip"):
        base = base[:-4]
    segments = base.split("-openkara-")
    if len(segments) != 2:
        raise ValueError(f"cannot parse target from {archive_name}")
    target = segments[1]
    is_reduced = target.endswith("-reduced")
    if is_reduced:
        target = target[: -len("-reduced")]
    if "darwin" in target:
        os_name = "macos"
    elif "linux" in target:
        os_name = "linux"
    elif "windows" in target:
        os_name = "windows"
    else:
        raise ValueError(f"unknown OS in target: {target}")
    if target.startswith("aarch64"):
        arch = "aarch64"
    elif target.startswith("x86_64"):
        arch = "x86_64"
    else:
        raise ValueError(f"unknown arch in target: {target}")
    return target, arch, os_name, is_reduced


def _resolve_compatible_models(release_id: str | None, target: str) -> list[str]:
    """Resolve model artifact IDs compatible with this target from the catalog."""
    if not release_id:
        return []
    # Look at the release manifest's compatibility matrix.
    manifest_path = RELEASES_DIR / f"{release_id}.json"
    if not manifest_path.is_file():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    compat = manifest.get("compatibility", [])
    # Find runtime artifact IDs for this target, then their compatible models.
    runtime_ids = {
        e["runtime_artifact_id"] for e in compat
        if e.get("target_triple") == target
    }
    if not runtime_ids:
        return []
    model_ids = {
        e["model_artifact_id"] for e in compat
        if e["runtime_artifact_id"] in runtime_ids
    }
    return sorted(model_ids)


def _build_entry(
    archive: Path, lock: dict[str, Any], release_id: str | None,
    compatible_models: list[str] | None,
) -> dict[str, Any]:
    target, arch, os_name, is_reduced = _parse_target(archive.name)
    target_config = lock["targets"][target]
    files = _read_archive(archive)
    if "build-manifest.json" not in files:
        raise FileNotFoundError(f"{archive.name}: missing build-manifest.json")
    manifest = json.loads(files["build-manifest.json"])
    build_meta = manifest.get("build", {})

    archive_size, archive_sha = _sha256_file(archive)

    # Per-file digests from the build manifest.
    file_digests = manifest.get("files", {})

    # Supply chain refs from the build manifest.
    sc = manifest.get("supply_chain", {})

    # Compatible models: explicit override or catalog resolution.
    if compatible_models is not None:
        supported = compatible_models
    else:
        supported = _resolve_compatible_models(release_id, target)

    # Toolchain from source lock + build manifest.
    toolchain = {
        "image": _runner_image(target, lock),
        "compiler": _compiler_name(target, lock),
        "cmake_flags": build_meta.get("cmake_args", target_config["cmake_args"]),
        "submodule_state": build_meta.get("submodule_state", {}),
        "deployment_target": target_config.get("deployment_target") or "n/a",
    }

    artifact_id = f"onnxruntime-{lock['upstream']['tag'].lstrip('v')}-openkara-{target}"
    if is_reduced:
        artifact_id += "-reduced"

    entry = {
        "kind": "runtime",
        "artifact_id": artifact_id,
        "filename": archive.name,
        "download_url": _download_url(artifact_id, archive.name, release_id, lock),
        "target_triple": target,
        "arch": arch,
        "os": os_name,
        "runtime": {
            "version": lock["upstream"]["tag"],
            "ort_c_api_level": str(lock["c_api_level"]["ort_api_version"]),
            "execution_providers": target_config["execution_providers"],
            "companion_files": target_config.get("companion_libraries", []),
            "deployment_target": target_config.get("deployment_target") or "n/a",
            "supported_model_artifact_ids": supported,
            "reduced_build": is_reduced,
            "ops_config_sha256": build_meta.get("ops_config_sha256"),
        },
        "toolchain": toolchain,
        "_digest": {
            "mode": "predeclared",
            "archive_digest": archive_sha,
            "byte_size": archive_size,
            "extracted_file_digests": file_digests,
        },
        "supply_chain": {
            "sbom": {
                "kind": "spdx-json",
                "path": sc.get("sbom", {}).get("path", "sbom.spdx.json"),
                "size": sc.get("sbom", {}).get("size"),
                "sha256": sc.get("sbom", {}).get("sha256"),
                "digest_algorithm": "sha256",
            },
            "provenance": {
                "kind": "source-build-provenance",
                "path": sc.get("provenance", {}).get("path", "provenance.json"),
                "size": sc.get("provenance", {}).get("size"),
                "sha256": sc.get("provenance", {}).get("sha256"),
                "digest_algorithm": "sha256",
            },
        },
        "deprecation": {
            "deprecated": False,
            "replacement_artifact_id": None,
            "sunset_generation": None,
        },
    }
    return entry


def _runner_image(target: str, lock: dict[str, Any]) -> str:
    return lock["targets"][target].get("runner", "unknown")


def _compiler_name(target: str, lock: dict[str, Any]) -> str:
    if "darwin" in target:
        return f"clang (Xcode {lock['toolchain'].get('xcode_version', '?')})"
    if "linux" in target:
        return f"gcc {lock['toolchain'].get('gcc_version', '?')}"
    if "windows" in target:
        return f"MSVC (VS {lock['toolchain'].get('visual_studio_version', '?')})"
    return "unknown"


def _download_url(artifact_id: str, filename: str, release_id: str | None,
                  lock: dict[str, Any]) -> str:
    if release_id:
        return f"https://github.com/thedavidweng/openkara-models/releases/download/infra-{release_id}/{filename}"
    return f"https://github.com/thedavidweng/openkara-models/releases/download/ort-{lock['upstream']['tag']}/{filename}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate runtime catalog spec entries.")
    parser.add_argument("--release", type=str, default=None,
                        help="Release ID for download URL + compatibility resolution.")
    parser.add_argument("--compatible-models", nargs="*", default=None,
                        help="Explicit model artifact IDs this runtime supports.")
    parser.add_argument("--output", type=Path, default=Path("runtime-entries.json"),
                        help="Output JSON path.")
    args = parser.parse_args()

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    archives = sorted(PACKAGES_DIR.glob("*.tar.gz")) + sorted(PACKAGES_DIR.glob("*.zip"))
    if not archives:
        print("No archives found in ort/packages/", file=sys.stderr)
        return 1

    entries: list[dict[str, Any]] = []
    for archive in archives:
        print(f"Processing {archive.name}...")
        entry = _build_entry(archive, lock, args.release, args.compatible_models)
        entries.append(entry)
        print(f"  target: {entry['target_triple']}  arch: {entry['arch']}  os: {entry['os']}")
        print(f"  artifact_id: {entry['artifact_id']}")
        print(f"  size: {entry['_digest']['byte_size']}  sha256: {entry['_digest']['archive_digest'][:12]}...")

    doc = {
        "schema_version": GENERATOR_VERSION,
        "release_id": args.release,
        "runtimes": entries,
    }
    args.output.write_text(json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"\nOK: {len(entries)} runtime entries written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
