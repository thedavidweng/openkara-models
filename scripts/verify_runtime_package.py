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
SOURCE_DIR = ROOT / "ort" / "source"

sys.path.insert(0, str(ROOT / "scripts"))
import ort_api_version  # noqa: E402
import archive_utils  # noqa: E402


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _load_lock() -> dict[str, Any]:
    with LOCK_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _extract_archive(archive: Path) -> dict[str, bytes]:
    """Extract all files from a tar.gz or zip archive into memory.

    Uses archive_utils.safe_read_archive which rejects unsafe members
    (absolute paths, traversal, symlinks/hardlinks that escape, duplicate
    normalized paths, excessive member counts, excessive extracted size).
    """
    return archive_utils.safe_read_archive(archive)


def _target_from_archive_name(name: str) -> tuple[str, bool]:
    """Extract (target_triple, is_reduced) from an archive name.

    Accepts names like:
      onnxruntime-1.27.1-openkara-aarch64-apple-darwin.tar.gz
      onnxruntime-1.27.1-openkara-aarch64-apple-darwin-reduced.tar.gz
    """
    base = name
    if base.endswith(".tar.gz"):
        base = base[:-7]
    elif base.endswith(".zip"):
        base = base[:-4]
    segments = base.split("-openkara-")
    if len(segments) != 2:
        raise ValueError(f"cannot parse target from archive name: {name}")
    target = segments[1]
    is_reduced = target.endswith("-reduced")
    if is_reduced:
        target = target[: -len("-reduced")]
    return target, is_reduced


def verify_archive(archive: Path, lock: dict[str, Any]) -> list[str]:
    """Returns a list of error strings (empty = OK)."""
    errors: list[str] = []

    try:
        target, is_reduced = _target_from_archive_name(archive.name)
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

    # Verify C API level: parse the pinned source header and compare it to
    # the value recorded in the package build manifest. The source lock no
    # longer carries a manually maintained ort_api_version; the header is the
    # source of truth.
    manifest_api = manifest.get("c_api_level", {}).get("ort_api_version")
    tag = lock["upstream"]["tag"]
    if SOURCE_DIR.is_dir():
        try:
            header_api = ort_api_version.parse_ort_api_version(SOURCE_DIR)
        except (FileNotFoundError, ValueError) as e:
            errors.append(f"failed to parse ORT_API_VERSION from pinned source header: {e}")
            header_api = None
        if header_api is not None:
            try:
                required = ort_api_version.required_api_version_for_tag(tag)
            except ValueError as e:
                errors.append(str(e))
                required = None
            if required is not None and header_api != required:
                errors.append(
                    f"pinned source ORT_API_VERSION={header_api} but required "
                    f"{required} for {tag}"
                )
            if manifest_api is not None and manifest_api != header_api:
                errors.append(
                    f"C API level mismatch: manifest ort_api_version={manifest_api} "
                    f"but pinned source header ORT_API_VERSION={header_api}"
                )
            if manifest_api is None:
                errors.append("build manifest missing c_api_level.ort_api_version")
    else:
        # Source not checked out locally (e.g. CI verify step on a different
        # runner). Fall back to the required value for the tag and require the
        # manifest to match it.
        try:
            required = ort_api_version.required_api_version_for_tag(tag)
        except ValueError as e:
            errors.append(str(e))
            required = None
        if required is not None and manifest_api != required:
            errors.append(
                f"C API level mismatch: manifest ort_api_version={manifest_api} "
                f"but required {required} for {tag} (pinned source not available "
                f"to parse header)"
            )

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

    # Reproducibility: build manifest must not contain build_host (hostname
    # varies per build host and breaks reproducibility).
    build_meta = manifest.get("build", {})
    if "build_host" in build_meta:
        errors.append(
            "build-manifest.json contains build_host field; remove it for "
            "reproducibility (hostname varies per build host)"
        )

    # Submodule verification: every submodule in the lock must be present in
    # the build manifest with the expected SHA.
    expected_submodules = lock.get("submodules", {})
    actual_submodules = build_meta.get("submodule_state", {})
    for path, info in expected_submodules.items():
        expected_sha = info["expected_sha"]
        actual_sha = actual_submodules.get(path)
        if actual_sha is None:
            errors.append(f"submodule {path} missing from build manifest")
        elif actual_sha != expected_sha:
            errors.append(
                f"submodule {path} SHA mismatch (manifest={actual_sha} "
                f"lock={expected_sha})"
            )

    # Reduced-build verification: the archive name and build manifest must
    # agree on whether this is a reduced build. A reduced archive's manifest
    # must record reduced_build=true and an ops_config_sha256 matching the
    # committed ort/required-operators.config.
    if is_reduced != build_meta.get("reduced_build", False):
        errors.append(
            f"reduced-build flag mismatch: archive name is_reduced={is_reduced} "
            f"but manifest reduced_build={build_meta.get('reduced_build', False)}"
        )
    if is_reduced:
        reduced_cfg = lock.get("reduced_build")
        if not reduced_cfg:
            errors.append("archive is reduced but source lock has no reduced_build section")
        else:
            ops_config_path = ROOT / reduced_cfg["config_path"]
            if not ops_config_path.is_file():
                errors.append(
                    f"required-operators config not found: {ops_config_path}"
                )
            else:
                expected_ops_sha = _sha256_bytes(ops_config_path.read_bytes())
                actual_ops_sha = build_meta.get("ops_config_sha256")
                if actual_ops_sha != expected_ops_sha:
                    errors.append(
                        f"ops_config_sha256 mismatch (manifest={actual_ops_sha} "
                        f"config={expected_ops_sha})"
                    )

    # Supply-chain verification: the archive must contain sbom.spdx.json and
    # provenance.json, and the build manifest must reference them with matching
    # size + SHA-256. The SBOM must be valid SPDX-2.3; the provenance must
    # reference the source lock's upstream commit.
    sc = manifest.get("supply_chain", {})
    for key in ("sbom", "provenance"):
        ref = sc.get(key)
        if not ref:
            errors.append(f"build manifest missing supply_chain.{key} ref")
            continue
        fname = ref.get("path", key.replace("sbom", "sbom.spdx.json").replace("provenance", "provenance.json"))
        if fname not in files:
            errors.append(f"archive missing supply-chain file: {fname}")
            continue
        actual_sha = _sha256_bytes(files[fname])
        if actual_sha != ref["sha256"]:
            errors.append(f"{fname}: sha256 mismatch (manifest={ref['sha256'][:12]}... actual={actual_sha[:12]}...)")
        if len(files[fname]) != ref["size"]:
            errors.append(f"{fname}: size mismatch (manifest={ref['size']} actual={len(files[fname])})")

    if "sbom.spdx.json" in files:
        try:
            sbom = json.loads(files["sbom.spdx.json"])
            if sbom.get("spdxVersion") != "SPDX-2.3":
                errors.append("sbom.spdx.json: not SPDX-2.3")
            if not sbom.get("packages"):
                errors.append("sbom.spdx.json: no packages")
        except json.JSONDecodeError as e:
            errors.append(f"sbom.spdx.json: invalid JSON: {e}")

    if "provenance.json" in files:
        try:
            prov = json.loads(files["provenance.json"])
            if prov.get("schema_version") != "openkara.runtime-provenance/v1":
                errors.append("provenance.json: unexpected schema_version")
            if prov.get("upstream", {}).get("commit_sha") != lock["upstream"]["commit_sha"]:
                errors.append("provenance.json: upstream commit_sha does not match source lock")
            if prov.get("upstream", {}).get("tag") != lock["upstream"]["tag"]:
                errors.append("provenance.json: upstream tag does not match source lock")
        except json.JSONDecodeError as e:
            errors.append(f"provenance.json: invalid JSON: {e}")

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
