#!/usr/bin/env python3
"""Generate SBOM + provenance records for a built ORT runtime archive.

For each runtime archive in ``ort/packages/``, produces an SPDX-2.3 SBOM and a
provenance JSON document, injects them into the archive (so the archive is
self-describing), and updates the build manifest to reference them. The
archive already contains per-file SHA-256 digests in its build manifest; this
adds the supply-chain layer on top.

The SBOM covers:
  - the ORT upstream package (from the source lock)
  - every dependency archive in the source lock's deps.entries
  - the runtime archive itself (as a package with its SHA-256)

The provenance records:
  - the source lock ref (upstream tag + commit SHA)
  - the build manifest ref (already inside the archive)
  - the archive's own SHA-256 + size
  - the producer commit (from the build manifest, if available)

Usage::

    python scripts/generate_runtime_supply_chain.py --archive ort/packages/<archive>
    python scripts/generate_runtime_supply_chain.py --all

CI freshness guard::

    python scripts/generate_runtime_supply_chain.py --all
    git diff --exit-code -- ort/packages
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "ort" / "source-lock.json"
PACKAGES_DIR = ROOT / "ort" / "packages"

SPDX_VERSION = "SPDX-2.3"
SPDX_SCHEMA = "https://spdx.org/schema/spdx-schema.json"


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return size, h.hexdigest()


def _load_lock() -> dict[str, Any]:
    with LOCK_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


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


def _write_tar(archive: Path, files: dict[str, bytes], fixed_timestamp: int = 315532800) -> None:
    import gzip
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=fixed_timestamp, filename="") as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for name in sorted(files):
                data = files[name]
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                info.mtime = fixed_timestamp
                info.mode = 0o644
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                tar.addfile(info, io.BytesIO(data))
    archive.write_bytes(buf.getvalue())


def _write_zip(archive: Path, files: dict[str, bytes], fixed_timestamp: int = 315532800) -> None:
    fixed_dt = datetime.fromtimestamp(fixed_timestamp, tz=timezone.utc)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(files):
            info = zipfile.ZipInfo(filename=name, date_time=fixed_dt.timetuple()[:6])
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, files[name])


def _build_sbom(
    lock: dict[str, Any], archive: Path, archive_sha: str, archive_size: int,
    target: str, build_manifest: dict[str, Any],
) -> dict[str, Any]:
    """SPDX-2.3 SBOM: ORT upstream + deps + the runtime archive package."""
    upstream = lock["upstream"]
    packages: list[dict[str, Any]] = [
        {
            "name": "onnxruntime",
            "SPDXID": "SPDXRef-Package-ORT-Upstream",
            "downloadLocation": f"git+https://github.com/{upstream['repo']}",
            "filesAnalyzed": False,
            "versionInfo": upstream["tag"],
            "licenseConcluded": "MIT",
            "licenseDeclared": "MIT",
            "copyrightText": "Copyright (c) Microsoft Corporation",
            "supplier": "Organization: Microsoft",
            "primaryPackagePurpose": "SOURCE",
            "externalRefs": [
                {
                    "referenceCategory": "SECURITY",
                    "referenceType": "cpe23Type",
                    "referenceLocator": f"cpe:2.3:a:microsoft:onnxruntime:{upstream['tag'].lstrip('v')}:*:*:*:*:*:*:*",
                },
                {
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceType": "gitoid",
                    "referenceLocator": f"gitoid:blob:sha1:{upstream['commit_sha']}",
                },
            ],
        }
    ]
    # Dependency packages from the source lock.
    for dep_name, dep_info in sorted(lock.get("deps", {}).get("entries", {}).items()):
        packages.append({
            "name": dep_name,
            "SPDXID": f"SPDXRef-Package-Dep-{dep_name.replace('_', '-')}",
            "downloadLocation": dep_info["url"],
            "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": "NOASSERTION",
            "copyrightText": "NOASSERTION",
            "checksums": [{"algorithm": "SHA1", "checksumValue": dep_info["sha1"]}],
            "primaryPackagePurpose": "SOURCE",
        })
    # The runtime archive itself.
    archive_name = archive.name
    packages.append({
        "name": f"onnxruntime-openkara-{target}",
        "SPDXID": "SPDXRef-Package-Runtime-Archive",
        "downloadLocation": "NOASSERTION",
        "filesAnalyzed": False,
        "versionInfo": upstream["tag"],
        "licenseConcluded": "MIT",
        "licenseDeclared": "MIT",
        "copyrightText": "Copyright (c) Microsoft Corporation; OpenKara build",
        "supplier": "Organization: OpenKara",
        "checksums": [{"algorithm": "SHA256", "checksumValue": archive_sha}],
        "primaryPackagePurpose": "BINARY",
        "annotations": [
            {
                "annotationDate": "1980-01-01T00:00:00Z",
                "annotationType": "OTHER",
                "annotator": "Organization: OpenKara",
                "comment": f"archive_size={archive_size} target={target}",
            }
        ],
    })
    return {
        "spdxVersion": SPDX_VERSION,
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"onnxruntime-openkara-{target}",
        "documentNamespace": f"https://openkara.org/spdx/runtime-{target}-{upstream['tag']}",
        "creationInfo": {
            "created": "1980-01-01T00:00:00Z",
            "creators": ["Organization: OpenKara"],
            "licenseListVersion": "3.21",
        },
        "packages": packages,
        "relationships": [
            {
                "spdxElementId": "SPDXRef-DOCUMENT",
                "relationshipType": "DESCRIBES",
                "relatedSpdxElement": "SPDXRef-Package-Runtime-Archive",
            },
            {
                "spdxElementId": "SPDXRef-Package-Runtime-Archive",
                "relationshipType": "GENERATED_FROM",
                "relatedSpdxElement": "SPDXRef-Package-ORT-Upstream",
            },
        ] + [
            {
                "spdxElementId": "SPDXRef-Package-ORT-Upstream",
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": f"SPDXRef-Package-Dep-{name.replace('_', '-')}",
            }
            for name in sorted(lock.get("deps", {}).get("entries", {}))
        ],
    }


def _build_provenance(
    lock: dict[str, Any], archive: Path, archive_sha: str, archive_size: int,
    target: str, build_manifest: dict[str, Any],
) -> dict[str, Any]:
    upstream = lock["upstream"]
    build_meta = build_manifest.get("build", {})
    return {
        "schema_version": "openkara.runtime-provenance/v1",
        "target": target,
        "upstream": {
            "project": upstream["project"],
            "repo": upstream["repo"],
            "tag": upstream["tag"],
            "commit_sha": upstream["commit_sha"],
            "release_url": upstream["release_url"],
        },
        "source_lock": {
            "lock_version": lock["lock_version"],
            "submodules": {
                path: info["expected_sha"]
                for path, info in sorted(lock.get("submodules", {}).items())
            },
        },
        "build": {
            "commit_sha": build_meta.get("commit_sha"),
            "tag": build_meta.get("tag"),
            "cmake_args": build_meta.get("cmake_args"),
            "build_os": build_meta.get("build_os"),
            "build_arch": build_meta.get("build_arch"),
            "reduced_build": build_meta.get("reduced_build", False),
            "ops_config_sha256": build_meta.get("ops_config_sha256"),
        },
        "c_api_level": lock["c_api_level"]["ort_api_version"],
        "subject": {
            "archive_name": archive.name,
            "archive_sha256": archive_sha,
            "archive_size": archive_size,
        },
        "attestation": {
            "type": "source-build-provenance",
            "note": (
                "Built from the pinned ORT source lock by scripts/build_runtime.py. "
                "The build manifest inside the archive records the exact CMake args, "
                "submodule SHAs, and per-file digests. Sigstore/GitHub artifact "
                "attestation is added by the publish workflow when the archive is "
                "uploaded as a release asset (issue #19 PR 5)."
            ),
        },
    }


def generate_for_archive(archive: Path, lock: dict[str, Any]) -> dict[str, Any]:
    """Generate + inject SBOM + provenance into a runtime archive. Returns a summary."""
    files = _read_archive(archive)
    if "build-manifest.json" not in files:
        raise FileNotFoundError(f"{archive.name}: missing build-manifest.json")
    build_manifest = json.loads(files["build-manifest.json"])

    # Parse target from archive name.
    base = archive.name
    if base.endswith(".tar.gz"):
        base = base[:-7]
    elif base.endswith(".zip"):
        base = base[:-4]
    segments = base.split("-openkara-")
    if len(segments) != 2:
        raise ValueError(f"cannot parse target from {archive.name}")
    target = segments[1].removesuffix("-reduced")

    archive_size, archive_sha = _sha256_file(archive)

    sbom = _build_sbom(lock, archive, archive_sha, archive_size, target, build_manifest)
    sbom_bytes = json.dumps(sbom, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
    prov = _build_provenance(lock, archive, archive_sha, archive_size, target, build_manifest)
    prov_bytes = json.dumps(prov, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")

    files["sbom.spdx.json"] = sbom_bytes
    files["provenance.json"] = prov_bytes

    # Update build manifest to reference the supply-chain files.
    manifest_files = build_manifest.setdefault("files", {})
    manifest_files["sbom.spdx.json"] = {
        "size": len(sbom_bytes), "sha256": _sha256_bytes(sbom_bytes),
    }
    manifest_files["provenance.json"] = {
        "size": len(prov_bytes), "sha256": _sha256_bytes(prov_bytes),
    }
    build_manifest["supply_chain"] = {
        "sbom": {
            "path": "sbom.spdx.json",
            "size": len(sbom_bytes),
            "sha256": _sha256_bytes(sbom_bytes),
        },
        "provenance": {
            "path": "provenance.json",
            "size": len(prov_bytes),
            "sha256": _sha256_bytes(prov_bytes),
        },
    }
    files["build-manifest.json"] = (
        json.dumps(build_manifest, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
        + b"\n"
    )

    # Rewrite the archive deterministically.
    if archive.name.endswith(".tar.gz"):
        _write_tar(archive, files)
    elif archive.suffix == ".zip":
        _write_zip(archive, files)
    else:
        raise ValueError(f"unknown archive format: {archive.name}")

    return {
        "archive": archive.name,
        "target": target,
        "archive_sha256": archive_sha,
        "archive_size": archive_size,
        "sbom_size": len(sbom_bytes),
        "provenance_size": len(prov_bytes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate SBOM + provenance for runtime archives.")
    parser.add_argument("--archive", type=Path, default=None, help="Single archive to process.")
    parser.add_argument("--all", action="store_true", help="Process all archives in ort/packages/.")
    args = parser.parse_args()

    if not args.all and not args.archive:
        parser.error("provide --archive or --all")

    lock = _load_lock()
    if args.all:
        archives = sorted(PACKAGES_DIR.glob("*.tar.gz")) + sorted(PACKAGES_DIR.glob("*.zip"))
        if not archives:
            print("No archives found in ort/packages/", file=sys.stderr)
            return 1
    else:
        if not args.archive.is_file():
            print(f"ERROR: archive not found: {args.archive}", file=sys.stderr)
            return 1
        archives = [args.archive]

    for archive in archives:
        print(f"Generating supply-chain for {archive.name}...")
        summary = generate_for_archive(archive, lock)
        print(f"  target:     {summary['target']}")
        print(f"  archive:    {summary['archive_size']} bytes  sha256 {summary['archive_sha256'][:12]}...")
        print(f"  sbom:       {summary['sbom_size']} bytes")
        print(f"  provenance: {summary['provenance_size']} bytes")

    print(f"OK: {len(archives)} archive(s) processed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
