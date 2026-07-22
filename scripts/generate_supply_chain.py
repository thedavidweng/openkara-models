#!/usr/bin/env python3
"""Generate supply-chain records for a catalog release and inject them.

Produces four artifacts per release, computes their size + SHA-256, and writes
them under ``catalog/supply-chain/<release-id>/``:

1. ``sbom.spdx.json``  — SPDX-2.3 JSON SBOM covering the repo + release assets.
2. ``LICENSE``         — verbatim copy of the repo LICENSE (MIT).
3. ``NOTICE``          — attribution notice for the release.
4. ``provenance.json`` — provenance ref (commit SHA + workflow + asset digests).

Then injects ``supply_chain`` refs into the release spec at
``catalog/specs/<release-id>.spec.json`` so the next
``generate_catalog_release.py`` run embeds them in the manifest. The generator
is deterministic: sorted keys, no wall-clock timestamps (uses spec.created_at).

Usage::

    python scripts/generate_supply_chain.py --release 2026-07-20-001

CI freshness guard::

    python scripts/generate_supply_chain.py --release 2026-07-20-001
    git diff --exit-code -- catalog/supply-chain catalog/specs
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SC_DIR = ROOT / "catalog" / "supply-chain"
SPECS_DIR = ROOT / "catalog" / "specs"
LICENSE_PATH = ROOT / "LICENSE"

SPDX_SCHEMA = "https://spdx.org/schema/spdx-schema.json"
SPDX_VERSION = "SPDX-2.3"


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _dump_json(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")


def _dump_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(text)


def _load_spec(release_id: str) -> tuple[Path, dict[str, Any]]:
    path = SPECS_DIR / f"{release_id}.spec.json"
    if not path.is_file():
        raise FileNotFoundError(f"spec not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return path, json.load(fh)


def _build_sbom(spec: dict[str, Any], release_id: str) -> dict[str, Any]:
    """Minimal SPDX-2.3 JSON SBOM: repo package + one package per artifact."""
    packages: list[dict[str, Any]] = [
        {
            "name": "openkara-models",
            "SPDXID": "SPDXRef-Package-Repo",
            "downloadLocation": f"git+https://github.com/{spec['producer']['repo']}",
            "filesAnalyzed": False,
            "versionInfo": spec["producer"]["commit_sha"],
            "licenseConcluded": "MIT",
            "licenseDeclared": "MIT",
            "copyrightText": "Copyright (c) 2026 Davy",
            "supplier": "Organization: OpenKara",
            "primaryPackagePurpose": "APPLICATION",
        }
    ]
    for kind in ("models", "runtimes", "bundles"):
        for art in spec.get("artifacts", {}).get(kind, []):
            packages.append(
                {
                    "name": art["artifact_id"],
                    "SPDXID": f"SPDXRef-Package-{art['artifact_id'].replace('.', '-')}",
                    "downloadLocation": art.get("download_url", "NOASSERTION"),
                    "filesAnalyzed": False,
                    "licenseConcluded": "MIT",
                    "licenseDeclared": "MIT",
                    "copyrightText": "NOASSERTION",
                    "checksums": [
                        {
                            "algorithm": "SHA256",
                            "checksumValue": art.get("_digest", {}).get(
                                "archive_digest", "NOASSERTION"
                            ),
                        }
                    ],
                    "primaryPackagePurpose": (
                        "BINARY" if kind == "runtimes" else "FILE"
                    ),
                }
            )
    return {
        "spdxVersion": SPDX_VERSION,
        "dataLicense": "CC0-1.0",
        "SPDXID": f"SPDXRef-DOCUMENT-{release_id.replace('-', '')}",
        "name": f"openkara-models-{release_id}",
        "documentNamespace": f"https://openkara.org/spdx/{release_id}",
        "creationInfo": {
            "created": spec["created_at"],
            "creators": ["Organization: OpenKara"],
            "licenseListVersion": "3.21",
        },
        "packages": packages,
        "relationships": [
            {
                "spdxElementId": "SPDXRef-DOCUMENT",
                "relationshipType": "DESCRIBES",
                "relatedSpdxElement": "SPDXRef-Package-Repo",
            }
        ],
    }


def _build_notice(spec: dict[str, Any], release_id: str) -> str:
    lines = [
        f"OpenKara Models — release {release_id}",
                "Copyright (c) 2026 Davy",
        "",
        "This release is distributed under the MIT License (see LICENSE).",
        "",
        "Third-party model weights are derived from Demucs by Alexandre Défossez",
        "(https://github.com/adefossez/demucs), released under the MIT License.",
        "",
        "Producer:",
        f"  repo:      {spec['producer']['repo']}",
        f"  commit:    {spec['producer']['commit_sha']}",
        f"  workflow:  {spec['producer']['workflow']}",
        f"  run_id:    {spec['producer']['run_id']}",
        "",
    ]
    return "\n".join(lines)


def _build_provenance(spec: dict[str, Any], release_id: str) -> dict[str, Any]:
    return {
        "release_id": release_id,
        "created_at": spec["created_at"],
        "producer": spec["producer"],
        "subject": [
            {
                "artifact_id": art["artifact_id"],
                "download_url": art.get("download_url"),
                "sha256": art.get("_digest", {}).get("archive_digest"),
                "size": art.get("_digest", {}).get("byte_size"),
            }
            for kind in ("models", "runtimes", "bundles")
            for art in spec.get("artifacts", {}).get(kind, [])
        ],
        "attestation": {
            "type": "commit-provenance",
            "url": (
                f"https://github.com/{spec['producer']['repo']}/commit/"
                f"{spec['producer']['commit_sha']}"
            ),
            "note": (
                "Commit-based provenance for the legacy migration release. New "
                "releases (#19/#22) add sigstore/github-attestation refs."
            ),
        },
    }


def _ref(url: str, size: int, sha: str, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "url": url,
        "size": size,
        "sha256": sha,
        "digest_algorithm": "sha256",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate supply-chain records for a release.")
    parser.add_argument("--release", required=True, help="release_id to back-fill.")
    args = parser.parse_args()

    try:
        spec_path, spec = _load_spec(args.release)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    out_dir = SC_DIR / args.release
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. SBOM
    sbom = _build_sbom(spec, args.release)
    sbom_bytes = json.dumps(sbom, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
    (out_dir / "sbom.spdx.json").write_bytes(sbom_bytes)

    # 2. LICENSE copy
    license_bytes = LICENSE_PATH.read_bytes()
    (out_dir / "LICENSE").write_bytes(license_bytes)

    # 3. NOTICE
    notice_text = _build_notice(spec, args.release)
    notice_bytes = notice_text.encode("utf-8")
    (out_dir / "NOTICE").write_bytes(notice_bytes)

    # 4. Provenance
    prov = _build_provenance(spec, args.release)
    prov_bytes = json.dumps(prov, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
    (out_dir / "provenance.json").write_bytes(prov_bytes)

    # Build supply_chain refs. URLs are immutable GitHub release-asset URLs
    # (published as release assets in issue #18 PR 4). The in-repo copies under
    # catalog/supply-chain/<release-id>/ are the source of truth; the release
    # assets are the immutable distribution copy.
    repo = spec["producer"]["repo"]
    base = f"https://github.com/{repo}/releases/download/infra-{args.release}"
    supply_chain = {
        "sbom": _ref(
            f"{base}/sbom.spdx.json", len(sbom_bytes), _sha256_bytes(sbom_bytes), "spdx-json"
        ),
        "license": _ref(
            f"{base}/LICENSE", len(license_bytes), _sha256_bytes(license_bytes), "license-file"
        ),
        "notice": _ref(
            f"{base}/NOTICE", len(notice_bytes), _sha256_bytes(notice_bytes), "notice-file"
        ),
        "provenance": _ref(
            f"{base}/provenance.json", len(prov_bytes), _sha256_bytes(prov_bytes), "github-attestation"
        ),
    }

    # Inject into spec (release-level + per-artifact).
    new_spec = copy.deepcopy(spec)
    new_spec["supply_chain"] = supply_chain
    for kind in ("models", "runtimes", "bundles"):
        for art in new_spec.get("artifacts", {}).get(kind, []):
            art["supply_chain"] = supply_chain
    _dump_json(spec_path, new_spec)

    print(f"OK: supply-chain records for {args.release} in {out_dir}")
    print(f"  sbom:        {len(sbom_bytes)} bytes  sha256 {_sha256_bytes(sbom_bytes)[:12]}...")
    print(f"  license:     {len(license_bytes)} bytes  sha256 {_sha256_bytes(license_bytes)[:12]}...")
    print(f"  notice:      {len(notice_bytes)} bytes  sha256 {_sha256_bytes(notice_bytes)[:12]}...")
    print(f"  provenance:  {len(prov_bytes)} bytes  sha256 {_sha256_bytes(prov_bytes)[:12]}...")
    print(f"  spec updated: {spec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
