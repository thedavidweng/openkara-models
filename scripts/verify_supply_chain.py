#!/usr/bin/env python3
"""Verify supply-chain records referenced by a catalog release manifest.

For every supply_chain ref in a release manifest (release-level + per-artifact),
verifies that the referenced file exists in the local ``catalog/supply-chain/<release-id>/``
directory and that its size + SHA-256 match the manifest's declared values.

Usage::

    python scripts/verify_supply_chain.py catalog/releases/<release-id>.json
    python scripts/verify_supply_chain.py --all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SC_DIR = ROOT / "catalog" / "supply-chain"
RELEASES_DIR = ROOT / "catalog" / "releases"


def _sha256_file(path: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return size, h.hexdigest()


def _filename_from_url(url: str) -> str:
    """Extract the filename from a supply-chain ref URL.

    URLs are GitHub release-asset URLs of the form:
    https://github.com/<repo>/releases/download/infra-<id>/<filename>
    """
    return url.rsplit("/", 1)[-1]


def _verify_ref(ref: dict[str, Any], release_id: str, errors: list[str]) -> None:
    fname = _filename_from_url(ref["url"])
    local = SC_DIR / release_id / fname
    if not local.is_file():
        errors.append(f"{release_id}: missing {fname} (url={ref['url']})")
        return
    size, sha = _sha256_file(local)
    if size != ref["size"]:
        errors.append(f"{release_id}: {fname} size mismatch: manifest={ref['size']} actual={size}")
    if sha != ref["sha256"]:
        errors.append(f"{release_id}: {fname} sha256 mismatch: manifest={ref['sha256'][:12]}... actual={sha[:12]}...")


def verify_manifest(path: Path) -> list[str]:
    """Returns a list of error strings (empty = OK)."""
    with path.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    release_id = doc.get("release_id", path.stem)
    errors: list[str] = []

    # Release-level supply_chain.
    sc = doc.get("supply_chain")
    if sc:
        for key in ("sbom", "license", "notice", "provenance"):
            if key in sc:
                _verify_ref(sc[key], release_id, errors)

    # Per-artifact supply_chain.
    for kind in ("models", "runtimes", "bundles"):
        for art in doc.get("artifacts", {}).get(kind, []):
            sc = art.get("supply_chain")
            if sc:
                for key in ("sbom", "license", "notice", "provenance"):
                    if key in sc:
                        _verify_ref(sc[key], release_id, errors)

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify supply-chain records in catalog releases.")
    parser.add_argument("manifest", nargs="?", type=Path, help="Release manifest to verify.")
    parser.add_argument("--all", action="store_true", help="Verify all manifests in catalog/releases/.")
    args = parser.parse_args()

    if not args.all and not args.manifest:
        parser.error("provide a manifest path or --all")

    all_errors: list[tuple[str, list[str]]] = []
    if args.all:
        for path in sorted(RELEASES_DIR.glob("*.json")):
            errs = verify_manifest(path)
            if errs:
                all_errors.append((str(path), errs))
    else:
        errs = verify_manifest(args.manifest)
        if errs:
            all_errors.append((str(args.manifest), errs))

    if all_errors:
        for path, errs in all_errors:
            for e in errs:
                print(f"FAIL: {e}", file=sys.stderr)
        print(f"supply-chain: {sum(len(e) for _, e in all_errors)} error(s)", file=sys.stderr)
        return 1

    count = len(list(RELEASES_DIR.glob("*.json"))) if args.all else 1
    print(f"supply-chain: {count} release(s) verified, 0 errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
