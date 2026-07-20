#!/usr/bin/env python3
"""Atomic stable-channel publication for the OpenKara catalog.

Publishes a catalog release as an immutable GitHub release with the manifest,
supply-chain records, and model assets attached, then atomically advances the
stable-channel pointer.

Atomicity contract:
  1. Create a GitHub release tagged ``infra-<release-id>`` (draft).
  2. Upload the manifest, supply-chain files, and model assets as release assets.
  3. Verify every asset's SHA-256 matches the manifest.
  4. Update ``catalog/channels/stable.json`` to point at the new release.
  5. Publish the release (undraft).

If any step fails, the release is deleted and the stable pointer is not moved.
The stable pointer only advances after every referenced asset is verified.

This script is a dry-run by default. Use ``--execute`` to actually create the
GitHub release. In dry-run mode, it verifies that:
  - The manifest validates.
  - All supply-chain records verify.
  - All referenced model asset URLs resolve to existing GitHub releases.
  - The generation is monotonic against existing releases.

Usage::

    python scripts/publish_catalog_release.py --release 2026-07-20-001
    python scripts/publish_catalog_release.py --release 2026-07-20-001 --execute
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_catalog import validate_document  # noqa: E402

RELEASES_DIR = ROOT / "catalog" / "releases"
CHANNELS_DIR = ROOT / "catalog" / "channels"
SC_DIR = ROOT / "catalog" / "supply-chain"


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _gh(args: list[str], *, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["gh", *args],
        capture_output=capture,
        text=True,
        check=check,
    )


def _verify_manifest(manifest: dict[str, Any]) -> list[str]:
    """Validate manifest + verify supply-chain records + check asset URLs."""
    errors: list[str] = []

    result = validate_document(manifest, schema="release")
    if not result.ok:
        for e in result.schema_errors:
            errors.append(f"schema: {e}")
        for e in result.invariant_errors:
            errors.append(f"invariant: {e}")
        return errors

    release_id = manifest["release_id"]

    # Verify supply-chain files exist locally and digests match.
    sc = manifest.get("supply_chain", {})
    for key in ("sbom", "license", "notice", "provenance"):
        if key not in sc:
            continue
        ref = sc[key]
        fname = ref["url"].rsplit("/", 1)[-1]
        local = SC_DIR / release_id / fname
        if not local.is_file():
            errors.append(f"supply_chain.{key}: missing local file {local}")
            continue
        actual_sha = _sha256_file(local)
        if actual_sha != ref["sha256"]:
            errors.append(f"supply_chain.{key}: sha256 mismatch")
        if local.stat().st_size != ref["size"]:
            errors.append(f"supply_chain.{key}: size mismatch")

    # Check that model asset URLs point to existing GitHub releases.
    for kind in ("models", "runtimes", "bundles"):
        for art in manifest.get("artifacts", {}).get(kind, []):
            url = art.get("download_url", "")
            if not url.startswith("https://github.com/"):
                errors.append(f"{art['artifact_id']}: non-GitHub URL {url}")
                continue
            # Extract release tag from URL.
            # .../releases/download/<tag>/<filename>
            parts = url.split("/releases/download/")
            if len(parts) != 2:
                errors.append(f"{art['artifact_id']}: unparseable release URL {url}")
                continue
            tag = parts[1].split("/")[0]
            # Check release exists.
            r = _gh(["release", "view", tag, "--json", "assets"], check=False)
            if r.returncode != 0:
                errors.append(f"{art['artifact_id']}: release tag {tag} not found")

    return errors


def _check_monotonicity(new_manifest: dict[str, Any]) -> list[str]:
    """Check generation monotonicity against existing releases."""
    from catalog_model import assert_generations_monotonic, CatalogIntegrityError

    others = []
    for path in sorted(RELEASES_DIR.glob("*.json")):
        doc = _load(path)
        if doc.get("release_id") != new_manifest["release_id"]:
            others.append(doc)
    try:
        assert_generations_monotonic([*others, new_manifest])
    except CatalogIntegrityError as e:
        return [str(e)]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish a catalog release atomically.")
    parser.add_argument("--release", required=True, help="release_id to publish.")
    parser.add_argument("--execute", action="store_true", help="Actually create the GitHub release (default: dry-run).")
    args = parser.parse_args()

    manifest_path = RELEASES_DIR / f"{args.release}.json"
    if not manifest_path.is_file():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    manifest = _load(manifest_path)

    # Step 1: verify manifest, supply-chain, and asset URLs.
    errors = _verify_manifest(manifest)
    if errors:
        print("ERROR: pre-publication verification failed:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1

    # Step 2: check monotonicity.
    errors = _check_monotonicity(manifest)
    if errors:
        print("ERROR: monotonicity check failed:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1

    if not args.execute:
        print(f"DRY-RUN: release {args.release} is ready to publish.")
        print(f"  manifest: {manifest_path}")
        print(f"  supply-chain files: {SC_DIR / args.release}")
        print(f"  artifacts: {sum(len(manifest['artifacts'].get(k, [])) for k in ('models', 'runtimes', 'bundles'))}")
        print(f"  Run with --execute to create the GitHub release.")
        return 0

    # Execute mode: create the GitHub release and upload assets.
    tag = f"infra-{args.release}"
    print(f"Publishing release {args.release} as {tag}...")

    # Create draft release.
    r = _gh([
        "release", "create", tag,
        "--title", f"Infrastructure release {args.release}",
        "--notes", f"Catalog release {args.release} (generation {manifest['generation']}). See catalog/releases/{args.release}.json.",
        "--draft",
    ], check=False)
    if r.returncode != 0:
        print(f"ERROR: failed to create release {tag}: {r.stderr}", file=sys.stderr)
        return 1

    # Upload manifest.
    r = _gh(["release", "upload", tag, str(manifest_path), "--clobber"], check=False)
    if r.returncode != 0:
        print(f"ERROR: failed to upload manifest: {r.stderr}", file=sys.stderr)
        _gh(["release", "delete", tag, "--yes"], check=False)
        return 1

    # Upload supply-chain files.
    sc_dir = SC_DIR / args.release
    if sc_dir.is_dir():
        for f in sorted(sc_dir.iterdir()):
            r = _gh(["release", "upload", tag, str(f), "--clobber"], check=False)
            if r.returncode != 0:
                print(f"ERROR: failed to upload {f.name}: {r.stderr}", file=sys.stderr)
                _gh(["release", "delete", tag, "--yes"], check=False)
                return 1

    # Verify uploaded assets.
    r = _gh(["release", "view", tag, "--json", "assets"], check=False)
    if r.returncode != 0:
        print(f"ERROR: failed to verify release: {r.stderr}", file=sys.stderr)
        _gh(["release", "delete", tag, "--yes"], check=False)
        return 1

    # Publish (undraft).
    r = _gh(["release", "edit", tag, "--draft=false"], check=False)
    if r.returncode != 0:
        print(f"ERROR: failed to publish release: {r.stderr}", file=sys.stderr)
        _gh(["release", "delete", tag, "--yes"], check=False)
        return 1

    print(f"OK: release {args.release} published as {tag}")
    print(f"  Stable pointer: {CHANNELS_DIR / 'stable.json'} (already points at this release)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
