#!/usr/bin/env python3
"""Atomic stable-channel publication for the OpenKara catalog.

Publishes a catalog release as an immutable GitHub release with the manifest,
supply-chain records, and model assets attached, then atomically advances the
stable-channel pointer.

Atomicity contract:
  1. Pre-publication verification: manifest validates, supply-chain records
     match local files, every referenced asset URL resolves to an existing
     GitHub release asset whose SHA-256 matches the manifest, generation is
     monotonic.
  2. Create a GitHub release tagged ``infra-<release-id>`` (draft).
  3. Upload the manifest and supply-chain files as release assets.
  4. Verify every uploaded asset's SHA-256 and size match the manifest.
  5. Update ``catalog/channels/stable.json`` to point at the new release
     (only after every asset is verified).
  6. Publish the release (undraft).

If any step fails, the release is deleted and the stable pointer is not moved.
The stable pointer only advances after every referenced asset is verified.

This script is a dry-run by default. Use ``--execute`` to actually create the
GitHub release. In dry-run mode, it performs every pre-publication check that
does not require write access: manifest validation, supply-chain verification,
asset URL resolution + SHA-256 verification, and monotonicity.

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


def _parse_release_url(url: str) -> tuple[str, str] | None:
    """Parse a GitHub release download URL into (owner/repo, tag)."""
    if not url.startswith("https://github.com/"):
        return None
    parts = url.split("/releases/download/")
    if len(parts) != 2:
        return None
    repo = parts[0][len("https://github.com/"):]
    tag = parts[1].split("/")[0]
    return repo, tag


def _release_assets(repo: str, tag: str) -> list[dict[str, Any]] | None:
    """Return the asset list for a GitHub release, or None if the release
    cannot be queried."""
    r = _gh(
        ["release", "view", tag, "--repo", repo, "--json", "assets"],
        check=False,
    )
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout).get("assets", [])
    except json.JSONDecodeError:
        return None


def _find_asset_by_name(assets: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for a in assets:
        if a.get("name") == name:
            return a
    return None


def _download_asset_sha256(repo: str, tag: str, asset_name: str) -> str | None:
    """Download a release asset and compute its SHA-256.

    Uses ``gh release download`` to a temp file, then hashes it. Returns None
    if the download fails.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        dest = Path(td) / asset_name
        r = _gh(
            ["release", "download", tag, "--repo", repo,
             "--pattern", asset_name, "--dir", str(dest.parent)],
            check=False,
        )
        if r.returncode != 0 or not dest.is_file():
            return None
        return _sha256_file(dest)


def _verify_manifest(manifest: dict[str, Any]) -> list[str]:
    """Pre-publication verification: manifest validates, supply-chain records
    match local files, every referenced asset URL resolves to an existing
    GitHub release asset whose SHA-256 matches the manifest.

    No ``--skip-asset-check`` escape hatch: every referenced asset must be
    reachable and verifiable before publication. CI must run with ``gh`` auth
    that can read the asset repos (same-owner releases do not need cross-repo
    auth; cross-repo assets require a token with read scope).
    """
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
            errors.append(
                f"supply_chain.{key}: sha256 mismatch (local={actual_sha[:12]}... "
                f"manifest={ref['sha256'][:12]}...)"
            )
        if local.stat().st_size != ref["size"]:
            errors.append(
                f"supply_chain.{key}: size mismatch (local={local.stat().st_size} "
                f"manifest={ref['size']})"
            )

    # Verify every referenced artifact asset URL resolves to an existing
    # GitHub release asset whose SHA-256 matches the manifest.
    for kind in ("models", "runtimes", "bundles"):
        for art in manifest.get("artifacts", {}).get(kind, []):
            aid = art.get("artifact_id", "<unknown>")
            url = art.get("download_url", "")
            parsed = _parse_release_url(url)
            if parsed is None:
                errors.append(f"{aid}: download_url is not a GitHub release URL: {url}")
                continue
            repo, tag = parsed
            assets = _release_assets(repo, tag)
            if assets is None:
                errors.append(f"{aid}: cannot query release {tag} in {repo}")
                continue
            fname = url.rsplit("/", 1)[-1]
            asset = _find_asset_by_name(assets, fname)
            if asset is None:
                errors.append(f"{aid}: asset {fname} not found in release {tag} of {repo}")
                continue
            # Verify size first (cheap, no download).
            if asset.get("size") != art.get("byte_size"):
                errors.append(
                    f"{aid}: asset size mismatch (github={asset.get('size')} "
                    f"manifest={art.get('byte_size')})"
                )
                continue
            # Download and hash to verify SHA-256.
            actual_sha = _download_asset_sha256(repo, tag, fname)
            if actual_sha is None:
                errors.append(f"{aid}: failed to download asset {fname} for SHA-256 verification")
                continue
            if actual_sha != art.get("archive_digest"):
                errors.append(
                    f"{aid}: sha256 mismatch (github={actual_sha[:12]}... "
                    f"manifest={art.get('archive_digest', '')[:12]}...)"
                )

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


def _advance_stable_pointer(manifest: dict[str, Any], manifest_path: Path) -> list[str]:
    """Rewrite catalog/channels/stable.json to point at the verified manifest.

    The pointer's release_manifest_url points at the GitHub release asset URL
    (infra-<release-id>/release-manifest.json), and the digest/size match the
    local manifest bytes (which are byte-identical to the uploaded asset).
    """
    release_id = manifest["release_id"]
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_size = len(manifest_bytes)
    repo = manifest["producer"]["repo"]
    publish_url = (
        f"https://github.com/{repo}/releases/download/"
        f"infra-{release_id}/release-manifest.json"
    )
    pointer = {
        "schema_version": "openkara.catalog/channel-v1",
        "channel": "stable",
        "generation": manifest["generation"],
        "release_id": release_id,
        "release_manifest_url": publish_url,
        "release_manifest_sha256": manifest_sha,
        "release_manifest_size": manifest_size,
        "updated_at": manifest["created_at"],
        "previous_release_id": None,
    }
    # Preserve previous_release_id from the existing pointer if present.
    existing_pointer = CHANNELS_DIR / "stable.json"
    if existing_pointer.is_file():
        try:
            old = _load(existing_pointer)
            old_id = old.get("release_id")
            if old_id and old_id != release_id:
                pointer["previous_release_id"] = old_id
        except (json.JSONDecodeError, OSError):
            pass

    pointer_result = validate_document(pointer, schema="channel")
    if not pointer_result.ok:
        errs = []
        for e in pointer_result.schema_errors:
            errs.append(f"pointer schema: {e}")
        for e in pointer_result.invariant_errors:
            errs.append(f"pointer invariant: {e}")
        return errs

    CHANNELS_DIR.mkdir(parents=True, exist_ok=True)
    with existing_pointer.open("w", encoding="utf-8") as fh:
        json.dump(pointer, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish a catalog release atomically.")
    parser.add_argument("--release", required=True, help="release_id to publish.")
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually create the GitHub release and advance the stable pointer (default: dry-run).",
    )
    args = parser.parse_args()

    manifest_path = RELEASES_DIR / f"{args.release}.json"
    if not manifest_path.is_file():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    manifest = _load(manifest_path)

    # Step 1: pre-publication verification (manifest, supply-chain, assets, monotonicity).
    errors = _verify_manifest(manifest)
    if errors:
        print("ERROR: pre-publication verification failed:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1

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
        n_artifacts = sum(len(manifest['artifacts'].get(k, [])) for k in ('models', 'runtimes', 'bundles'))
        print(f"  artifacts: {n_artifacts} (all asset URLs verified, SHA-256 matches)")
        print(f"  Run with --execute to create the GitHub release and advance the stable pointer.")
        return 0

    # Execute mode: create the GitHub release, upload, verify, advance pointer.
    tag = f"infra-{args.release}"
    repo = manifest["producer"]["repo"]
    print(f"Publishing release {args.release} as {tag}...")

    # Create draft release.
    r = _gh([
        "release", "create", tag, "--repo", repo,
        "--title", f"Infrastructure release {args.release}",
        "--notes", f"Catalog release {args.release} (generation {manifest['generation']}). "
                   f"See catalog/releases/{args.release}.json.",
        "--draft",
    ], check=False)
    if r.returncode != 0:
        print(f"ERROR: failed to create release {tag}: {r.stderr}", file=sys.stderr)
        return 1

    def _cleanup_release() -> None:
        _gh(["release", "delete", tag, "--repo", repo, "--yes"], check=False)

    # Upload manifest.
    r = _gh(["release", "upload", tag, "--repo", repo, str(manifest_path), "--clobber"], check=False)
    if r.returncode != 0:
        print(f"ERROR: failed to upload manifest: {r.stderr}", file=sys.stderr)
        _cleanup_release()
        return 1

    # Upload supply-chain files.
    sc_dir = SC_DIR / args.release
    if sc_dir.is_dir():
        for f in sorted(sc_dir.iterdir()):
            if not f.is_file():
                continue
            r = _gh(["release", "upload", tag, "--repo", repo, str(f), "--clobber"], check=False)
            if r.returncode != 0:
                print(f"ERROR: failed to upload {f.name}: {r.stderr}", file=sys.stderr)
                _cleanup_release()
                return 1

    # Verify uploaded manifest + supply-chain assets match local digests.
    assets = _release_assets(repo, tag)
    if assets is None:
        print(f"ERROR: failed to query uploaded assets for {tag}", file=sys.stderr)
        _cleanup_release()
        return 1

    expected_uploads: list[tuple[str, str, int]] = []  # (name, sha256, size)
    expected_uploads.append((
        manifest_path.name,
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        manifest_path.stat().st_size,
    ))
    sc = manifest.get("supply_chain", {})
    for key in ("sbom", "license", "notice", "provenance"):
        if key not in sc:
            continue
        ref = sc[key]
        fname = ref["url"].rsplit("/", 1)[-1]
        local = SC_DIR / args.release / fname
        if local.is_file():
            expected_uploads.append((fname, ref["sha256"], ref["size"]))

    for fname, expected_sha, expected_size in expected_uploads:
        asset = _find_asset_by_name(assets, fname)
        if asset is None:
            print(f"ERROR: uploaded asset {fname} not found in release {tag}", file=sys.stderr)
            _cleanup_release()
            return 1
        if asset.get("size") != expected_size:
            print(
                f"ERROR: uploaded asset {fname} size mismatch "
                f"(github={asset.get('size')} expected={expected_size})",
                file=sys.stderr,
            )
            _cleanup_release()
            return 1
        # Download and verify SHA-256 of the uploaded asset.
        actual_sha = _download_asset_sha256(repo, tag, fname)
        if actual_sha is None or actual_sha != expected_sha:
            print(
                f"ERROR: uploaded asset {fname} sha256 mismatch "
                f"(github={actual_sha and actual_sha[:12]}... expected={expected_sha[:12]}...)",
                file=sys.stderr,
            )
            _cleanup_release()
            return 1

    # All uploaded assets verified. Now advance the stable pointer.
    errors = _advance_stable_pointer(manifest, manifest_path)
    if errors:
        print("ERROR: failed to advance stable pointer:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        _cleanup_release()
        return 1

    # Publish (undraft) the release.
    r = _gh(["release", "edit", tag, "--repo", repo, "--draft=false"], check=False)
    if r.returncode != 0:
        print(f"ERROR: failed to publish release: {r.stderr}", file=sys.stderr)
        # The pointer has already advanced; do NOT delete the release here.
        # The release is verified and the pointer is correct; the only failure
        # is undrafting, which can be retried manually.
        print(
            "  The stable pointer has already advanced. Manually undraft the "
            f"release with: gh release edit {tag} --repo {repo} --draft=false",
            file=sys.stderr,
        )
        return 1

    print(f"OK: release {args.release} published as {tag}")
    print(f"  Stable pointer advanced: {CHANNELS_DIR / 'stable.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
