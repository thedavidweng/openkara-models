#!/usr/bin/env python3
"""Deterministic immutable release-manifest generator for the OpenKara catalog.

Reads a release spec (``catalog/specs/<release-id>.spec.json``), assembles an
immutable release manifest, validates it against the v1 schema + invariants,
and writes three artifacts:

1. ``catalog/releases/<release-id>.json`` — the immutable manifest.
2. ``catalog/channels/stable.json`` — the stable-channel pointer, advanced only
   after the manifest validates.
3. ``latest.json`` — a temporary migration adapter (issue #18 PR 4 deletes it)
   carrying ``{ "<variant>": {tag, url, sha256, size} }`` for OpenKara PR #165.

Generation is deterministic: ``created_at`` comes from the spec (never wall
clock), and JSON is serialized with sorted keys, 2-space indent, UTF-8, and a
trailing newline. Running twice on the same spec + commit produces byte-identical
output.

The generator fails (non-zero exit, no stable pointer written) when:

- the manifest or pointer fails validation;
- ``release_id`` already exists in ``catalog/releases/`` with different content;
- ``generation`` is non-monotonic against existing releases.

Usage::

    python scripts/generate_catalog_release.py --spec catalog/specs/<id>.spec.json
    python scripts/generate_catalog_release.py --spec ... --force

CI freshness guard::

    python scripts/generate_catalog_release.py --spec catalog/specs/<id>.spec.json
    git diff --exit-code -- catalog/releases catalog/channels latest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from catalog_model import (  # noqa: E402
    CatalogIntegrityError,
    assert_generations_monotonic,
)
from validate_catalog import validate_document  # noqa: E402

CATALOG_DIR = ROOT / "catalog"
RELEASES_DIR = CATALOG_DIR / "releases"
CHANNELS_DIR = CATALOG_DIR / "channels"
LATEST_JSON_PATH = ROOT / "latest.json"

_GH_RELEASE_TAG_RE = re.compile(r"/releases/download/([A-Za-z0-9_.-]+)/")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _sha256_file(path: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return size, h.hexdigest()


def _dump_json(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _release_tag_from_url(url: str) -> str:
    m = _GH_RELEASE_TAG_RE.search(url)
    if m:
        return m.group(1)
    return ""


# --------------------------------------------------------------------------- #
# Digest resolution
# --------------------------------------------------------------------------- #


def _resolve_digests(art: dict[str, Any], spec_dir: Path) -> dict[str, Any]:
    """Fill byte_size, archive_digest, extracted_file_digests from ``_digest``.

    ``_digest`` is stripped from the output. Two modes:

    - ``{"mode": "asset", "path": "<local-file>"}`` — compute size + sha256 of
      the file and use it as the single extracted file (keyed by basename).
    - ``{"mode": "predeclared", "byte_size", "archive_digest",
      "extracted_file_digests"}`` — use the supplied values verbatim (for
      already-published assets not available locally).
    """
    digest_spec = art.pop("_digest", None)
    if digest_spec is None:
        raise ValueError(
            f"artifact {art.get('artifact_id')!r} missing '_digest' "
            f"(use asset or predeclared)"
        )
    mode = digest_spec.get("mode")
    if mode == "asset":
        asset_path = (spec_dir / digest_spec["path"]).resolve()
        if not asset_path.is_file():
            raise FileNotFoundError(f"asset not found for {art.get('artifact_id')!r}: {asset_path}")
        size, sha = _sha256_file(asset_path)
        art["byte_size"] = size
        art["archive_digest"] = sha
        art["extracted_file_digests"] = {
            asset_path.name: {"size": size, "sha256": sha}
        }
    elif mode == "predeclared":
        art["byte_size"] = digest_spec["byte_size"]
        art["archive_digest"] = digest_spec["archive_digest"]
        art["extracted_file_digests"] = digest_spec["extracted_file_digests"]
    else:
        raise ValueError(f"artifact {art.get('artifact_id')!r} has unknown _digest mode {mode!r}")
    return art


def _build_manifest(spec: dict[str, Any], spec_dir: Path) -> dict[str, Any]:
    artifacts_in = spec.get("artifacts", {})
    artifacts_out: dict[str, list[dict[str, Any]]] = {}
    for kind in ("models", "runtimes", "bundles"):
        out_list: list[dict[str, Any]] = []
        for art in artifacts_in.get(kind, []):
            out_list.append(_resolve_digests(dict(art), spec_dir))
        artifacts_out[kind] = out_list

    manifest: dict[str, Any] = {
        "schema_version": "openkara.catalog/release-v1",
        "release_id": spec["release_id"],
        "generation": spec["generation"],
        "created_at": spec["created_at"],
        "producer": spec["producer"],
        "artifacts": artifacts_out,
        "compatibility": spec.get("compatibility", []),
    }
    for opt in (
        "supply_chain",
        "gates",
        "minimum_consumer_schema",
        "maximum_consumer_schema",
        "notes",
    ):
        if opt in spec:
            manifest[opt] = spec[opt]
    return manifest


def _build_pointer(
    spec: dict[str, Any],
    manifest_publish_url: str,
    manifest_sha: str,
    manifest_size: int,
) -> dict[str, Any]:
    return {
        "schema_version": "openkara.catalog/channel-v1",
        "channel": "stable",
        "generation": spec["generation"],
        "release_id": spec["release_id"],
        "release_manifest_url": manifest_publish_url,
        "release_manifest_sha256": manifest_sha,
        "release_manifest_size": manifest_size,
        "updated_at": spec["created_at"],
        "previous_release_id": spec.get("previous_release_id"),
    }


def _build_latest_adapter(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """{ "<variant>": {tag, url, sha256, size} } for every model artifact.

    Matches the shape OpenKara PR #165 fetches from ``latest.json``. Deleted in
    issue #18 PR 4 after OpenKara #167 switches to the versioned schema.
    """
    adapter: dict[str, dict[str, Any]] = {}
    for model in manifest.get("artifacts", {}).get("models", []):
        variant = model.get("variant")
        if not variant:
            continue
        url = model.get("download_url", "")
        adapter[variant] = {
            "tag": _release_tag_from_url(url),
            "url": url,
            "sha256": model.get("archive_digest"),
            "size": model.get("byte_size"),
        }
    return adapter


# --------------------------------------------------------------------------- #
# Existing-release guards
# --------------------------------------------------------------------------- #


def _load_existing_release(release_id: str, releases_dir: Path) -> Path | None:
    """Return the path of an already-committed manifest with this release_id, if any."""
    if not releases_dir.is_dir():
        return None
    candidate = releases_dir / f"{release_id}.json"
    return candidate if candidate.is_file() else None


def _other_releases(release_id: str, releases_dir: Path) -> list[dict[str, Any]]:
    """All committed manifests except the one matching ``release_id``."""
    if not releases_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(releases_dir.glob("*.json")):
        try:
            doc = _load_json(path)
        except (json.JSONDecodeError, OSError):
            continue
        if doc.get("release_id") == release_id:
            continue
        out.append(doc)
    return out


class GuardError(Exception):
    """Raised when a release cannot be (re)generated without violating invariants."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


def _guard_against_contradictory_releases(
    new_id: str,
    new_gen: int,
    new_bytes: bytes,
    releases_dir: Path,
    *,
    force: bool = False,
) -> None:
    """Reject regeneration that would change bytes for an existing release_id,
    regress generation, or break monotonicity.

    ``force`` allows rewriting an existing release_id only when the new bytes
    are byte-identical to the committed manifest (idempotent regeneration).
    Different bytes for an existing release_id are always rejected — release
    manifests are immutable; bump ``generation`` and use a new ``release_id``.
    """
    existing_path = _load_existing_release(new_id, releases_dir)
    if existing_path is not None:
        existing_bytes = existing_path.read_bytes()
        if existing_bytes != new_bytes:
            raise GuardError(
                "immutable_release_id",
                f"release_id {new_id!r} already exists at {existing_path} with different "
                "content. Release manifests are immutable; do not regenerate a published "
                "release with changed bytes. Bump generation and use a new release_id.",
            )
        # Identical bytes: idempotent regeneration is allowed (with or without
        # --force). --force is accepted for backward compatibility but is not
        # required for identical rewrites. Skip the monotonicity check since
        # we are not creating a new release — just verifying immutability.
        return

    others = _other_releases(new_id, releases_dir)
    if others and max(d.get("generation", 0) for d in others) > new_gen:
        raise GuardError(
            "non_monotonic_generation",
            f"generation {new_gen} < existing max "
            f"{max(d.get('generation', 0) for d in others)}",
        )
    # Monotonicity across the full set including the new release.
    new_doc = json.loads(new_bytes.decode("utf-8"))
    assert_generations_monotonic([*others, new_doc])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate an immutable catalog release manifest + stable pointer + latest.json."
    )
    parser.add_argument("--spec", type=Path, required=True, help="Release spec JSON path.")
    parser.add_argument(
        "--manifest-publish-url",
        type=str,
        default=None,
        help="Immutable URL the published manifest will live at (GitHub release asset). "
        "Defaults to the in-repo raw URL placeholder.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Accepted for backward compatibility. Idempotent regeneration (byte-identical "
        "output for an existing release_id) is always allowed; mutating a published "
        "release is always rejected — bump generation and use a new release_id instead.",
    )
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        default=CATALOG_DIR,
        help="Catalog root (default: repo catalog/). Tests point this at a temp dir.",
    )
    parser.add_argument(
        "--latest-json-path",
        type=Path,
        default=LATEST_JSON_PATH,
        help="latest.json adapter path (default: repo root latest.json).",
    )
    args = parser.parse_args()

    spec_path: Path = args.spec.resolve()
    if not spec_path.is_file():
        print(f"ERROR: spec not found: {spec_path}", file=sys.stderr)
        return 1
    spec = _load_json(spec_path)
    spec_dir = spec_path.parent

    releases_dir = args.catalog_dir / "releases"
    channels_dir = args.catalog_dir / "channels"

    try:
        manifest = _build_manifest(spec, spec_dir)
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Validate the manifest before writing anything.
    result = validate_document(manifest, schema="release")
    if not result.ok:
        print("ERROR: manifest failed validation:", file=sys.stderr)
        for e in result.schema_errors:
            print(f"  SCHEMA: {e}", file=sys.stderr)
        for e in result.invariant_errors:
            print(f"  INVARIANT: {e}", file=sys.stderr)
        return 1

    release_id = manifest["release_id"]
    release_path = releases_dir / f"{release_id}.json"

    # Serialize once to get canonical bytes, then guard against contradictions.
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")

    try:
        _guard_against_contradictory_releases(
            release_id,
            manifest["generation"],
            manifest_bytes,
            releases_dir,
            force=args.force,
        )
    except (CatalogIntegrityError, GuardError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Write the immutable manifest.
    _dump_json(release_path, manifest)
    manifest_size = release_path.stat().st_size
    _, manifest_sha = _sha256_file(release_path)

    # Validate the candidate pointer shape but DO NOT write the stable pointer.
    # The stable pointer is advanced only by publish_catalog_release.py after
    # every referenced asset is uploaded and SHA-256-verified. Writing it here
    # would advance the pointer before the assets exist, breaking the
    # atomicity contract (issue #18 PR 4).
    publish_url = args.manifest_publish_url or (
        f"https://github.com/thedavidweng/openkara-models/releases/download/"
        f"infra-{release_id}/release-manifest.json"
    )
    pointer = _build_pointer(spec, publish_url, manifest_sha, manifest_size)
    pointer_result = validate_document(pointer, schema="channel")
    if not pointer_result.ok:
        print("ERROR: candidate stable pointer failed validation:", file=sys.stderr)
        for e in pointer_result.schema_errors:
            print(f"  SCHEMA: {e}", file=sys.stderr)
        for e in pointer_result.invariant_errors:
            print(f"  INVARIANT: {e}", file=sys.stderr)
        # Roll back the manifest write so a bad pointer never leaves a half state.
        release_path.unlink(missing_ok=True)
        return 1

    # latest.json migration adapter.
    adapter = _build_latest_adapter(manifest)
    _dump_json(args.latest_json_path, adapter)

    print(f"OK: release {release_id} (generation {manifest['generation']})")
    print(f"  manifest:  {release_path} ({manifest_size} bytes, sha256 {manifest_sha[:12]}...)")
    print(f"  pointer:   NOT advanced (use scripts/publish_catalog_release.py --execute)")
    print(f"  adapter:   {args.latest_json_path} ({len(adapter)} variants)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
