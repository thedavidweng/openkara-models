#!/usr/bin/env python3
"""Guard direct stable-pointer changes in pull requests.

The catalog validation workflow runs the publisher only as a dry run, so a
hand-edited ``catalog/channels/stable.json`` could otherwise advance the stable
pointer without either release gate. This guard inspects the pointer in the
working tree, compares it with the pull request's base commit, and requires the
transition to be authorised by:

- a committed runtime-only qualification record for the new stable release
  (validated against the promoted-from pointer), or
- strict release evidence (all three immutable report envelopes) when no
  runtime-only record exists.

It establishes schema/byte integrity and the pointer transition; it does not
prove that a producer executed measurements.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_catalog import validate_document  # noqa: E402
from validate_release_evidence import qualify_release  # noqa: E402
from validate_runtime_only import (  # noqa: E402
    load_record,
    qualify_runtime_only,
)

STABLE_PATH = "catalog/channels/stable.json"
CANDIDATE_PATH = "catalog/channels/candidate.json"


def _pointer(raw: bytes) -> dict[str, Any]:
    doc = json.loads(raw)
    result = validate_document(doc, schema="channel")
    if not result.ok:
        raise ValueError(f"invalid channel pointer: {result.all_codes()}")
    return doc


def _git_show(ref: str, path: str, *, cwd: Path) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"], cwd=str(cwd), capture_output=True,
    )
    if result.returncode != 0:
        raise ValueError(f"cannot read {path} at {ref}: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def verify_pointer_binding(pointer: dict[str, Any], manifest: dict[str, Any],
                           manifest_bytes: bytes) -> list[str]:
    """The head pointer must name the manifest it claims, byte-exactly."""
    import hashlib

    errors: list[str] = []
    release_id = manifest.get("release_id")
    if pointer.get("schema_version") != "openkara.catalog/channel-v1":
        errors.append("stable pointer schema_version is not openkara.catalog/channel-v1")
    if pointer.get("channel") != "stable":
        errors.append("stable pointer channel is not 'stable'")
    if pointer.get("release_id") != release_id:
        errors.append("stable pointer release_id does not match the manifest")
    if pointer.get("generation") != manifest.get("generation"):
        errors.append("stable pointer generation does not match the manifest")
    expected_url = (
        f"https://github.com/{manifest.get('producer', {}).get('repo')}/releases/download/"
        f"infra-{release_id}/release-manifest.json"
    )
    if pointer.get("release_manifest_url") != expected_url:
        errors.append("stable pointer release_manifest_url is not the publisher's manifest asset")
    if pointer.get("release_manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest():
        errors.append("stable pointer manifest SHA-256 does not match the committed manifest")
    if pointer.get("release_manifest_size") != len(manifest_bytes):
        errors.append("stable pointer manifest size does not match the committed manifest")
    return errors


def remote_manifest_errors(pointer: dict[str, Any], manifest_bytes: bytes, *,
                           release_assets: Callable[[str, str], list[dict] | None] | None = None,
                           download_sha256: Callable[[str, str, str], str | None] | None = None,
                           release_state: Callable[[str, str], dict[str, Any] | None] | None = None
                           ) -> list[str]:
    """Verify the remote pointer asset exists and matches the local manifest bytes.

    Uses the publisher's trusted release-asset helpers by default: asset list,
    exact byte size, then downloaded SHA-256.
    """
    import hashlib

    url = pointer.get("release_manifest_url", "")
    if not url:
        return ["stable pointer has no release_manifest_url"]
    if release_assets is None or download_sha256 is None or release_state is None:
        from publish_catalog_release import (
            _download_asset_sha256,
            _find_asset_by_name,
            _parse_release_url,
            _release_assets,
            _release_state,
        )
        release_assets = release_assets or _release_assets
        download_sha256 = download_sha256 or _download_asset_sha256
        release_state = release_state or _release_state
        find_asset = _find_asset_by_name
        parsed = _parse_release_url(url)
    else:
        # Test seams still need the same URL parsing as the publisher.
        from publish_catalog_release import _find_asset_by_name, _parse_release_url
        find_asset = _find_asset_by_name
        parsed = _parse_release_url(url)
    if parsed is None:
        return ["stable pointer release_manifest_url is not a GitHub release asset"]
    repo, tag = parsed
    name = url.rsplit("/", 1)[-1]
    if name != "release-manifest.json":
        return ["stable pointer asset name is not release-manifest.json"]
    state = release_state(repo, tag)
    if state is None:
        return [f"cannot query release state for {tag} in {repo}"]
    if state.get("isDraft") is not False:
        return [f"release {tag} in {repo} is still a draft"]
    if not state.get("publishedAt"):
        return [f"release {tag} in {repo} has no publishedAt timestamp"]
    assets = release_assets(repo, tag)
    if assets is None:
        return [f"cannot query release {tag} in {repo} for the pointer manifest asset"]
    asset = find_asset(assets, name)
    if asset is None:
        return [f"pointer manifest asset {name} is missing from release {tag} in {repo}"]
    if asset.get("size") != len(manifest_bytes):
        return [
            f"pointer manifest asset size {asset.get('size')!r} != local manifest "
            f"size {len(manifest_bytes)}"
        ]
    actual = download_sha256(repo, tag, name)
    if actual is None:
        return [f"failed to download pointer manifest asset {name} for SHA-256 verification"]
    expected = hashlib.sha256(manifest_bytes).hexdigest()
    if actual != expected:
        return [
            f"pointer manifest asset SHA-256 {actual[:12]}... != local manifest "
            f"{expected[:12]}..."
        ]
    return []


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_transition(base_pointer: dict[str, Any], head_pointer: dict[str, Any],
                      manifest: dict[str, Any], manifest_bytes: bytes,
                      record: dict[str, Any] | None,
                      download: Callable[[str], bytes],
                      root: Path = ROOT,
                      remote_manifest_check: Callable[[dict[str, Any], bytes], list[str]] | None = None
                      ) -> list[str]:
    """Return errors for a stable-pointer transition (empty means authorised)."""
    if head_pointer == base_pointer:
        return []
    errors = verify_pointer_binding(head_pointer, manifest, manifest_bytes)
    checker = remote_manifest_check or remote_manifest_errors
    errors.extend(checker(head_pointer, manifest_bytes))
    if head_pointer.get("previous_release_id") != base_pointer.get("release_id"):
        errors.append("stable pointer previous_release_id is not the base stable release")
    base_manifest_path = root / "catalog" / "releases" / f"{base_pointer.get('release_id')}.json"
    if not base_manifest_path.is_file():
        errors.append("promoted-from stable manifest is missing")
    else:
        base_manifest = _read_json(base_manifest_path)
        if manifest.get("producer", {}).get("repo") != \
                base_manifest.get("producer", {}).get("repo"):
            errors.append(
                "candidate producer repository differs from the promoted-from stable release"
            )
    candidate_path = root / "catalog" / "channels" / "candidate.json"
    if record is not None:
        if not candidate_path.is_file():
            errors.append("runtime-only promotion requires the candidate channel pointer")
        else:
            candidate = _read_json(candidate_path)
            if candidate.get("release_id") != manifest["release_id"] or \
                    candidate.get("release_manifest_sha256") != head_pointer.get("release_manifest_sha256") or \
                    candidate.get("release_manifest_size") != head_pointer.get("release_manifest_size"):
                errors.append("candidate pointer does not point at the promoted release")
        errors.extend(qualify_runtime_only(
            manifest, manifest_bytes, record, download, root,
            require_current_pointers=False, expected_baseline=base_pointer,
        ))
    else:
        errors.extend(qualify_release(manifest, download))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", required=True,
                        help="pull request base commit (git fetch-depth: 0)")
    args = parser.parse_args()

    head_path = ROOT / STABLE_PATH
    if not head_path.is_file():
        print("ERROR: stable channel pointer is missing", file=sys.stderr)
        return 1
    try:
        base_pointer = _pointer(_git_show(args.base_ref, STABLE_PATH, cwd=ROOT))
        head_pointer = _pointer(head_path.read_bytes())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if head_pointer == base_pointer:
        print("OK: stable pointer unchanged")
        return 0

    manifest_path = ROOT / "catalog" / "releases" / f"{head_pointer['release_id']}.json"
    if not manifest_path.is_file():
        print(f"ERROR: stable pointer references missing manifest {manifest_path}", file=sys.stderr)
        return 1
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    result = validate_document(manifest, schema="release")
    if not result.ok:
        print("ERROR: stable pointer manifest is invalid:", file=sys.stderr)
        for code in result.all_codes():
            print(f"  {code}", file=sys.stderr)
        return 1

    record = None
    record_path = ROOT / "catalog" / "qualifications" / f"{head_pointer['release_id']}.json"
    if record_path.is_file():
        try:
            record = load_record(head_pointer["release_id"])
        except ValueError as exc:
            print(f"ERROR: invalid runtime-only qualification record: {exc}", file=sys.stderr)
            return 1

    from publish_catalog_release import _download_report

    errors = verify_transition(
        base_pointer, head_pointer, manifest, manifest_bytes, record, _download_report,
    )
    if errors:
        print("ERROR: stable pointer change is not authorised:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    print("OK: stable pointer change is authorised by release evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
