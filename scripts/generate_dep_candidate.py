#!/usr/bin/env python3
"""Generate a dependency update candidate: update locks + summary.

Given a detection result (from detect_ort_release.py or
detect_model_weight_revision.py), this script:
  1. For ORT updates, verifies the release tag/commit against nuget.org's
     nuspec (no GitHub API calls), downloads the Microsoft.ML.OnnxRuntime
     NuGet package, verifies its Microsoft author + nuget.org repository
     signatures via ``dotnet nuget verify --all``, refreshes
     ``ort/nuget-lock.json`` (package metadata, signers, targets, and every
     used member's size + SHA-256) and updates ``ort/source-lock.json``.
  2. For model updates, updates ``models/source-lock.json``.
  3. Validates the updated locks.
  4. Prints a JSON + Markdown summary for the candidate PR body, explicitly
     flagging Intel macOS carry-forward when NuGet ships no ``osx-x64`` binary.

Usage::

    python scripts/generate_dep_candidate.py \\
        --ort-tag v1.32.0 --ort-commit fab30a46491a40f5e4c968b80e8a59acbbb91b53 \\
        --output candidate-summary.json

    python scripts/generate_dep_candidate.py \\
        --model htdemucs --model-commit def456 \\
        --output candidate-summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "ort" / "source-lock.json"
NUGET_LOCK_PATH = ROOT / "ort" / "nuget-lock.json"
MODEL_LOCK_PATH = ROOT / "models" / "source-lock.json"
PACKAGE_ID = "Microsoft.ML.OnnxRuntime"
APPLE_TEAM_ID = "UBF8T346G9"
_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")

sys.path.insert(0, str(ROOT / "scripts"))


def update_model_lock(model: str, commit_sha: str) -> dict[str, Any]:
    """Update the model source lock with a new weights revision commit SHA."""
    lock = json.loads(MODEL_LOCK_PATH.read_text(encoding="utf-8"))
    entry = lock.get("models", {}).get(model)
    if entry is None:
        raise SystemExit(f"ERROR: model {model!r} not in {MODEL_LOCK_PATH}")
    old_commit = entry["weights"]["commit_sha"]
    entry["weights"]["commit_sha"] = commit_sha
    MODEL_LOCK_PATH.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "type": "model",
        "model": model,
        "old_commit": old_commit,
        "new_commit": commit_sha,
        "lock_path": str(MODEL_LOCK_PATH),
    }


def _retarget_version(targets: dict[str, Any], old_version: str, new_version: str) -> dict[str, Any]:
    """Rewrite versioned root/filename/symlink strings in NuGet target specs."""
    old_major = old_version.split(".")[0]
    new_major = new_version.split(".")[0]

    def rewrite(text: str) -> str:
        updated = text.replace(f"-{old_version}", f"-{new_version}").replace(f".{old_version}", f".{new_version}")
        if old_major != new_major:
            updated = updated.replace(f".{old_major}.", f".{new_major}.")
        return updated

    updated_targets: dict[str, Any] = {}
    for target, config in targets.items():
        updated_targets[target] = {
            **config,
            "root": rewrite(config["root"]),
            "files": {rewrite(dest): source for dest, source in config["files"].items()},
            "symlinks": {rewrite(dest): rewrite(link) for dest, link in config.get("symlinks", {}).items()},
        }
    return updated_targets


def _current_intel_summary(root: Path = ROOT) -> dict[str, Any] | None:
    """Read the active Intel macOS runtime from stable-v2, if present."""
    pointer_path = root / "catalog" / "channels" / "stable-v2.json"
    if not pointer_path.is_file():
        return None
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    release_path = root / "catalog" / "releases" / f"{pointer['release_id']}.json"
    if not release_path.is_file():
        return None
    release = json.loads(release_path.read_text(encoding="utf-8"))
    for runtime in release.get("artifacts", {}).get("runtimes", []):
        if runtime.get("target_triple") == "x86_64-apple-darwin" and not runtime.get("deprecation", {}).get("deprecated"):
            return {
                "release_id": pointer["release_id"],
                "artifact_id": runtime["artifact_id"],
                "filename": runtime["filename"],
                "download_url": runtime["download_url"],
                "archive_digest": runtime["archive_digest"],
                "tag": runtime.get("upstream", {}).get("tag", ""),
            }
    return None


def update_ort_lock(
    tag: str,
    commit_sha: str,
    cache_dir: Path | None = None,
    reuse_cache: bool = False,
) -> dict[str, Any]:
    """Verify the NuGet release, refresh ort/nuget-lock.json and ort/source-lock.json."""
    import nuget_package
    import ort_api_version
    import repackage_nuget_runtime
    from validate_source_lock import validate_lock

    tag_match = _TAG_RE.fullmatch(tag)
    if tag_match is None:
        raise ValueError(f"ORT tag must be stable vX.Y.Z, got {tag!r}")
    if not re.fullmatch(r"[0-9a-f]{40}", commit_sha):
        raise ValueError("ORT commit must be a full lowercase SHA")

    version = tag[1:]
    if nuget_package.nuspec_commit(PACKAGE_ID, version) != commit_sha:
        raise ValueError("ORT tag does not resolve to the requested commit")

    source_lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    nuget_lock = json.loads(NUGET_LOCK_PATH.read_text(encoding="utf-8"))
    old_tag = source_lock["upstream"]["tag"]
    old_commit = source_lock["upstream"]["commit_sha"]
    old_version = nuget_lock["package"]["version"]

    pkg_url = nuget_package.package_url(PACKAGE_ID, version)
    dest = (cache_dir or (ROOT / "ort" / "nuget-cache")) / f"{PACKAGE_ID}.{version}.nupkg"
    nupkg = nuget_package.download_unpinned(pkg_url, dest, reuse_existing=reuse_cache)
    pkg_size, pkg_sha256 = nuget_package.sha256_file(nupkg)
    verify_output = nuget_package.verify_signatures(nupkg, PACKAGE_ID, version)
    signers = nuget_package.parse_signers(verify_output)

    targets = _retarget_version(nuget_lock["targets"], old_version, version)
    common = dict(nuget_lock["common"])
    include_prefix = common["include"].rstrip("/") + "/"

    with zipfile.ZipFile(nupkg) as package:
        names = set(package.namelist())
        has_osx_x64 = "runtimes/osx-x64/native/libonnxruntime.dylib" in names
        header_names = sorted(
            name for name in names
            if name.startswith(include_prefix) and not name.endswith("/")
        )
        if f"{include_prefix}onnxruntime_c_api.h" not in header_names:
            raise ValueError(f"{PACKAGE_ID} {version} is missing {include_prefix}onnxruntime_c_api.h")

        def read_header(name: str) -> bytes:
            return package.read(f"{include_prefix}{name}")

        required_headers = ort_api_version.required_headers(read_header("onnxruntime_c_api.h"), read_header)
        for req_header in required_headers:
            if f"{include_prefix}{req_header}" not in header_names:
                raise ValueError(f"missing required C API header: {include_prefix}{req_header}")
        api_version = ort_api_version.parse_header(read_header("onnxruntime_c_api.h").decode("utf-8"))

        member_names: list[str] = [common["LICENSE"], common["ThirdPartyNotices.txt"], *header_names]
        for config in targets.values():
            for source in config["files"].values():
                if source not in member_names:
                    member_names.append(source)

        old_members = nuget_lock.get("members", {})
        members: dict[str, Any] = {}
        for name in sorted(set(member_names)):
            if name not in names:
                raise ValueError(f"{PACKAGE_ID} {version} missing required member: {name}")
            data = package.read(name)
            digest = hashlib.sha256(data).hexdigest()
            entry: dict[str, Any] = {"size": len(data), "sha256": digest}
            if name == "runtimes/osx-arm64/native/libonnxruntime.dylib":
                if f"Developer ID Application: Microsoft Corporation ({APPLE_TEAM_ID})".encode() not in data:
                    raise ValueError(f"{name}: missing Microsoft Developer ID signature ({APPLE_TEAM_ID})")
                entry["apple_team_id"] = APPLE_TEAM_ID
                old_note = old_members.get(name, {}).get("note")
                if old_note:
                    entry["note"] = old_note
            if version == old_version:
                prev = old_members.get(name, {})
                if prev.get("size") == entry["size"] and prev.get("sha256") == entry["sha256"]:
                    for extra in ("pre_outage_member", "note"):
                        if extra in prev:
                            entry[extra] = prev[extra]
            members[name] = entry

    updated_nuget_lock = {
        "schema_version": nuget_lock["schema_version"],
        "upstream": {
            "project": nuget_lock["upstream"]["project"],
            "tag": tag,
            "commit_sha": commit_sha,
        },
        "package": {
            "id": PACKAGE_ID,
            "version": version,
            "url": pkg_url,
            "gallery_url": nuget_package.gallery_url(PACKAGE_ID, version),
            "size": pkg_size,
            "sha256": pkg_sha256,
            "signers": signers,
        },
        "common": common,
        "targets": targets,
        "members": members,
    }
    nuget_errors = repackage_nuget_runtime.validate_lock(updated_nuget_lock)
    if nuget_errors:
        raise ValueError("; ".join(nuget_errors))

    source_lock["upstream"]["tag"] = tag
    source_lock["upstream"]["commit_sha"] = commit_sha
    source_lock["upstream"]["release_url"] = nuget_package.gallery_url(PACKAGE_ID, version)
    source_errors = validate_lock(source_lock)
    if source_errors:
        raise ValueError("; ".join(source_errors))

    NUGET_LOCK_PATH.write_text(json.dumps(updated_nuget_lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOCK_PATH.write_text(json.dumps(source_lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    previous_intel = _current_intel_summary()
    return {
        "type": "ort",
        "old_tag": old_tag,
        "new_tag": tag,
        "old_commit": old_commit,
        "new_commit": commit_sha,
        "lock_path": str(LOCK_PATH),
        "nuget_lock_path": str(NUGET_LOCK_PATH),
        "ort_api_version": api_version,
        "nuget_package": updated_nuget_lock["package"],
        "source_lock_submodules_refreshed": False,
        "intel_macos": {
            "status": "nuget" if has_osx_x64 else "carried-forward-on-upstream-unavailable",
            "has_nuget_osx_x64": has_osx_x64,
            "previous": previous_intel,
        },
    }


def generate_candidate_summary(changes: dict[str, Any]) -> str:
    """Generate a Markdown summary for the candidate PR body."""
    lines = ["## Dependency update candidate", ""]
    if changes["type"] == "ort":
        pkg = changes.get("nuget_package", {})
        signers = pkg.get("signers", {})
        intel = changes.get("intel_macos", {})
        prev_intel = intel.get("previous") or {}
        lines.extend([
            "**Type:** ONNX Runtime NuGet update",
            f"**Old:** `{changes['old_tag']}` (`{changes['old_commit'][:12]}`)",
            f"**New:** `{changes['new_tag']}` (`{changes['new_commit'][:12]}`)",
        ])
        if pkg:
            lines.extend([
                f"**NuGet package:** `{pkg['id']} {pkg['version']}` (`{pkg['size']}` bytes, sha256 `{pkg['sha256']}`)",
                f"**NuGet URL:** {pkg['url']}",
                f"**Verified signers:** author `{signers.get('author', {}).get('sha256_fingerprint', 'n/a')[:16]}...`, "
                f"repository `{signers.get('repository', {}).get('sha256_fingerprint', 'n/a')[:16]}...`",
            ])
        if changes.get("ort_api_version") is not None:
            lines.append(f"**ORT C API version:** `{changes['ort_api_version']}`")

        lines.extend(["", "### Intel macOS (`x86_64-apple-darwin`) status", ""])
        if intel.get("has_nuget_osx_x64"):
            lines.append("- `Microsoft.ML.OnnxRuntime` ships `osx-x64` in this release.")
        else:
            prev_desc = (
                f"`{prev_intel.get('tag') or 'unknown'}` (`{prev_intel['artifact_id']}`, "
                f"sha256 `{prev_intel['archive_digest'][:12]}...`)"
                if prev_intel else "the previous verified `stable-v2` Intel macOS archive"
            )
            lines.extend([
                "- **Degraded (carried forward if upstream source is unavailable):** `Microsoft.ML.OnnxRuntime` on nuget.org does not ship an `osx-x64` binary.",
                f"- If `github.com/microsoft/onnxruntime` cannot be cloned, `ort-build.yml` keeps {prev_desc} and runs Rosetta smoke against it rather than failing this candidate.",
            ])

        if not changes.get("source_lock_submodules_refreshed", True):
            lines.extend([
                "",
                "### Source lock note",
                "",
                "- Derived `(tag, commit_sha)` from nuget.org without calling the GitHub API; `ort/source-lock.json` `submodules` and `deps.entries` were preserved as-is.",
            ])

        lines.extend([
            "",
            "### Required coupled changes",
            "",
            "- [ ] Verify the signed NuGet repackage and native smoke tests (`official-runtime-smoke.yml`)",
            "- [ ] Pass Linux runtime differential & performance gates (`quality-gates.yml`)",
            "- [ ] Check Intel macOS runtime build or verified carry-forward (`ort-build.yml`)",
            "- [ ] Publish and promote the v2 catalog (`catalog-release-v2.yml`) after review",
            "",
            "### Review checklist",
            "",
            "- [ ] Runtime comparison passes",
            "- [ ] NuGet signature verification and member digests match",
            "- [ ] Intel macOS status (rebuilt or carried forward) is acceptable for this release",
            "- [ ] No stable pointer changed",
        ])
    elif changes["type"] == "model":
        lines.extend([
            "**Type:** Model-weight revision update",
            f"**Model:** `{changes['model']}`",
            f"**Old commit:** `{changes['old_commit'][:12]}`" if changes.get("old_commit") else "**Old commit:** n/a",
            f"**New commit:** `{changes['new_commit'][:12]}`",
            "",
            "### Required coupled changes",
            "",
            "- [ ] Re-convert affected ONNX models (convert.yml)",
            "- [ ] Verify export fidelity and runtime compatibility",
            "- [ ] Compare candidate and stable runtime outputs",
        ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a dependency update candidate.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ort-tag", help="New ORT release tag.")
    group.add_argument("--model", help="Model name for weight revision update.")
    parser.add_argument("--ort-commit", help="New ORT commit SHA (required with --ort-tag).")
    parser.add_argument("--model-commit", help="New model commit SHA (required with --model).")
    parser.add_argument("--cache", type=Path, default=ROOT / "ort" / "nuget-cache",
                        help="NuGet download cache directory.")
    parser.add_argument("--reuse-cache", action="store_true",
                        help="Reuse an existing cached .nupkg instead of re-downloading (signatures are still verified).")
    parser.add_argument("--output", type=Path, default=Path("candidate-summary.json"),
                        help="Output JSON summary path.")
    parser.add_argument("--markdown", type=Path, default=None,
                        help="Optional Markdown summary path.")
    args = parser.parse_args()

    if args.ort_tag:
        if not args.ort_commit:
            parser.error("--ort-commit is required with --ort-tag")
        changes = update_ort_lock(args.ort_tag, args.ort_commit, cache_dir=args.cache, reuse_cache=args.reuse_cache)
    elif args.model:
        if not args.model_commit:
            parser.error("--model-commit is required with --model")
        changes = update_model_lock(args.model, args.model_commit)
    else:
        parser.error("provide --ort-tag or --model")

    args.output.write_text(json.dumps(changes, indent=2, sort_keys=True) + "\n")
    print(f"summary: {args.output}")

    if args.markdown:
        args.markdown.write_text(generate_candidate_summary(changes))
        print(f"markdown: {args.markdown}")

    print(generate_candidate_summary(changes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
