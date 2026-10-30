#!/usr/bin/env python3
"""Pinned C API headers for runtime archives that shipped without them.

Stable runtime archives published before #161 contain only the shared library,
licenses and build manifest. Measuring such an archive with the native C API
harness needs the exact header closure for its recorded ORT commit and API
level, but the historical archive bytes must not be rewritten. The committed
``ort/pinned-headers.json`` pins the upstream commit and the byte length and
SHA-256 of every header; this module fetches those exact bytes, verifies them,
and refuses to return a set whose closure or API level disagrees with the
runtime's build manifest.

The headers compile only the measurement harness. They never substitute for
the runtime library.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import archive_utils
import ort_api_version

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "ort" / "pinned-headers.json"


class PinnedHeaderError(ValueError):
    """The pinned header set cannot be trusted for this runtime archive."""


@dataclass(frozen=True)
class PinnedHeaders:
    tag: str
    commit_sha: str
    ort_api_version: int
    files: dict[str, bytes]

    def file_records(self) -> list[dict[str, Any]]:
        return [
            {"name": name, "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}
            for name, raw in sorted(self.files.items())
        ]

    def provenance(self) -> dict[str, Any]:
        return {
            "source": "pinned",
            "upstream_commit_sha": self.commit_sha,
            "ort_api_version": self.ort_api_version,
            "files": self.file_records(),
        }


def _lock_entry(commit_sha: str, api_version: int) -> dict[str, Any] | None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    matches = [
        entry for entry in lock["headers"].values()
        if entry["upstream"]["commit_sha"] == commit_sha
        and entry["ort_api_version"] == api_version
    ]
    if len(matches) > 1:
        raise PinnedHeaderError(
            f"duplicate pinned header entries for {commit_sha} API {api_version}"
        )
    return matches[0] if matches else None


def load_for(build_manifest: dict[str, Any]) -> PinnedHeaders | None:
    """Return the verified pin for a runtime build manifest, or None if absent."""
    build = build_manifest.get("build", {})
    commit_sha = build.get("commit_sha") or build_manifest.get("upstream", {}).get("commit_sha")
    api_version = build.get("ort_api_version")
    if not isinstance(commit_sha, str) or not commit_sha:
        raise PinnedHeaderError("runtime build manifest has no upstream commit")
    if type(api_version) is not int or api_version <= 0:
        raise PinnedHeaderError("runtime build manifest has no ORT API version")
    entry = _lock_entry(commit_sha, api_version)
    if entry is None:
        return None
    upstream = entry["upstream"]
    if upstream.get("repo") != "microsoft/onnxruntime":
        raise PinnedHeaderError("pinned header entry points at an unexpected repository")
    files: dict[str, bytes] = {}
    for name, ref in entry["files"].items():
        raw = ort_api_version.pinned_header(upstream, name)
        if len(raw) != ref["size"] or hashlib.sha256(raw).hexdigest() != ref["sha256"]:
            raise PinnedHeaderError(f"pinned header bytes changed upstream: {name}")
        files[name] = raw
    required = ort_api_version.required_headers(files["onnxruntime_c_api.h"], files.__getitem__)
    if set(required) != set(files):
        raise PinnedHeaderError(
            f"pinned header closure mismatch: required {sorted(required)} != pinned {sorted(files)}"
        )
    parsed = ort_api_version.parse_header(files["onnxruntime_c_api.h"].decode("utf-8"))
    if parsed != api_version:
        raise PinnedHeaderError(
            f"pinned header API {parsed} != runtime manifest API {api_version}"
        )
    return PinnedHeaders(
        tag=upstream["tag"],
        commit_sha=commit_sha,
        ort_api_version=api_version,
        files=files,
    )


def for_archive(archive: Path) -> tuple[PinnedHeaders | None, dict[str, Any]]:
    """Return ``(pin, build_manifest)``; the pin is None when headers are packaged."""
    files = archive_utils.safe_read_archive(archive)
    if "build-manifest.json" not in files:
        raise PinnedHeaderError(f"{archive.name}: missing build-manifest.json")
    manifest = json.loads(files["build-manifest.json"])
    packaged = [
        name for name in files
        if name.rsplit("/", 1)[-1].startswith("onnxruntime_") and name.endswith(".h")
    ]
    if packaged:
        return None, manifest
    return load_for(manifest), manifest


def verify_provenance(provenance: dict[str, Any], artifact: dict[str, Any]) -> list[str]:
    """Check a reported header provenance against the catalog runtime artifact."""
    problems: list[str] = []
    api_version = provenance.get("ort_api_version")
    declared_api = artifact.get("runtime", {}).get("ort_c_api_level")
    if str(api_version) != str(declared_api):
        problems.append(
            f"header API {api_version!r} != runtime ort_c_api_level {declared_api!r}"
        )
    commit_sha = provenance.get("upstream_commit_sha")
    if not isinstance(commit_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", commit_sha):
        problems.append("header upstream commit must be a full 40-hex commit")
        return problems
    rows = provenance.get("files") or []
    observed = {(row["name"].rsplit("/", 1)[-1], row["sha256"], row["size"]) for row in rows}
    if len(rows) != len(observed) or not observed:
        problems.append("header provenance files must be unique and non-empty")
        return problems
    source = provenance.get("source")
    if source == "pinned":
        entry = _lock_entry(commit_sha, api_version)
        if entry is None:
            problems.append("no committed pinned header set for this runtime commit/API")
            return problems
        expected = {
            (name, ref["sha256"], ref["size"]) for name, ref in entry["files"].items()
        }
        if observed != expected:
            problems.append("pinned header files differ from the committed pin")
    elif source == "packaged":
        declared = {
            name.rsplit("/", 1)[-1]: (value["sha256"], value["size"])
            for name, value in artifact.get("extracted_file_digests", {}).items()
            if name.rsplit("/", 1)[-1].startswith("onnxruntime_") and name.endswith(".h")
        }
        if not declared:
            problems.append("report claims packaged headers but the runtime declares none")
        elif observed != {(name, sha, size) for name, (sha, size) in declared.items()}:
            problems.append("packaged header provenance differs from the runtime artifact")
    else:
        problems.append(f"unknown header provenance source: {source!r}")
    return problems


def packaged_provenance(build_manifest: dict[str, Any]) -> dict[str, Any]:
    """Provenance for a runtime archive that packages its own header closure."""
    build = build_manifest.get("build", {})
    records = [
        {"name": name, "sha256": value["sha256"], "size": value["size"]}
        for name, value in sorted(build_manifest.get("files", {}).items())
        if name.rsplit("/", 1)[-1].startswith("onnxruntime_") and name.endswith(".h")
    ]
    return {
        "source": "packaged",
        "upstream_commit_sha": build.get("commit_sha") or build_manifest.get("upstream", {}).get("commit_sha"),
        "ort_api_version": build.get("ort_api_version"),
        "files": records,
    }
