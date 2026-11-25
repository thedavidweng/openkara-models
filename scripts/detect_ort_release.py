#!/usr/bin/env python3
"""Detect new stable ONNX Runtime releases and compare against the source lock.

Polls the version list of the Microsoft-signed ``Microsoft.ML.OnnxRuntime``
package on nuget.org (the GitHub repository is not a dependency), filters
prereleases, picks the highest ``X.Y.Z`` stable version by semantic version
(not by publish order, since patch releases of older minors are published
after newer minors), reads the source commit from the package's nuspec
``<repository commit="...">``, and compares it against ort/source-lock.json. If a newer stable release is available, prints the
release info and exits with code 0.
If the lock is up-to-date, exits with code 1. On API failure, exits with
code 2.

This script does NOT update the source lock or open a PR — it only detects.
The candidate generator (scripts/generate_dep_candidate.py) uses this script's
output to update the lock and open a candidate PR.

Usage::

    python scripts/detect_ort_release.py
    python scripts/detect_ort_release.py --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "ort" / "source-lock.json"
PACKAGE_ID = "microsoft.ml.onnxruntime"
NUGET_FLAT = "https://api.nuget.org/v3-flatcontainer"
GALLERY = "https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime"
_STABLE_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "openkara-models-dep-detection"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"nuget.org error {e.code}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"network error: {e.reason}") from e


def _nuget_versions() -> list[str]:
    data = json.loads(_http_get(f"{NUGET_FLAT}/{PACKAGE_ID}/index.json").decode("utf-8"))
    versions = data.get("versions") if isinstance(data, dict) else None
    if not isinstance(versions, list):
        raise RuntimeError("unexpected nuget.org response: expected a version list")
    return [str(v) for v in versions]


def _nuspec_commit(version: str) -> str:
    """Read the source commit Microsoft records in the package nuspec."""
    import xml.etree.ElementTree as ET

    raw = _http_get(f"{NUGET_FLAT}/{PACKAGE_ID}/{version}/{PACKAGE_ID}.nuspec")
    root = ET.fromstring(raw)
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] == "repository":
            commit = (element.get("commit") or "").lower()
            if _COMMIT_RE.match(commit):
                return commit
    raise RuntimeError(f"nuspec for {version} has no repository commit")


def _is_prerelease(tag: str) -> bool:
    """Filter prerelease tags: rc, alpha, beta, preview, dev, pre.

    Matches prerelease markers only when they appear as a hyphen-delimited
    segment or followed by digits (e.g. ``v1.28.0-rc1``, ``v1.28.0-alpha``).
    This avoids false positives like ``v1.28.0-architecture`` matching ``rc``.
    """
    low = tag.lower()
    # Match markers at a segment boundary: after '-' or at start, optionally
    # followed by digits/dots/dashes (version suffix), before end or another
    # segment boundary. This avoids false positives like ``v1.28.0-architecture``
    # matching ``rc``.
    pattern = r"(?:^|-)(?:rc|alpha|beta|preview|dev|pre)(?:[\d.-]|$)"
    return re.search(pattern, low) is not None


def _stable_version(tag: str | None) -> tuple[int, int, int] | None:
    """Parse a core ORT stable tag (``vX.Y.Z``) into a comparable tuple.

    Returns ``None`` for prereleases, plugin tags (``plugin-ep-cuda/v0.3.0``)
    and anything else that is not a plain ``vX.Y.Z`` release tag.
    """
    if not tag:
        return None
    m = _STABLE_TAG_RE.match(tag)
    if m is None:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def select_latest_stable(releases: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the release with the highest stable semver tag.

    Release lists are not ordered by version: a patch for an older minor
    (e.g. ``v1.30.1``) can be published after a newer minor (``v1.31.1``).
    Ordering by version avoids reporting the older tag as "latest".
    """
    best: dict[str, Any] | None = None
    best_version: tuple[int, int, int] | None = None
    for rel in releases:
        tag = rel.get("tag_name", "")
        if rel.get("prerelease") or rel.get("draft") or _is_prerelease(tag):
            continue
        version = _stable_version(tag)
        if version is None:
            continue
        if best_version is None or version > best_version:
            best, best_version = rel, version
    if best is None:
        raise RuntimeError("no stable releases found")
    return best


def detect_latest_stable(token: str | None = None) -> dict[str, Any]:
    """Return the latest (highest semver) stable ORT release on nuget.org.

    ``token`` is accepted for interface compatibility and unused: nuget.org
    needs no authentication.
    """
    releases = [{"tag_name": f"v{v}", "prerelease": "-" in v, "version": v} for v in _nuget_versions()]
    rel = select_latest_stable(releases)
    version = rel["version"]
    return {
        "tag": rel["tag_name"],
        "commit_sha": _nuspec_commit(version),
        "release_url": f"{GALLERY}/{version}",
        "published_at": "",
        "name": f"Microsoft.ML.OnnxRuntime {version}",
        "body": "",
    }


def compare_with_lock(latest: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    """Compare the latest stable release with the source lock.

    An update is reported only when the latest stable tag is newer than the
    locked tag, or when the tags match but the tag now resolves to a
    different commit. An older latest tag never counts as an update.
    """
    lock_upstream = lock.get("upstream", {})
    lock_tag = lock_upstream.get("tag")
    lock_sha = lock_upstream.get("commit_sha")
    latest_version = _stable_version(latest["tag"])
    lock_version = _stable_version(lock_tag)
    if latest_version is not None and lock_version is not None:
        if latest_version != lock_version:
            update_available = latest_version > lock_version
        else:
            update_available = latest["commit_sha"] != lock_sha
    else:
        # Unparseable tags: fall back to exact identity comparison.
        update_available = latest["tag"] != lock_tag or latest["commit_sha"] != lock_sha
    return {
        "lock_tag": lock_tag,
        "lock_commit_sha": lock_sha,
        "latest_tag": latest["tag"],
        "latest_commit_sha": latest["commit_sha"],
        "update_available": update_available,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect new stable ORT releases.")
    parser.add_argument("--json", action="store_true", help="Output JSON.")
    args = parser.parse_args()

    try:
        latest = detect_latest_stable()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    comparison = compare_with_lock(latest, lock)

    if args.json:
        print(json.dumps({"latest": latest, "comparison": comparison}, indent=2))
    else:
        print(f"Latest stable: {latest['tag']} ({latest['commit_sha'][:12]})")
        print(f"Lock:          {comparison['lock_tag']} ({comparison['lock_commit_sha'][:12]})")
        if comparison["update_available"]:
            print(f"UPDATE AVAILABLE: {comparison['lock_tag']} -> {comparison['latest_tag']}")
        else:
            print("Lock is up to date.")

    return 0 if comparison["update_available"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
