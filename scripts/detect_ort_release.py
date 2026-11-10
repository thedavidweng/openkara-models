#!/usr/bin/env python3
"""Detect new stable ONNX Runtime releases and compare against the source lock.

Polls the ONNX Runtime GitHub releases API, filters prereleases and non-core
tags (e.g. ``plugin-ep-*``), picks the highest ``vX.Y.Z`` stable tag by
semantic version (not by publish order, since patch releases of older minors
are published after newer minors), and compares it against
ort/source-lock.json. If a newer stable release is available, prints the
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
GITHUB_API = "https://api.github.com/repos/microsoft/onnxruntime/releases?per_page=100"
_STABLE_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def _github_get(url: str, token: str | None = None) -> Any:
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"GitHub API error {e.code}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"network error: {e.reason}") from e


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

    The GitHub releases list is ordered by creation/publish time, so a patch
    for an older minor (e.g. ``v1.30.1``) can appear before a newer minor
    (``v1.31.1``). Ordering by version avoids reporting the older tag as
    "latest".
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


def _tag_commit_sha(repo: str, tag: str, token: str | None = None) -> str:
    """Get the commit SHA for a tag via the GitHub API."""
    url = f"https://api.github.com/repos/{repo}/git/ref/tags/{tag}"
    data = _github_get(url, token)
    ref_type = data.get("object", {}).get("type")
    sha = data.get("object", {}).get("sha", "")
    if ref_type == "tag":
        # Dereference annotated tag to commit.
        url2 = f"https://api.github.com/repos/{repo}/git/tags/{sha}"
        data2 = _github_get(url2, token)
        sha = data2.get("object", {}).get("sha", sha)
    return sha


def detect_latest_stable(token: str | None = None) -> dict[str, Any]:
    """Return the latest (highest semver) stable ORT release info."""
    releases = _github_get(GITHUB_API, token)
    if not isinstance(releases, list):
        raise RuntimeError("unexpected API response: expected list of releases")
    rel = select_latest_stable(releases)
    tag = rel["tag_name"]
    commit_sha = _tag_commit_sha("microsoft/onnxruntime", tag, token)
    return {
        "tag": tag,
        "commit_sha": commit_sha,
        "release_url": rel.get("html_url", ""),
        "published_at": rel.get("published_at", ""),
        "name": rel.get("name", ""),
        "body": (rel.get("body") or "")[:500],
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
    parser.add_argument("--token", default=None, help="GitHub API token (for rate limits).")
    args = parser.parse_args()

    token = args.token or __import__("os").environ.get("GITHUB_TOKEN")

    try:
        latest = detect_latest_stable(token)
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
