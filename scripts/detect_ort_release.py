#!/usr/bin/env python3
"""Detect new stable ONNX Runtime releases and compare against the source lock.

Polls the ONNX Runtime GitHub releases API, filters prereleases, and compares
the latest stable tag + commit SHA against ort/source-lock.json. If a new
stable release is available, prints the release info and exits with code 0.
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
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "ort" / "source-lock.json"
GITHUB_API = "https://api.github.com/repos/microsoft/onnxruntime/releases"


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
    """Filter prerelease tags: rc, alpha, beta, preview, dev."""
    low = tag.lower()
    markers = ("rc", "alpha", "beta", "preview", "dev", "-pre")
    return any(m in low for m in markers)


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
    """Return the latest stable ORT release info."""
    releases = _github_get(GITHUB_API, token)
    if not isinstance(releases, list):
        raise RuntimeError("unexpected API response: expected list of releases")
    for rel in releases:
        tag = rel.get("tag_name", "")
        if not tag:
            continue
        if rel.get("prerelease") or _is_prerelease(tag):
            continue
        commit_sha = _tag_commit_sha("microsoft/onnxruntime", tag, token)
        return {
            "tag": tag,
            "commit_sha": commit_sha,
            "release_url": rel.get("html_url", ""),
            "published_at": rel.get("published_at", ""),
            "name": rel.get("name", ""),
            "body": (rel.get("body") or "")[:500],
        }
    raise RuntimeError("no stable releases found")


def compare_with_lock(latest: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    """Compare the latest stable release with the source lock."""
    lock_upstream = lock.get("upstream", {})
    return {
        "lock_tag": lock_upstream.get("tag"),
        "lock_commit_sha": lock_upstream.get("commit_sha"),
        "latest_tag": latest["tag"],
        "latest_commit_sha": latest["commit_sha"],
        "update_available": (
            latest["tag"] != lock_upstream.get("tag") or
            latest["commit_sha"] != lock_upstream.get("commit_sha")
        ),
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
