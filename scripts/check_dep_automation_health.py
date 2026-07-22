#!/usr/bin/env python3
"""Check dependency automation health.

Compares known upstream stable versions/revisions with committed locks.
If the automation has not created a candidate within a defined interval,
fails and prints a maintenance message.

Checks:
  1. ORT source lock tag is not stale (compared to latest stable GitHub
     release, ignoring prereleases).
  2. The dep-detection workflow has run within the expected interval
     (checked via workflow run timestamp, if GITHUB_TOKEN is available).
  3. No duplicate candidate branches exist for the same upstream version.

Exits 0 if healthy, 1 if stale (candidate needed), 2 if automation
failure detected (duplicate candidates or detection workflow never run),
3 on API error.

Usage::

    python scripts/check_dep_automation_health.py
    python scripts/check_dep_automation_health.py --max-stale-days 14
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
LOCK_PATH = ROOT / "ort" / "source-lock.json"


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


def check_ort_lock_stale(token: str | None = None) -> dict[str, Any]:
    """Check if the ORT source lock is stale (behind latest stable)."""
    import detect_ort_release
    try:
        latest = detect_ort_release.detect_latest_stable(token)
    except RuntimeError as e:
        return {"status": "api_error", "error": str(e)}
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    lock_tag = lock["upstream"]["tag"]
    comparison = detect_ort_release.compare_with_lock(latest, lock)
    return {
        "status": "stale" if comparison["update_available"] else "current",
        "lock_tag": lock_tag,
        "latest_tag": latest["tag"],
        "update_available": comparison["update_available"],
    }


def check_workflow_recency(token: str | None = None, max_days: int = 14) -> dict[str, Any]:
    """Check if the dep-detection workflow has run recently."""
    if not token:
        return {"status": "skipped", "reason": "no GITHUB_TOKEN"}
    try:
        # Get recent workflow runs for dep-detection.yml.
        url = ("https://api.github.com/repos/thedavidweng/openkara-models/"
               "actions/workflows/dep-detection.yml/runs?per_page=5")
        data = _github_get(url, token)
        runs = data.get("workflow_runs", [])
        if not runs:
            return {"status": "no_runs", "reason": "no workflow runs found"}
        latest_run = runs[0]
        run_at = datetime.fromisoformat(
            latest_run["created_at"].replace("Z", "+00:00")
        )
        now = datetime.now(timezone.utc)
        age_days = (now - run_at).days
        return {
            "status": "stale" if age_days > max_days else "current",
            "last_run": latest_run["created_at"],
            "age_days": age_days,
            "max_days": max_days,
        }
    except RuntimeError as e:
        return {"status": "api_error", "error": str(e)}


def check_duplicate_candidates(token: str | None = None) -> dict[str, Any]:
    """Check for duplicate candidate branches."""
    if not token:
        return {"status": "skipped", "reason": "no GITHUB_TOKEN"}
    try:
        url = ("https://api.github.com/repos/thedavidweng/openkara-models/branches?per_page=100")
        data = _github_get(url, token)
        branches = [b["name"] for b in data if isinstance(data, list)]
        candidates = [b for b in branches if b.startswith("deps/candidate-")]
        # Group by prefix to detect duplicates.
        prefixes: dict[str, list[str]] = {}
        for b in candidates:
            # Extract the version part (e.g. "ort-v1.28.0" from "deps/candidate-ort-v1.28.0").
            parts = b.replace("deps/candidate-", "").split("-")
            prefix = "-".join(parts[:2]) if len(parts) >= 2 else parts[0]
            prefixes.setdefault(prefix, []).append(b)
        duplicates = {k: v for k, v in prefixes.items() if len(v) > 1}
        return {
            "status": "duplicates" if duplicates else "clean",
            "candidates": candidates,
            "duplicates": duplicates,
        }
    except RuntimeError as e:
        return {"status": "api_error", "error": str(e)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check dependency automation health.")
    parser.add_argument("--max-stale-days", type=int, default=14,
                        help="Max days without a detection run before stale (default: 14).")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")

    ort_check = check_ort_lock_stale(token)
    workflow_check = check_workflow_recency(token, args.max_stale_days)
    duplicate_check = check_duplicate_candidates(token)

    results = {
        "ort_lock": ort_check,
        "workflow_recency": workflow_check,
        "duplicate_candidates": duplicate_check,
    }

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"ORT lock: {ort_check['status']} (lock={ort_check.get('lock_tag')}, latest={ort_check.get('latest_tag')})")
        print(f"Workflow recency: {workflow_check['status']} (age={workflow_check.get('age_days')}d)")
        print(f"Duplicate candidates: {duplicate_check['status']}")

    # Determine exit code.
    if any(r.get("status") == "api_error" for r in results.values()):
        if not args.json:
            print("ERROR: API failure detected", file=sys.stderr)
        return 3
    if ort_check["status"] == "stale":
        if not args.json:
            print("STALE: ORT lock is behind latest stable release", file=sys.stderr)
        return 1
    if workflow_check["status"] == "stale":
        if not args.json:
            print("STALE: dep-detection workflow has not run recently", file=sys.stderr)
        return 1
    if workflow_check["status"] == "no_runs":
        # The detection workflow has never run. This is NOT healthy — the
        # automation is not functioning. Treat as automation failure (exit 2)
        # so the health check cannot silently report "healthy" when no
        # detection has ever happened.
        if not args.json:
            print("NO_RUNS: dep-detection workflow has never run — automation not functioning",
                  file=sys.stderr)
        return 2
    if duplicate_check["status"] == "duplicates":
        if not args.json:
            print("DUPLICATES: duplicate candidate branches detected", file=sys.stderr)
        return 2
    if not args.json:
        print("OK: automation healthy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
