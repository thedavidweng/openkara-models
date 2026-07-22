#!/usr/bin/env python3
"""Detect HuggingFace model-weight revision changes using immutable commit SHAs.

Polls the HuggingFace API for model-weight revision commit SHAs (not
lastModified timestamps, per issue #20 invariant #1). Compares against the
committed model source lock. If a new revision is available, prints the
info and exits with code 0. If up-to-date, exits with code 1. On API
failure, exits with code 2.

This script does NOT update the lock or open a PR — it only detects. The
candidate generator uses this script's output.

Usage::

    python scripts/detect_model_weight_revision.py --model htdemucs
    python scripts/detect_model_weight_revision.py --model htdemucs --json
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

# Model -> HuggingFace repo mapping.
HF_MODELS = {
    "htdemucs": "facebook/htdemucs",
    "htdemucs_ft": "facebook/htdemucs_ft",
}


def _hf_get(url: str, token: str | None = None) -> Any:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HuggingFace API error {e.code}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"network error: {e.reason}") from e


def _get_model_info(repo: str, token: str | None = None) -> dict[str, Any]:
    """Get model info including all revisions with commit SHAs."""
    url = f"https://huggingface.co/api/models/{repo}"
    return _hf_get(url, token)


def detect_latest_revision(model: str, token: str | None = None) -> dict[str, Any]:
    """Return the latest revision info for a model."""
    repo = HF_MODELS.get(model)
    if not repo:
        raise ValueError(f"unknown model: {model} (known: {list(HF_MODELS.keys())})")
    info = _get_model_info(repo, token)
    # The main revision is in sha (commit SHA of the main branch).
    # siblings contain the file list; we want the commit SHA.
    sha = info.get("sha")
    if not sha:
        raise RuntimeError(f"no sha found for {repo}")
    return {
        "model": model,
        "repo": repo,
        "commit_sha": sha,
        "last_modified": info.get("lastModified", ""),
        "model_id": info.get("modelId", ""),
        "tags": info.get("tags", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect HuggingFace model-weight revisions.")
    parser.add_argument("--model", required=True, choices=list(HF_MODELS.keys()))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--token", default=None, help="HuggingFace API token.")
    parser.add_argument("--lock", type=Path, default=None,
                        help="Model source lock path (for comparison).")
    args = parser.parse_args()

    token = args.token or __import__("os").environ.get("HF_TOKEN")

    try:
        latest = detect_latest_revision(args.model, token)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Compare with lock if provided.
    comparison = None
    if args.lock and args.lock.is_file():
        lock = json.loads(args.lock.read_text(encoding="utf-8"))
        lock_sha = lock.get("commit_sha") or lock.get("weights", {}).get("commit_sha")
        comparison = {
            "lock_commit_sha": lock_sha,
            "latest_commit_sha": latest["commit_sha"],
            "update_available": lock_sha != latest["commit_sha"],
        }

    if args.json:
        print(json.dumps({"latest": latest, "comparison": comparison}, indent=2))
    else:
        print(f"Model: {args.model} ({latest['repo']})")
        print(f"Latest commit: {latest['commit_sha'][:12]}")
        if comparison:
            print(f"Lock commit:   {comparison['lock_commit_sha'][:12] if comparison['lock_commit_sha'] else 'n/a'}")
            if comparison["update_available"]:
                print("UPDATE AVAILABLE")
            else:
                print("Lock is up to date.")

    if comparison:
        return 0 if comparison["update_available"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
