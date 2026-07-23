#!/usr/bin/env python3
"""Download a stable model from the catalog and verify its SHA-256 digest.

Used by the ORT build workflow's "Download stable model from catalog" step
on every target. Replaces an inline ``python - <<'PY'`` heredoc that does
not work under PowerShell on Windows runners.

Usage::

    python scripts/download_stable_model.py \\
        --catalog catalog/releases/2026-07-20-001.json \\
        --artifact htdemucs.balanced.fp32.onnx \\
        --dest models/htdemucs.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path,
                        help="Catalog release JSON file.")
    parser.add_argument("--artifact", required=True,
                        help="artifact_id to download from the catalog.")
    parser.add_argument("--dest", required=True, type=Path,
                        help="Destination path for the downloaded model.")
    args = parser.parse_args()

    if not args.catalog.is_file():
        print(f"ERROR: catalog not found: {args.catalog}", file=sys.stderr)
        return 1

    manifest = json.loads(args.catalog.read_text())
    models = manifest.get("artifacts", {}).get("models", [])
    match = next((a for a in models
                  if a.get("artifact_id") == args.artifact), None)
    if match is None:
        print(f"ERROR: artifact {args.artifact!r} not in catalog "
              f"{args.catalog}", file=sys.stderr)
        return 1

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {match['download_url']} -> {args.dest}")
    urllib.request.urlretrieve(match["download_url"], args.dest)

    got = hashlib.sha256(args.dest.read_bytes()).hexdigest()
    want = match["archive_digest"]
    if got != want:
        print(f"ERROR: sha256 mismatch: got {got}, expected {want}",
              file=sys.stderr)
        return 1
    print(f"model downloaded + verified (sha256={got})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
