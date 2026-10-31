#!/usr/bin/env python3
"""Assemble and verify ``catalog/qualifications/<release-id>.json``.

This helper does not measure anything and cannot manufacture evidence. It
records refs to already-produced report files, runs the full runtime-only
qualification against those exact bytes, and writes the record only when every
binding, budget and coverage check passes.

Upload the report files as immutable GitHub release assets first, then run::

    python scripts/assemble_runtime_only_qualification.py \\
        --release 2026-10-30-001 \\
        --native-report https://github.com/.../runtime-only-native-aarch64-apple-darwin.json=/tmp/native-macos.json \\
        --app-report https://github.com/.../openkara-app-aarch64-apple-darwin.json=/tmp/app-macos.json

The candidate must already be published to the candidate channel and the
current stable pointer must still reference the baseline release. Missing,
failing or partial evidence writes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_runtime_only import qualify_runtime_only  # noqa: E402

RECORD_DIR = ROOT / "catalog" / "qualifications"


def _pair(value: str) -> tuple[str, Path]:
    url, separator, raw_path = value.partition("=")
    if not separator or not url.startswith("https://") or not raw_path:
        raise argparse.ArgumentTypeError("expected <url>=<local-report-file>")
    return url, Path(raw_path)


def _ref(url: str, path: Path) -> dict:
    raw = path.read_bytes()
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: report is not JSON: {exc}") from exc
    for field in ("report_id", "schema_version"):
        if not isinstance(doc.get(field), str) or not doc[field]:
            raise SystemExit(f"{path}: report is missing {field}")
    return {
        "url": url,
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "report_id": doc["report_id"],
        "schema_version": doc["schema_version"],
    }


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--baseline-release", required=True)
    parser.add_argument("--native-report", action="append", type=_pair, required=True,
                        help="<immutable-url>=<local-report-file>; repeat per target")
    parser.add_argument("--app-report", action="append", type=_pair, required=True,
                        help="<immutable-url>=<local-report-file>; repeat per target")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    manifest_path = ROOT / "catalog" / "releases" / f"{args.release}.json"
    baseline_path = ROOT / "catalog" / "releases" / f"{args.baseline_release}.json"
    if not manifest_path.is_file() or not baseline_path.is_file():
        raise SystemExit("candidate or baseline release manifest not found")

    stable = _load(ROOT / "catalog" / "channels" / "stable.json")
    baseline_bytes = baseline_path.read_bytes()
    if stable.get("release_id") != args.baseline_release or \
            stable.get("release_manifest_sha256") != hashlib.sha256(baseline_bytes).hexdigest() or \
            stable.get("release_manifest_size") != len(baseline_bytes):
        raise SystemExit("baseline is not the current stable release; refusing to assemble")

    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    native_refs, app_refs = [], []
    payloads: dict[str, bytes] = {}
    for url, path in args.native_report:
        if url in payloads:
            raise SystemExit(f"duplicate report URL: {url}")
        native_refs.append(_ref(url, path))
        payloads[url] = path.read_bytes()
    for url, path in args.app_report:
        if url in payloads:
            raise SystemExit(f"duplicate report URL: {url}")
        app_refs.append(_ref(url, path))
        payloads[url] = path.read_bytes()

    record = {
        "schema_version": "openkara.runtime-only-qualification/v1",
        "release_id": manifest["release_id"],
        "generation": manifest["generation"],
        "manifest": {
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "size": len(manifest_bytes),
        },
        "baseline_release_id": args.baseline_release,
        "baseline_manifest": {
            "sha256": hashlib.sha256(baseline_bytes).hexdigest(),
            "size": len(baseline_bytes),
        },
        "native_reports": native_refs,
        "app_reports": app_refs,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    def download(url: str) -> bytes:
        if url not in payloads:
            raise ValueError(f"report URL is not one of the supplied files: {url}")
        return payloads[url]

    errors = qualify_runtime_only(manifest, manifest_bytes, record, download)
    if errors:
        print("ERROR: runtime-only qualification did not pass; record not written:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1

    output = args.output or RECORD_DIR / f"{args.release}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"OK: wrote runtime-only qualification record {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
