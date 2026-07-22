#!/usr/bin/env python3
"""Validate a runtime benchmark report against the v1 schema + invariants.

Checks:
  - JSON Schema structure (ort/benchmark-report-v1.json).
  - Every result has no shape_errors (NaN/Inf/shape mismatch would fail the
    native inference gate).
  - No fallback_node_count_mismatch (reduced runtime partitioning must match
    the full baseline).
  - frames == 343980 for release-tier reports (full window).
  - warm_median_s and first_window_s are positive and finite.

Usage::

    python scripts/validate_benchmark_report.py benchmark-report.json
    python scripts/validate_benchmark_report.py --tier release benchmark-report.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "ort" / "benchmark-report-v1.json"
FULL_WINDOW_FRAMES = 343980


def validate_report(report: dict[str, Any], tier: str = "pr") -> list[str]:
    errors: list[str] = []

    # Schema validation.
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    try:
        jsonschema.validate(report, schema)
    except jsonschema.ValidationError as e:
        errors.append(f"schema: {e.message} at {list(e.absolute_path)}")
        return errors

    # Invariant: no shape errors.
    for r in report.get("results", []):
        if r.get("shape_errors"):
            errors.append(f"{r['model']}: shape errors: {r['shape_errors']}")
        if r.get("fallback_node_count_mismatch"):
            errors.append(
                f"{r['model']}: fallback node count mismatch "
                f"{r['fallback_node_count_mismatch']}"
            )
        for key in ("cold_load_s", "first_window_s", "warm_median_s", "warm_p95_s"):
            v = r.get(key)
            if v is not None and (not math.isfinite(v) or v < 0):
                errors.append(f"{r['model']}: {key} is not finite/positive: {v}")

    # Release tier: must be full window.
    if tier == "release":
        if report.get("frames") != FULL_WINDOW_FRAMES:
            errors.append(
                f"release-tier report must use frames={FULL_WINDOW_FRAMES}, "
                f"got {report.get('frames')}"
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a runtime benchmark report.")
    parser.add_argument("report", type=Path, help="Report JSON to validate.")
    parser.add_argument("--tier", choices=["pr", "release"], default="pr",
                        help="Gate tier: pr (default) or release (requires full window).")
    args = parser.parse_args()

    if not args.report.is_file():
        print(f"ERROR: report not found: {args.report}", file=sys.stderr)
        return 1

    report = json.loads(args.report.read_text(encoding="utf-8"))
    errors = validate_report(report, tier=args.tier)
    if errors:
        print(f"FAIL: benchmark report has {len(errors)} error(s):", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1

    print(f"OK: benchmark report valid ({len(report.get('results', []))} result(s), tier={args.tier})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
