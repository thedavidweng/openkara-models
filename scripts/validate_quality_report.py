#!/usr/bin/env python3
"""Validate a quality report against the v1 schema + invariants.

Checks:
  - JSON Schema structure (quality/quality-report-v1.json).
  - No result has NaN or Inf in the ONNX output.
  - All shape_match and expected_shape_match are true.
  - All MSE values are within the tier threshold.
  - Output digests are valid SHA-256.

Usage::

    python scripts/validate_quality_report.py quality-report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "quality" / "quality-report-v1.json"


def validate_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    try:
        jsonschema.validate(report, schema)
    except jsonschema.ValidationError as e:
        errors.append(f"schema: {e.message} at {list(e.absolute_path)}")
        return errors

    threshold = report.get("mse_threshold", float("inf"))
    for r in report.get("results", []):
        if r.get("onnx_has_nan"):
            errors.append(f"{r['fixture_id']}: ONNX output has NaN")
        if r.get("onnx_has_inf"):
            errors.append(f"{r['fixture_id']}: ONNX output has Inf")
        if r.get("shape_match") is False:
            errors.append(f"{r['fixture_id']}: shape mismatch (onnx vs pytorch)")
        if r.get("expected_shape_match") is False:
            errors.append(f"{r['fixture_id']}: expected shape mismatch")
        mse = r.get("mse")
        if mse is not None and mse > threshold:
            errors.append(f"{r['fixture_id']}: MSE {mse} > threshold {threshold}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a quality report.")
    parser.add_argument("report", type=Path, help="Report JSON to validate.")
    args = parser.parse_args()

    if not args.report.is_file():
        print(f"ERROR: report not found: {args.report}", file=sys.stderr)
        return 1

    report = json.loads(args.report.read_text(encoding="utf-8"))
    errors = validate_report(report)
    if errors:
        print(f"FAIL: {len(errors)} error(s):", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1

    print(f"OK: quality report valid ({len(report.get('results', []))} result(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
