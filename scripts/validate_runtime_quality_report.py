#!/usr/bin/env python3
"""Validate a runtime quality report against the v1 schema + invariants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "quality" / "runtime-quality-report-v1.json"


def validate_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    try:
        jsonschema.validate(report, schema)
    except jsonschema.ValidationError as e:
        errors.append(f"schema: {e.message} at {list(e.absolute_path)}")
        return errors

    threshold = report.get("rtf_threshold", float("inf"))
    for r in report.get("results", []):
        if r.get("shape_errors"):
            errors.append(f"{r['fixture_id']}: shape errors: {r['shape_errors']}")
        if r.get("rtf_warm", 0) > threshold:
            errors.append(f"{r['fixture_id']}: RTF {r['rtf_warm']} > threshold {threshold}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a runtime quality report.")
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

    print(f"OK: runtime quality report valid ({len(report.get('results', []))} result(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
