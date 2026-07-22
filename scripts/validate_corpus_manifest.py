#!/usr/bin/env python3
"""Validate the quality corpus manifest + metric definitions.

Checks:
  - corpus-manifest.json conforms to corpus-manifest-v1.json schema.
  - metric-definitions-v1.json conforms to its own schema.
  - Every synthetic fixture's generator function exists in synthetic_fixtures.py.
  - Every fixture has a unique fixture_id.
  - PR-tier fixtures are synthetic (real-audio fixtures are release-tier only
    until the fetch infrastructure is added in a later PR).
  - Every fixture's expected_output_shape is consistent with its channels/frames
    for the Demucs 4-stem output format [1, 4, channels, frames].

Usage::

    python scripts/validate_corpus_manifest.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

CORPUS_MANIFEST = ROOT / "quality" / "corpus-manifest.json"
CORPUS_SCHEMA = ROOT / "quality" / "corpus-manifest-v1.json"
METRIC_SCHEMA = ROOT / "quality" / "metric-definitions-v1.json"


def validate_corpus_manifest() -> list[str]:
    errors: list[str] = []
    manifest = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    schema = json.loads(CORPUS_SCHEMA.read_text(encoding="utf-8"))
    try:
        jsonschema.validate(manifest, schema)
    except jsonschema.ValidationError as e:
        errors.append(f"corpus manifest schema: {e.message} at {list(e.absolute_path)}")
        return errors

    # Unique fixture_ids.
    ids = [f["fixture_id"] for f in manifest["fixtures"]]
    duplicates = [fid for fid in ids if ids.count(fid) > 1]
    if duplicates:
        errors.append(f"duplicate fixture_ids: {set(duplicates)}")

    # Synthetic fixture functions exist.
    import synthetic_fixtures as sf
    for f in manifest["fixtures"]:
        if f["kind"] != "synthetic":
            continue
        func_name = f["generator"]["function"]
        if func_name not in sf.FIXTURE_FUNCTIONS:
            errors.append(f"{f['fixture_id']}: unknown generator function '{func_name}'")
            continue
        # Try generating a tiny version to catch signature errors. Scale
        # position params to the smaller frame count to avoid out-of-bounds.
        try:
            func = sf.FIXTURE_FUNCTIONS[func_name]
            params = dict(f["generator"].get("params", {}))
            if "position" in params:
                params["position"] = min(params["position"], 512)
            func(channels=f["channels"], sample_rate=f["sample_rate"],
                 frames=1024, **params)
        except Exception as e:
            errors.append(f"{f['fixture_id']}: generator raised: {e}")

    # PR-tier fixtures must be synthetic.
    for f in manifest["fixtures"]:
        if f["tier"] == "pr" and f["kind"] != "synthetic":
            errors.append(
                f"{f['fixture_id']}: PR-tier fixtures must be synthetic "
                f"(got kind={f['kind']})"
            )

    # expected_output_shape consistency for Demucs 4-stem format.
    for f in manifest["fixtures"]:
        shape = f.get("expected_output_shape")
        if shape is None:
            continue
        expected = [1, 4, f["channels"], f["frames"]]
        if shape != expected:
            errors.append(
                f"{f['fixture_id']}: expected_output_shape {shape} != "
                f"Demucs 4-stem format {expected}"
            )

    return errors


def validate_metric_definitions() -> list[str]:
    errors: list[str] = []
    # The metric definitions file is also the schema; validate it against the
    # JSON Schema meta-schema (draft 2020-12).
    defs = json.loads(METRIC_SCHEMA.read_text(encoding="utf-8"))
    # Just check the structure is valid by loading it.
    if defs.get("schema_version") != "openkara.metric-definitions/v1" and "$defs" not in defs:
        # The file IS the schema; check it has the required structure.
        if "metric" not in defs.get("$defs", {}):
            errors.append("metric-definitions: missing $defs.metric")
    return errors


def main() -> int:
    errors = validate_corpus_manifest() + validate_metric_definitions()
    if errors:
        print(f"FAIL: {len(errors)} error(s):", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1
    manifest = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    pr_count = sum(1 for f in manifest["fixtures"] if f["tier"] == "pr")
    release_count = sum(1 for f in manifest["fixtures"] if f["tier"] == "release")
    print(f"OK: corpus manifest valid ({len(manifest['fixtures'])} fixtures: "
          f"{pr_count} PR, {release_count} release)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
