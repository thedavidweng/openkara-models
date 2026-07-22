#!/usr/bin/env python3
"""Validate OpenKara catalog documents against the v1 schema and invariants.

Usage::

    python scripts/validate_catalog.py catalog/releases/<release-id>.json
    python scripts/validate_catalog.py --schema channel catalog/channels/stable.json
    python scripts/validate_catalog.py --all-fixtures

``--all-fixtures`` validates every document under ``catalog/fixtures/valid/``
(must pass) and ``catalog/fixtures/invalid/`` (must fail, optionally checked
against a sibling ``<name>.expected.json`` sidecar declaring
``expected_error_codes`` and ``expected_stage``).

Exit code is non-zero if any document fails (or, for invalid fixtures, fails to
fail as expected).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema

# Make ``scripts/`` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from catalog_model import (  # noqa: E402
    CATALOG_DIR,
    CHANNEL_SCHEMA_PATH,
    FIXTURES_DIR,
    RELEASE_SCHEMA_PATH,
    SCHEMA_VERSION_CHANNEL,
    SCHEMA_VERSION_RELEASE,
    ValidationError,
    validate_channel_invariants,
    validate_release_invariants,
)


@dataclass
class ValidationResult:
    schema_errors: list[str] = field(default_factory=list)
    invariant_errors: list[ValidationError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.schema_errors and not self.invariant_errors

    def all_codes(self) -> list[str]:
        codes: list[str] = []
        codes.extend(f"schema:{e}" for e in self.schema_errors)
        codes.extend(e.code for e in self.invariant_errors)
        return codes


_SCHEMA_CACHE: dict[str, dict[str, Any]] = {}


def _load_schema(path: Path) -> dict[str, Any]:
    key = str(path)
    if key not in _SCHEMA_CACHE:
        with path.open("r", encoding="utf-8") as fh:
            _SCHEMA_CACHE[key] = json.load(fh)
    return _SCHEMA_CACHE[key]


def _detect_schema_version(doc: dict[str, Any]) -> str | None:
    sv = doc.get("schema_version")
    if sv == SCHEMA_VERSION_RELEASE:
        return "release"
    if sv == SCHEMA_VERSION_CHANNEL:
        return "channel"
    return None


def validate_document(
    doc: dict[str, Any],
    schema: str | None = None,
) -> ValidationResult:
    """Validate a single catalog document.

    ``schema`` is ``"release"`` or ``"channel"``; when omitted it is detected
    from the document's ``schema_version`` field.
    """
    if schema is None:
        schema = _detect_schema_version(doc)
    if schema is None:
        return ValidationResult(
            schema_errors=[f"unknown schema_version: {doc.get('schema_version')!r}"]
        )

    schema_doc = _load_schema(
        RELEASE_SCHEMA_PATH if schema == "release" else CHANNEL_SCHEMA_PATH
    )
    validator = jsonschema.Draft202012Validator(schema_doc)
    schema_errors = sorted(
        # Render each jsonschema error as a compact "path: message" string.
        f"{'/'.join(str(p) for p in err.absolute_path) or '<root>'}: {err.message}"
        for err in validator.iter_errors(doc)
    )

    invariant_errors: list[ValidationError] = []
    if not schema_errors:
        if schema == "release":
            invariant_errors = validate_release_invariants(doc)
        else:
            invariant_errors = validate_channel_invariants(doc)

    return ValidationResult(schema_errors=schema_errors, invariant_errors=invariant_errors)


# --------------------------------------------------------------------------- #
# Fixture mode
# --------------------------------------------------------------------------- #


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _validate_fixture_pair(
    name: str, fixture_path: Path, expected: dict[str, Any] | None
) -> tuple[bool, str]:
    """Validate one invalid fixture against its expected sidecar."""
    try:
        doc = _load_json(fixture_path)
    except json.JSONDecodeError as e:
        # Malformed JSON counts as a schema-stage failure.
        if expected and expected.get("expected_stage", "either") not in {"schema", "either"}:
            return False, f"{name}: JSON parse error but expected stage "
        return True, f"{name}: rejected as malformed JSON ({e.msg})"

    result = validate_document(doc)
    if result.ok:
        return False, f"{name}: expected to fail but passed validation"

    if expected:
        stage = expected.get("expected_stage", "either")
        if stage == "schema" and not result.schema_errors:
            return False, f"{name}: expected schema-stage failure but only invariants failed"
        if stage == "invariant" and not result.invariant_errors:
            return False, f"{name}: expected invariant-stage failure but only schema failed"
        expected_codes = set(expected.get("expected_error_codes", []))
        if expected_codes:
            actual_codes = set(result.all_codes())
            if not (expected_codes & actual_codes):
                return False, (
                    f"{name}: expected one of {sorted(expected_codes)} but got "
                    f"{sorted(actual_codes)}"
                )
    return True, f"{name}: rejected as expected ({', '.join(result.all_codes())})"


def run_all_fixtures() -> int:
    """Validate every fixture. Returns process exit code."""
    failures: list[str] = []
    passed = 0

    valid_dir = FIXTURES_DIR / "valid"
    invalid_dir = FIXTURES_DIR / "invalid"

    if not valid_dir.is_dir():
        failures.append(f"missing valid fixtures dir: {valid_dir}")
    if not invalid_dir.is_dir():
        failures.append(f"missing invalid fixtures dir: {invalid_dir}")

    # Valid fixtures: must pass.
    for fixture_path in sorted(valid_dir.glob("*.json")):
        name = fixture_path.name
        try:
            doc = _load_json(fixture_path)
        except json.JSONDecodeError as e:
            failures.append(f"{name}: malformed JSON: {e.msg}")
            continue
        result = validate_document(doc)
        if not result.ok:
            failures.append(
                f"{name}: expected valid but failed: "
                f"schema={result.schema_errors} invariants={[str(e) for e in result.invariant_errors]}"
            )
        else:
            passed += 1

    # Invalid fixtures: must fail, checked against optional sidecar.
    for fixture_path in sorted(invalid_dir.glob("*.json")):
        if fixture_path.name.endswith(".expected.json"):
            continue
        name = fixture_path.name
        sidecar = fixture_path.with_suffix(".expected.json")
        expected = _load_json(sidecar) if sidecar.is_file() else None
        ok, msg = _validate_fixture_pair(name, fixture_path, expected)
        if ok:
            passed += 1
        else:
            failures.append(msg)

    for f in failures:
        print(f"FAIL {f}", file=sys.stderr)
    print(f"fixtures: {passed} passed, {len(failures)} failed")
    return 1 if failures else 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate OpenKara catalog documents (schema + invariants)."
    )
    parser.add_argument("path", nargs="?", type=Path, help="Catalog document to validate.")
    parser.add_argument(
        "--schema",
        choices=["release", "channel"],
        help="Force schema (otherwise detected from schema_version).",
    )
    parser.add_argument(
        "--all-fixtures",
        action="store_true",
        help="Validate every fixture under catalog/fixtures/.",
    )
    args = parser.parse_args()

    if args.all_fixtures:
        return run_all_fixtures()

    if args.path is None:
        parser.error("provide a path or --all-fixtures")

    path: Path = args.path.resolve()
    if not path.is_file():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 1

    try:
        doc = _load_json(path)
    except json.JSONDecodeError as e:
        print(f"ERROR: malformed JSON: {e.msg}", file=sys.stderr)
        return 1

    result = validate_document(doc, schema=args.schema)
    if result.ok:
        print(f"OK: {path}")
        return 0

    for e in result.schema_errors:
        print(f"SCHEMA ERROR: {e}", file=sys.stderr)
    for e in result.invariant_errors:
        print(f"INVARIANT ERROR: {e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
