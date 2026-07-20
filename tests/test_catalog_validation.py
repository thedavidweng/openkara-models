"""Unit tests for the OpenKara catalog schema, validator, and fixtures.

Covers the issue #18 PR 1 required cases: missing digests, duplicate IDs,
incompatible links, target mismatch, non-monotonic generation, and unknown
required schema fields. Also guards that fixtures stay in sync with the
generator.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from catalog_model import (  # noqa: E402
    CatalogIntegrityError,
    assert_generations_monotonic,
    validate_channel_invariants,
    validate_release_invariants,
)
from generate_catalog_fixtures import (  # noqa: E402
    CHANNEL_MUTATIONS,
    RELEASE_MUTATIONS,
)
from validate_catalog import validate_document  # noqa: E402

FIXTURES = ROOT_DIR / "catalog" / "fixtures"
VALID = FIXTURES / "valid"
INVALID = FIXTURES / "invalid"


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _valid_release() -> dict:
    return _load(VALID / "release-minimal.json")


def _valid_channel() -> dict:
    return _load(VALID / "channel-stable.json")


class SchemaStructureTests(unittest.TestCase):
    def test_valid_release_passes(self):
        self.assertTrue(validate_document(_valid_release()).ok)

    def test_valid_channel_passes(self):
        self.assertTrue(validate_document(_valid_channel()).ok)

    def test_unknown_schema_version_rejected(self):
        doc = _valid_release()
        doc["schema_version"] = "openkara.catalog/v999"
        result = validate_document(doc)
        self.assertFalse(result.ok)
        self.assertTrue(any("schema_version" in e for e in result.schema_errors))

    def test_unknown_top_level_field_rejected(self):
        doc = _valid_release()
        doc["release_digest"] = "deadbeef"
        self.assertFalse(validate_document(doc).ok)

    def test_missing_required_field_rejected(self):
        doc = _valid_release()
        del doc["producer"]
        result = validate_document(doc)
        self.assertFalse(result.ok)
        self.assertTrue(any("producer" in e for e in result.schema_errors))

    def test_model_without_required_operator_config_passes(self):
        """Pre-#19 model releases may omit required_operator_config; #19 PR 2
        back-fills it and makes it required for new releases."""
        doc = _valid_release()
        doc["artifacts"]["models"][0]["model"].pop("required_operator_config")
        self.assertTrue(validate_document(doc).ok)

    def test_release_without_supply_chain_or_gates_passes(self):
        """v1 requires identity + integrity only. supply_chain (#18 PR 3) and
        gates (#21) are optional until their owning issues back-fill them."""
        doc = _valid_release()
        doc.pop("supply_chain")
        doc.pop("gates")
        for kind in ("models", "runtimes", "bundles"):
            for art in doc["artifacts"][kind]:
                art.pop("supply_chain", None)
        self.assertTrue(validate_document(doc).ok)


class InvariantTests(unittest.TestCase):
    def test_missing_file_digest_detected(self):
        doc = _valid_release()
        doc["artifacts"]["models"][0]["extracted_file_digests"]["htdemucs.onnx"].pop(
            "sha256"
        )
        # Missing required key is a schema-stage failure.
        result = validate_document(doc)
        self.assertFalse(result.ok)

    def test_duplicate_artifact_id_detected(self):
        doc = _valid_release()
        doc["artifacts"]["bundles"][0]["artifact_id"] = doc["artifacts"]["models"][0][
            "artifact_id"
        ]
        codes = {e.code for e in validate_release_invariants(doc)}
        self.assertIn("duplicate_artifact_id", codes)

    def test_dangling_runtime_ref_detected(self):
        doc = _valid_release()
        doc["artifacts"]["models"][0]["model"]["compatible_runtime_ids"][0] = (
            "ort.does-not-exist"
        )
        codes = {e.code for e in validate_release_invariants(doc)}
        self.assertIn("dangling_runtime_ref", codes)

    def test_dangling_model_ref_detected(self):
        doc = _valid_release()
        doc["artifacts"]["runtimes"][0]["runtime"]["supported_model_artifact_ids"][0] = (
            "model.does-not-exist"
        )
        codes = {e.code for e in validate_release_invariants(doc)}
        self.assertIn("dangling_model_ref", codes)

    def test_bundle_target_mismatch_detected(self):
        doc = _valid_release()
        doc["artifacts"]["bundles"][0]["target_triple"] = "x86_64-unknown-linux-gnu"
        codes = {e.code for e in validate_release_invariants(doc)}
        self.assertIn("target_mismatch", codes)

    def test_bundle_provider_mismatch_detected(self):
        doc = _valid_release()
        doc["artifacts"]["bundles"][0]["execution_provider"] = "directml"
        codes = {e.code for e in validate_release_invariants(doc)}
        self.assertIn("provider_mismatch", codes)

    def test_mutable_download_url_detected(self):
        doc = _valid_release()
        doc["artifacts"]["models"][0]["download_url"] = (
            "https://example.com/latest/htdemucs.onnx"
        )
        codes = {e.code for e in validate_release_invariants(doc)}
        self.assertIn("mutable_download_url", codes)

    def test_compatibility_edge_target_must_match_runtime(self):
        doc = _valid_release()
        doc["compatibility"][0]["runtime_artifact_id"] = "ort.does-not-exist"
        codes = {e.code for e in validate_release_invariants(doc)}
        self.assertIn("dangling_runtime_ref", codes)

    def test_deprecation_replacement_must_exist(self):
        doc = _valid_release()
        doc["artifacts"]["models"][0]["deprecation"]["deprecated"] = True
        doc["artifacts"]["models"][0]["deprecation"]["replacement_artifact_id"] = (
            "gone.artifact"
        )
        codes = {e.code for e in validate_release_invariants(doc)}
        self.assertIn("dangling_replacement", codes)

    def test_target_arch_must_match_target_triple(self):
        doc = _valid_release()
        doc["artifacts"]["runtimes"][0]["arch"] = "x86_64"  # target is aarch64
        codes = {e.code for e in validate_release_invariants(doc)}
        self.assertIn("target_arch_mismatch", codes)

    def test_channel_mutable_url_detected(self):
        doc = _valid_channel()
        doc["release_manifest_url"] = (
            "https://example.com/latest/release-manifest.json"
        )
        codes = {e.code for e in validate_channel_invariants(doc)}
        self.assertIn("mutable_manifest_url", codes)


class GenerationMonotonicityTests(unittest.TestCase):
    def _rel(self, release_id: str, generation: int) -> dict:
        doc = _valid_release()
        doc["release_id"] = release_id
        doc["generation"] = generation
        return doc

    def test_monotonic_generations_pass(self):
        assert_generations_monotonic(
            [self._rel("2026-07-20-001", 1), self._rel("2026-07-21-001", 2)]
        )

    def test_non_monotonic_generation_rejected(self):
        with self.assertRaises(CatalogIntegrityError):
            assert_generations_monotonic(
                [self._rel("2026-07-21-001", 1), self._rel("2026-07-20-001", 2)]
            )

    def test_duplicate_release_id_rejected(self):
        with self.assertRaises(CatalogIntegrityError):
            assert_generations_monotonic(
                [self._rel("2026-07-20-001", 1), self._rel("2026-07-20-001", 1)]
            )


class FixtureCoverageTests(unittest.TestCase):
    """Every declared mutation must have a matching generated fixture that
    fails validation as expected."""

    def test_all_release_mutations_have_failing_fixtures(self):
        for name, stage, codes, _ in RELEASE_MUTATIONS:
            path = INVALID / f"release-{name}.json"
            self.assertTrue(path.is_file(), f"missing fixture {path}")
            doc = _load(path)
            result = validate_document(doc)
            self.assertFalse(result.ok, f"{name} should fail validation")
            if stage == "schema":
                self.assertTrue(result.schema_errors, f"{name} expected schema failure")
            elif stage == "invariant":
                self.assertTrue(
                    result.invariant_errors, f"{name} expected invariant failure"
                )
            if codes:
                actual = {e.code for e in result.invariant_errors}
                self.assertTrue(
                    set(codes) & actual,
                    f"{name} expected one of {codes}, got {actual}",
                )

    def test_all_channel_mutations_have_failing_fixtures(self):
        for name, stage, codes, _ in CHANNEL_MUTATIONS:
            path = INVALID / f"channel-{name}.json"
            self.assertTrue(path.is_file(), f"missing fixture {path}")
            doc = _load(path)
            result = validate_document(doc)
            self.assertFalse(result.ok, f"{name} should fail validation")
            if codes:
                actual = {e.code for e in result.invariant_errors}
                self.assertTrue(set(codes) & actual, f"{name} got {actual}")


class FixturesAreFreshTests(unittest.TestCase):
    """Regenerate fixtures and require no diff — guards against hand edits."""

    def test_fixtures_match_generator(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "generate_catalog_fixtures.py")],
            capture_output=True,
            text=True,
            cwd=str(ROOT_DIR),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        diff = subprocess.run(
            ["git", "diff", "--exit-code", "--", "catalog/fixtures"],
            capture_output=True,
            text=True,
            cwd=str(ROOT_DIR),
        )
        self.assertEqual(
            diff.returncode,
            0,
            "catalog/fixtures are stale; regenerate with "
            "python scripts/generate_catalog_fixtures.py\n" + diff.stdout,
        )


if __name__ == "__main__":
    unittest.main()
