"""Tests for the immutable release-manifest generator (issue #18 PR 2)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from validate_catalog import validate_document  # noqa: E402

SPEC = ROOT_DIR / "catalog" / "specs" / "2026-07-20-001.spec.json"
RELEASE = ROOT_DIR / "catalog" / "releases" / "2026-07-20-001.json"
POINTER = ROOT_DIR / "catalog" / "channels" / "stable.json"
LATEST = ROOT_DIR / "latest.json"


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _run(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "generate_catalog_release.py"), *args],
        capture_output=True,
        text=True,
        cwd=str(ROOT_DIR),
        **kw,
    )


class GeneratedCatalogTests(unittest.TestCase):
    def test_generated_manifest_validates(self):
        self.assertTrue(validate_document(_load(RELEASE), schema="release").ok)

    def test_generated_pointer_validates(self):
        self.assertTrue(validate_document(_load(POINTER), schema="channel").ok)

    def test_latest_json_matches_pr165_shape(self):
        adapter = _load(LATEST)
        # PR #165 fetches .htdemucs.{url,sha256,tag} and .htdemucs_ft.{...}
        for variant in ("htdemucs", "htdemucs_ft"):
            self.assertIn(variant, adapter)
            entry = adapter[variant]
            self.assertEqual({"tag", "url", "sha256", "size"}, set(entry))
            self.assertTrue(entry["url"].startswith("https://"))
            self.assertTrue(entry["tag"])
            self.assertIsInstance(entry["size"], int)
            self.assertEqual(len(entry["sha256"]), 64)

    @staticmethod
    def _stable_four_stem_models() -> dict[str, dict]:
        """variant -> the stable manifest's selectable four-stem artifact
        (the same rule _build_latest_adapter applies: four-stem stem profile,
        waveform tensor interface, not deprecated)."""
        pointer = _load(POINTER)
        manifest = _load(ROOT_DIR / "catalog" / "releases" / f"{pointer['release_id']}.json")
        selected: dict[str, dict] = {}
        for model in manifest["artifacts"]["models"]:
            if model.get("model", {}).get("stem_profile") != "four-stem":
                continue
            if model.get("model", {}).get("tensor_interface") != "waveform":
                continue
            if (model.get("deprecation") or {}).get("deprecated"):
                continue
            selected[model["variant"]] = model
        return selected

    def test_latest_adapter_never_selects_spectral_core_artifacts(self):
        """A legacy latest.json consumer cannot run a spectral-core graph;
        the adapter must skip them even when they are the smallest artifact
        of a variant (regression: generation 8)."""
        import generate_catalog_release as gen

        manifest = {
            "artifacts": {
                "models": [
                    {
                        "variant": "htdemucs",
                        "artifact_id": "waveform",
                        "download_url": "https://example.com/releases/download/model-v1/htdemucs.onnx",
                        "archive_digest": "a" * 64,
                        "byte_size": 300,
                        "deprecation": {"deprecated": False},
                        "model": {"stem_profile": "four-stem",
                                  "tensor_interface": "waveform"},
                    },
                    {
                        "variant": "htdemucs",
                        "artifact_id": "spectral",
                        "download_url": "https://example.com/releases/download/model-spectral-v1/htdemucs.spectral.onnx",
                        "archive_digest": "b" * 64,
                        "byte_size": 200,
                        "deprecation": {"deprecated": False},
                        "model": {"stem_profile": "four-stem",
                                  "tensor_interface": "spectral-core"},
                    },
                ]
            }
        }
        adapter = gen._build_latest_adapter(manifest)
        self.assertEqual(adapter["htdemucs"]["sha256"], "a" * 64)

    def test_latest_json_uses_stable_four_stem_digests(self):
        """latest.json must carry the digests of the stable generation's
        selectable four-stem artifacts (never a karaoke projection, never a
        deprecated artifact shadowing its replacement)."""
        adapter = _load(LATEST)
        selected = self._stable_four_stem_models()
        self.assertEqual(set(adapter), set(selected))
        for variant, model in selected.items():
            self.assertEqual(adapter[variant]["sha256"], model["archive_digest"])

    def test_manifest_and_latest_digests_agree(self):
        adapter = _load(LATEST)
        for variant, model in self._stable_four_stem_models().items():
            self.assertEqual(adapter[variant]["sha256"], model["archive_digest"])
            self.assertEqual(adapter[variant]["size"], model["byte_size"])

    def test_pointer_references_manifest_digest(self):
        import hashlib

        pointer = _load(POINTER)
        # Dynamically resolve the manifest the pointer actually references,
        # so the test works across generation advances.
        release_id = pointer["release_id"]
        manifest_path = ROOT_DIR / "catalog" / "releases" / f"{release_id}.json"
        manifest_bytes = manifest_path.read_bytes()
        self.assertEqual(
            pointer["release_manifest_sha256"], hashlib.sha256(manifest_bytes).hexdigest()
        )
        self.assertEqual(pointer["release_manifest_size"], len(manifest_bytes))


class DeterminismTests(unittest.TestCase):
    def test_regenerating_in_temp_dir_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            r1 = _run(
                ["--spec", str(SPEC), "--catalog-dir", str(tdp / "a"),
                 "--latest-json-path", str(tdp / "a" / "latest.json")]
            )
            r2 = _run(
                ["--spec", str(SPEC), "--catalog-dir", str(tdp / "b"),
                 "--latest-json-path", str(tdp / "b" / "latest.json")]
            )
            self.assertEqual(r1.returncode, 0, r1.stderr)
            self.assertEqual(r2.returncode, 0, r2.stderr)
            self.assertEqual(
                (tdp / "a" / "releases" / "2026-07-20-001.json").read_bytes(),
                (tdp / "b" / "releases" / "2026-07-20-001.json").read_bytes(),
            )
            self.assertEqual(
                (tdp / "a" / "latest.json").read_bytes(),
                (tdp / "b" / "latest.json").read_bytes(),
            )


class GuardTests(unittest.TestCase):
    def test_non_monotonic_generation_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            # Seed an existing release at generation 5.
            seed = _load(SPEC)
            seed["release_id"] = "2026-01-01-001"
            seed["generation"] = 5
            seed_path = tdp / "seed.spec.json"
            with seed_path.open("w") as fh:
                json.dump(seed, fh)
            r_seed = _run(
                ["--spec", str(seed_path), "--catalog-dir", str(tdp / "cat"),
                 "--latest-json-path", str(tdp / "seed-latest.json")]
            )
            self.assertEqual(r_seed.returncode, 0, r_seed.stderr)
            # Now try to generate generation 1 against that catalog dir.
            r_new = _run(
                ["--spec", str(SPEC), "--catalog-dir", str(tdp / "cat"),
                 "--latest-json-path", str(tdp / "new-latest.json")]
            )
            self.assertNotEqual(r_new.returncode, 0, r_new.stderr or r_new.stdout)
            self.assertIn("generation", (r_new.stderr + r_new.stdout).lower())

    def test_invalid_spec_does_not_write_pointer(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            bad = _load(SPEC)
            bad["artifacts"]["models"][0]["model"]["compatible_runtime_ids"] = ["nope"]
            bad_path = tdp / "bad.spec.json"
            with bad_path.open("w") as fh:
                json.dump(bad, fh)
            r = _run(
                ["--spec", str(bad_path), "--catalog-dir", str(tdp / "cat"),
                 "--latest-json-path", str(tdp / "latest.json")]
            )
            self.assertNotEqual(r.returncode, 0, r.stderr or r.stdout)
            # No pointer and no manifest should be left behind on failure.
            self.assertFalse((tdp / "cat" / "channels" / "stable.json").is_file())
            self.assertFalse((tdp / "cat" / "releases").is_dir() and any((tdp / "cat" / "releases").iterdir()))

    def test_mutating_existing_release_id_rejected(self):
        """Regenerating an existing release_id with different bytes must fail;
        --force must not bypass immutability for changed content."""
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            # First, generate the canonical release into a temp catalog.
            r1 = _run(
                ["--spec", str(SPEC), "--catalog-dir", str(tdp / "cat"),
                 "--latest-json-path", str(tdp / "latest.json")]
            )
            self.assertEqual(r1.returncode, 0, r1.stderr)
            original = (tdp / "cat" / "releases" / "2026-07-20-001.json").read_bytes()
            # Now build a spec that produces different bytes for the same id.
            mutated = _load(SPEC)
            mutated["notes"] = "mutated notes that change the manifest bytes"
            mutated_path = tdp / "mutated.spec.json"
            with mutated_path.open("w") as fh:
                json.dump(mutated, fh)
            # Without --force: must fail.
            r2 = _run(
                ["--spec", str(mutated_path), "--catalog-dir", str(tdp / "cat"),
                 "--latest-json-path", str(tdp / "latest.json")]
            )
            self.assertNotEqual(r2.returncode, 0, r2.stderr or r2.stdout)
            self.assertIn("immutable_release_id", r2.stderr)
            # The committed manifest must be untouched.
            self.assertEqual(
                (tdp / "cat" / "releases" / "2026-07-20-001.json").read_bytes(),
                original,
            )
            # With --force: still rejected because bytes differ.
            r3 = _run(
                ["--spec", str(mutated_path), "--catalog-dir", str(tdp / "cat"),
                 "--latest-json-path", str(tdp / "latest.json"), "--force"]
            )
            self.assertNotEqual(r3.returncode, 0, r3.stderr or r3.stdout)
            self.assertIn("immutable_release_id", r3.stderr)

    def test_idempotent_regeneration_allowed(self):
        """Regenerating the same spec into a catalog that already contains the
        identical release must succeed (idempotent)."""
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            r1 = _run(
                ["--spec", str(SPEC), "--catalog-dir", str(tdp / "cat"),
                 "--latest-json-path", str(tdp / "latest.json")]
            )
            self.assertEqual(r1.returncode, 0, r1.stderr)
            before = (tdp / "cat" / "releases" / "2026-07-20-001.json").read_bytes()
            # Regenerate the same spec into the same catalog dir.
            r2 = _run(
                ["--spec", str(SPEC), "--catalog-dir", str(tdp / "cat"),
                 "--latest-json-path", str(tdp / "latest.json")]
            )
            self.assertEqual(r2.returncode, 0, r2.stderr)
            after = (tdp / "cat" / "releases" / "2026-07-20-001.json").read_bytes()
            self.assertEqual(before, after)


class FreshnessTests(unittest.TestCase):
    def test_committed_catalog_matches_generator(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            r = _run(
                ["--spec", str(SPEC), "--catalog-dir", str(tdp / "cat"),
                 "--latest-json-path", str(tdp / "latest.json")]
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(
                (tdp / "cat" / "releases" / "2026-07-20-001.json").read_bytes(),
                RELEASE.read_bytes(),
            )
            # The adapter regenerated from THIS spec must be self-consistent
            # with THIS spec's manifest. (The committed latest.json tracks the
            # newest generation — CI's regenerate-all-specs loop covers its
            # freshness — so it is not compared against this older spec.)
            tmp_manifest = _load(tdp / "cat" / "releases" / "2026-07-20-001.json")
            tmp_adapter = _load(tdp / "latest.json")
            for model in tmp_manifest["artifacts"]["models"]:
                if model.get("model", {}).get("stem_profile") != "four-stem":
                    continue
                if (model.get("deprecation") or {}).get("deprecated"):
                    continue
                self.assertEqual(
                    tmp_adapter[model["variant"]]["sha256"], model["archive_digest"]
                )

    def test_generator_does_not_advance_stable_pointer(self):
        """The generator must NOT write catalog/channels/stable.json. The stable
        pointer is advanced only by publish_catalog_release.py after every
        referenced asset is uploaded and SHA-256-verified (issue #18 PR 4
        atomicity contract)."""
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            r = _run(
                ["--spec", str(SPEC), "--catalog-dir", str(tdp / "cat"),
                 "--latest-json-path", str(tdp / "latest.json")]
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertFalse(
                (tdp / "cat" / "channels" / "stable.json").is_file(),
                "generate_catalog_release.py must not advance the stable pointer; "
                "use publish_catalog_release.py --execute instead.",
            )


if __name__ == "__main__":
    unittest.main()
