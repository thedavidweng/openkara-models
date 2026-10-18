"""Tests for supply-chain record generation and verification (issue #18 PR 3)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import generate_supply_chain as generator
from generate_catalog_release import _build_manifest

RELEASE_ID = "2026-07-20-001"
SPEC = ROOT_DIR / "catalog" / "specs" / f"{RELEASE_ID}.spec.json"
RELEASE = ROOT_DIR / "catalog" / "releases" / f"{RELEASE_ID}.json"
SC_DIR = ROOT_DIR / "catalog" / "supply-chain" / RELEASE_ID


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _run(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "generate_supply_chain.py"), *args],
        capture_output=True,
        text=True,
        cwd=str(ROOT_DIR),
        **kw,
    )


def _run_verify(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "verify_supply_chain.py"), *args],
        capture_output=True,
        text=True,
        cwd=str(ROOT_DIR),
        **kw,
    )


class SupplyChainFilesTests(unittest.TestCase):
    def test_sbom_exists_and_is_valid_spdx(self):
        sbom = _load(SC_DIR / "sbom.spdx.json")
        self.assertEqual(sbom["spdxVersion"], "SPDX-2.3")
        self.assertEqual(sbom["dataLicense"], "CC0-1.0")
        self.assertIn("packages", sbom)
        # Repo package + 2 model packages = 3
        self.assertEqual(len(sbom["packages"]), 3)

    def test_license_copy_matches_repo_license(self):
        import hashlib

        self.assertEqual(
            hashlib.sha256((SC_DIR / "LICENSE").read_bytes()).hexdigest(),
            hashlib.sha256((ROOT_DIR / "LICENSE").read_bytes()).hexdigest(),
        )

    def test_notice_contains_release_id_and_producer(self):
        notice = (SC_DIR / "NOTICE").read_text()
        self.assertIn(RELEASE_ID, notice)
        self.assertIn("MIT", notice)
        self.assertIn("Demucs", notice)

    def test_provenance_contains_commit_and_subjects(self):
        prov = _load(SC_DIR / "provenance.json")
        self.assertEqual(prov["release_id"], RELEASE_ID)
        self.assertIn("commit_sha", prov["producer"])
        self.assertEqual(len(prov["subject"]), 2)  # 2 model artifacts
        for s in prov["subject"]:
            self.assertTrue(s["sha256"])
            self.assertIsInstance(s["size"], int)


class SupplyChainRefsInManifestTests(unittest.TestCase):
    def test_manifest_has_release_level_supply_chain(self):
        manifest = _load(RELEASE)
        sc = manifest.get("supply_chain")
        self.assertIsNotNone(sc)
        for key in ("sbom", "license", "notice", "provenance"):
            self.assertIn(key, sc)
            self.assertEqual(sc[key]["url"][:8], "https://")
            self.assertEqual(sc[key]["digest_algorithm"], "sha256")

    def test_manifest_artifacts_have_supply_chain(self):
        manifest = _load(RELEASE)
        for model in manifest["artifacts"]["models"]:
            self.assertIn("supply_chain", model)
            for key in ("sbom", "license", "notice", "provenance"):
                self.assertIn(key, model["supply_chain"])

    def test_supply_chain_digests_match_local_files(self):
        import hashlib

        manifest = _load(RELEASE)
        sc = manifest["supply_chain"]
        for key, fname in [("sbom", "sbom.spdx.json"), ("license", "LICENSE"),
                           ("notice", "NOTICE"), ("provenance", "provenance.json")]:
            local = SC_DIR / fname
            self.assertTrue(local.is_file(), f"{fname} missing")
            self.assertEqual(sc[key]["size"], local.stat().st_size)
            self.assertEqual(
                sc[key]["sha256"],
                hashlib.sha256(local.read_bytes()).hexdigest(),
            )


class VerifySupplyChainTests(unittest.TestCase):
    def test_verify_all_passes(self):
        r = _run_verify(["--all"])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("0 errors", r.stdout)

    def test_verify_single_manifest_passes(self):
        r = _run_verify([str(RELEASE)])
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_verify_detects_tampered_file(self):
        notice = SC_DIR / "NOTICE"
        original = notice.read_bytes()
        try:
            notice.write_bytes(b"TAMPERED")
            r = _run_verify([str(RELEASE)])
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("sha256 mismatch", r.stderr)
        finally:
            notice.write_bytes(original)


class FreshnessTests(unittest.TestCase):
    def test_supply_chain_files_match_generator(self):
        """Every historical spec must reproduce its published bytes."""
        for spec_path in sorted(SPEC.parent.glob("*.spec.json")):
            spec = _load(spec_path)
            release_id = spec["release_id"]
            with self.subTest(release_id=release_id):
                sc_dir = SC_DIR.parent / release_id
                originals = {p: p.read_bytes() for p in sc_dir.iterdir() if p.is_file()}
                original_spec = spec_path.read_bytes()
                r = _run(["--release", release_id])
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual(spec_path.read_bytes(), original_spec)
                for path, original in originals.items():
                    self.assertEqual(path.read_bytes(), original, str(path))
                manifest = _build_manifest(_load(spec_path), spec_path.parent)
                manifest_bytes = (
                    json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
                ).encode("utf-8")
                self.assertEqual(
                    manifest_bytes,
                    (RELEASE.parent / f"{release_id}.json").read_bytes(),
                )


class GeneratorVersionTests(unittest.TestCase):
    def test_new_sbom_relationships_resolve(self):
        spec = _load(SPEC)
        spec.pop("supply_chain_generator_version", None)
        sbom = generator._build_sbom(spec, "2099-01-01-001")
        self.assertEqual(sbom["SPDXID"], "SPDXRef-DOCUMENT")
        ids = {sbom["SPDXID"], *(p["SPDXID"] for p in sbom["packages"])}
        for relationship in sbom["relationships"]:
            self.assertIn(relationship["spdxElementId"], ids)
            self.assertIn(relationship["relatedSpdxElement"], ids)

    def test_unknown_generator_version_rejected(self):
        spec = _load(SPEC)
        spec["supply_chain_generator_version"] = 3
        with self.assertRaisesRegex(ValueError, "supply_chain_generator_version"):
            generator._build_sbom(spec, "2099-01-01-001")

    def test_new_release_pins_current_version_and_regenerates_identically(self):
        import verify_supply_chain

        release_id = "2099-01-01-001"
        spec = _load(SPEC)
        spec.pop("supply_chain_generator_version", None)
        spec["release_id"] = release_id
        with tempfile.TemporaryDirectory() as td:
            specs_dir = Path(td) / "specs"
            specs_dir.mkdir()
            sc_dir = Path(td) / "supply-chain"
            spec_path = specs_dir / f"{release_id}.spec.json"
            spec_path.write_text(json.dumps(spec))
            with (
                patch.object(generator, "SPECS_DIR", specs_dir),
                patch.object(generator, "SC_DIR", sc_dir),
                patch.object(sys, "argv", ["generate_supply_chain.py", "--release", release_id]),
            ):
                self.assertEqual(generator.main(), 0)
                generated_spec = _load(spec_path)
                self.assertEqual(generated_spec["supply_chain_generator_version"], 2)
                sbom = _load(sc_dir / release_id / "sbom.spdx.json")
                self.assertEqual(sbom["SPDXID"], "SPDXRef-DOCUMENT")
                manifest = _build_manifest(generated_spec, specs_dir)
                self.assertNotIn("supply_chain_generator_version", manifest)
                manifest_path = Path(td) / f"{release_id}.json"
                manifest_path.write_text(json.dumps(manifest))
                with patch.object(verify_supply_chain, "SC_DIR", sc_dir):
                    self.assertEqual(verify_supply_chain.verify_manifest(manifest_path), [])
                originals = {p: p.read_bytes() for p in sc_dir.rglob("*") if p.is_file()}
                originals[spec_path] = spec_path.read_bytes()
                self.assertEqual(generator.main(), 0)
                for path, original in originals.items():
                    self.assertEqual(path.read_bytes(), original, str(path))


if __name__ == "__main__":
    unittest.main()
