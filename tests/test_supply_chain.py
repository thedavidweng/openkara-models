"""Tests for supply-chain record generation and verification (issue #18 PR 3)."""

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
        """Regenerating supply-chain records must produce byte-identical output."""
        import hashlib

        originals = {}
        for fname in ("sbom.spdx.json", "LICENSE", "NOTICE", "provenance.json"):
            p = SC_DIR / fname
            if p.is_file():
                originals[fname] = hashlib.sha256(p.read_bytes()).hexdigest()

        r = _run(["--release", RELEASE_ID])
        self.assertEqual(r.returncode, 0, r.stderr)

        for fname, original_sha in originals.items():
            new_sha = hashlib.sha256((SC_DIR / fname).read_bytes()).hexdigest()
            self.assertEqual(original_sha, new_sha, f"{fname} changed after regeneration")


if __name__ == "__main__":
    unittest.main()
