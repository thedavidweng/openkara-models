"""Tests for the atomic publication script (issue #18 PR 4)."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

RELEASE_ID = "2026-07-20-001"
MANIFEST = ROOT_DIR / "catalog" / "releases" / f"{RELEASE_ID}.json"


def _run_publish(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "publish_catalog_release.py"), *args],
        capture_output=True,
        text=True,
        cwd=str(ROOT_DIR),
    )


class DryRunTests(unittest.TestCase):
    def test_dry_run_passes_for_committed_release(self):
        r = _run_publish(["--release", RELEASE_ID, "--skip-asset-check"])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("DRY-RUN", r.stdout)
        self.assertIn("ready to publish", r.stdout)

    def test_dry_run_fails_for_nonexistent_release(self):
        r = _run_publish(["--release", "9999-99-99-999", "--skip-asset-check"])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("manifest not found", r.stderr)

    def test_dry_run_verifies_supply_chain(self):
        """Tampering with a supply-chain file must fail the dry-run."""
        sc_file = ROOT_DIR / "catalog" / "supply-chain" / RELEASE_ID / "NOTICE"
        original = sc_file.read_bytes()
        try:
            sc_file.write_bytes(b"TAMPERED")
            r = _run_publish(["--release", RELEASE_ID, "--skip-asset-check"])
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("supply_chain.notice", r.stderr)
        finally:
            sc_file.write_bytes(original)


class MonotonicityGuardTests(unittest.TestCase):
    def test_dry_run_checks_monotonicity(self):
        """The dry-run must pass monotonicity checks against existing releases."""
        r = _run_publish(["--release", RELEASE_ID, "--skip-asset-check"])
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
