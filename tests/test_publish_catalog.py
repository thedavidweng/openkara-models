"""Tests for the atomic publication script (issue #18 PR 4).

The publication script verifies every referenced asset URL resolves to an
existing GitHub release asset whose SHA-256 matches the manifest. These tests
mock the ``gh`` subprocess calls so they run without network or auth.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import publish_catalog_release as pub  # noqa: E402

RELEASE_ID = "2026-07-20-001"
MANIFEST = ROOT_DIR / "catalog" / "releases" / f"{RELEASE_ID}.json"


def _load_manifest() -> dict:
    with MANIFEST.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _make_gh(
    asset_responses: dict[tuple[str, str], list[dict]],
    download_shas: dict[tuple[str, str, str], str] | None = None,
):
    """Build a mock for publish_catalog_release._gh.

    ``asset_responses`` maps (repo, tag) -> list of asset dicts (the JSON
    returned by ``gh release view --json assets``).
    ``download_shas`` maps (repo, tag, asset_name) -> sha256 hex (the result
    of downloading and hashing the asset).
    """
    download_shas = download_shas or {}

    def _fake_gh(args, *, check=True, capture=True):
        if args[0] == "release" and args[1] == "view":
            tag = args[2]
            repo = None
            for i, a in enumerate(args):
                if a == "--repo" and i + 1 < len(args):
                    repo = args[i + 1]
            key = (repo or "", tag)
            assets = asset_responses.get(key)
            if assets is None:
                return subprocess.CompletedProcess(args, 1, "", f"release {tag} not found")
            return subprocess.CompletedProcess(args, 0, json.dumps({"assets": assets}), "")
        if args[0] == "release" and args[1] == "download":
            tag = args[2]
            repo = pattern = dir_ = None
            for i, a in enumerate(args):
                if a == "--repo" and i + 1 < len(args):
                    repo = args[i + 1]
                if a == "--pattern" and i + 1 < len(args):
                    pattern = args[i + 1]
                if a == "--dir" and i + 1 < len(args):
                    dir_ = args[i + 1]
            sha = download_shas.get((repo or "", tag, pattern or ""))
            if sha is None:
                return subprocess.CompletedProcess(args, 1, "", "download failed")
            # Write a dummy file so _sha256_file can hash something. We patch
            # _sha256_file separately to return the expected digest.
            dest = Path(dir_) / (pattern or "")
            dest.write_bytes(b"x" * 16)
            return subprocess.CompletedProcess(args, 0, "", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    return _fake_gh


def _asset_response_for_manifest(manifest: dict) -> dict[tuple[str, str], list[dict]]:
    responses: dict[tuple[str, str], list[dict]] = {}
    for kind in ("models", "runtimes", "bundles"):
        for art in manifest.get("artifacts", {}).get(kind, []):
            url = art["download_url"]
            parsed = pub._parse_release_url(url)
            assert parsed is not None, f"unparseable URL: {url}"
            repo, tag = parsed
            fname = url.rsplit("/", 1)[-1]
            responses.setdefault((repo, tag), []).append({
                "name": fname,
                "size": art["byte_size"],
            })
    return responses


def _download_shas_for_manifest(manifest: dict) -> dict[tuple[str, str, str], str]:
    shas: dict[tuple[str, str, str], str] = {}
    for kind in ("models", "runtimes", "bundles"):
        for art in manifest.get("artifacts", {}).get(kind, []):
            url = art["download_url"]
            parsed = pub._parse_release_url(url)
            assert parsed is not None
            repo, tag = parsed
            fname = url.rsplit("/", 1)[-1]
            shas[(repo, tag, fname)] = art["archive_digest"]
    return shas


def _run_main_in_process(argv: list[str], gh_mock, sha256_mock=None):
    """Run pub.main() in-process with patched argv, _gh, and optionally _sha256_file."""
    sha256_mock = sha256_mock or (lambda _path: "0" * 64)
    with mock.patch.object(sys, "argv", ["publish_catalog_release.py", *argv]):
        with mock.patch.object(pub, "_gh", gh_mock):
            with mock.patch.object(pub, "_sha256_file", sha256_mock):
                out = io.StringIO()
                err = io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    code = pub.main()
    return code, out.getvalue(), err.getvalue()


class DryRunTests(unittest.TestCase):
    def test_dry_run_passes_for_committed_release_with_matching_assets(self):
        manifest = _load_manifest()
        assets = _asset_response_for_manifest(manifest)
        shas = _download_shas_for_manifest(manifest)

        def sha256_for_path(path: Path) -> str:
            # Supply-chain files are hashed by the real function; artifacts are
            # hashed by _download_asset_sha256 which calls _sha256_file on the
            # downloaded temp file. We return the manifest's expected digest
            # for artifact files and the real digest for supply-chain files.
            return shas.get(("", "", path.name), "") or pub._sha256_file(path)

        # Build a sha256 mock that returns the right digest per (repo, tag, name).
        # _download_asset_sha256 downloads to a temp dir then calls _sha256_file.
        # We patch _download_asset_sha256 to return the expected sha directly.
        download_shas = shas

        def fake_download_sha256(repo: str, tag: str, asset_name: str) -> str | None:
            return download_shas.get((repo, tag, asset_name))

        with mock.patch.object(sys, "argv", ["publish_catalog_release.py", "--release", RELEASE_ID]):
            with mock.patch.object(pub, "_gh", _make_gh(assets, shas)):
                with mock.patch.object(pub, "_download_asset_sha256", fake_download_sha256):
                    out = io.StringIO()
                    err = io.StringIO()
                    with redirect_stdout(out), redirect_stderr(err):
                        code = pub.main()
        self.assertEqual(code, 0, err.getvalue())
        self.assertIn("DRY-RUN", out.getvalue())
        self.assertIn("ready to publish", out.getvalue())

    def test_dry_run_fails_for_nonexistent_release(self):
        with mock.patch.object(sys, "argv", ["publish_catalog_release.py", "--release", "9999-99-99-999"]):
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = pub.main()
        self.assertNotEqual(code, 0)
        self.assertIn("manifest not found", err.getvalue())

    def test_dry_run_verifies_supply_chain(self):
        """Tampering with a supply-chain file must fail the dry-run."""
        sc_file = ROOT_DIR / "catalog" / "supply-chain" / RELEASE_ID / "NOTICE"
        original = sc_file.read_bytes()
        try:
            sc_file.write_bytes(b"TAMPERED")
            manifest = _load_manifest()
            assets = _asset_response_for_manifest(manifest)
            shas = _download_shas_for_manifest(manifest)

            def fake_download_sha256(repo, tag, asset_name):
                return shas.get((repo, tag, asset_name))

            with mock.patch.object(sys, "argv", ["publish_catalog_release.py", "--release", RELEASE_ID]):
                with mock.patch.object(pub, "_gh", _make_gh(assets, shas)):
                    with mock.patch.object(pub, "_download_asset_sha256", fake_download_sha256):
                        out = io.StringIO()
                        err = io.StringIO()
                        with redirect_stdout(out), redirect_stderr(err):
                            code = pub.main()
            self.assertNotEqual(code, 0)
            self.assertIn("supply_chain.notice", err.getvalue())
        finally:
            sc_file.write_bytes(original)

    def test_dry_run_fails_on_sha256_mismatch(self):
        """If a GitHub asset's SHA-256 does not match the manifest, dry-run fails."""
        manifest = _load_manifest()
        assets = _asset_response_for_manifest(manifest)
        wrong_shas = {k: "0" * 64 for k in _download_shas_for_manifest(manifest)}

        def fake_download_sha256(repo, tag, asset_name):
            return wrong_shas.get((repo, tag, asset_name))

        with mock.patch.object(sys, "argv", ["publish_catalog_release.py", "--release", RELEASE_ID]):
            with mock.patch.object(pub, "_gh", _make_gh(assets, wrong_shas)):
                with mock.patch.object(pub, "_download_asset_sha256", fake_download_sha256):
                    out = io.StringIO()
                    err = io.StringIO()
                    with redirect_stdout(out), redirect_stderr(err):
                        code = pub.main()
        self.assertNotEqual(code, 0)
        self.assertIn("sha256 mismatch", err.getvalue())

    def test_dry_run_fails_when_asset_not_found(self):
        """If a referenced asset is missing from the GitHub release, dry-run fails."""
        manifest = _load_manifest()
        assets: dict[tuple[str, str], list[dict]] = {}
        for kind in ("models", "runtimes", "bundles"):
            for art in manifest.get("artifacts", {}).get(kind, []):
                url = art["download_url"]
                parsed = pub._parse_release_url(url)
                assert parsed is not None
                repo, tag = parsed
                assets.setdefault((repo, tag), [])

        def fake_download_sha256(repo, tag, asset_name):
            return None

        with mock.patch.object(sys, "argv", ["publish_catalog_release.py", "--release", RELEASE_ID]):
            with mock.patch.object(pub, "_gh", _make_gh(assets, {})):
                with mock.patch.object(pub, "_download_asset_sha256", fake_download_sha256):
                    out = io.StringIO()
                    err = io.StringIO()
                    with redirect_stdout(out), redirect_stderr(err):
                        code = pub.main()
        self.assertNotEqual(code, 0)
        self.assertIn("not found in release", err.getvalue())


class MonotonicityGuardTests(unittest.TestCase):
    def test_dry_run_checks_monotonicity(self):
        """The dry-run must pass monotonicity checks against existing releases."""
        manifest = _load_manifest()
        assets = _asset_response_for_manifest(manifest)
        shas = _download_shas_for_manifest(manifest)

        def fake_download_sha256(repo, tag, asset_name):
            return shas.get((repo, tag, asset_name))

        with mock.patch.object(sys, "argv", ["publish_catalog_release.py", "--release", RELEASE_ID]):
            with mock.patch.object(pub, "_gh", _make_gh(assets, shas)):
                with mock.patch.object(pub, "_download_asset_sha256", fake_download_sha256):
                    out = io.StringIO()
                    err = io.StringIO()
                    with redirect_stdout(out), redirect_stderr(err):
                        code = pub.main()
        self.assertEqual(code, 0, err.getvalue())

    def test_dry_run_accepts_candidate_channel(self):
        manifest = _load_manifest()
        assets = _asset_response_for_manifest(manifest)
        shas = _download_shas_for_manifest(manifest)

        def fake_download_sha256(repo, tag, asset_name):
            return shas.get((repo, tag, asset_name))

        argv = ["publish_catalog_release.py", "--release", RELEASE_ID,
                "--channel", "candidate"]
        with mock.patch.object(sys, "argv", argv):
            with mock.patch.object(pub, "_gh", _make_gh(assets, shas)):
                with mock.patch.object(pub, "_download_asset_sha256", fake_download_sha256):
                    out = io.StringIO()
                    err = io.StringIO()
                    with redirect_stdout(out), redirect_stderr(err):
                        code = pub.main()
        self.assertEqual(code, 0, err.getvalue())
        self.assertIn("candidate", out.getvalue())


class ChannelPointerTests(unittest.TestCase):
    """_advance_channel_pointer writes the selected channel file only."""

    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.channels = Path(self._tmp.name)
        self._patcher = mock.patch.object(pub, "CHANNELS_DIR", self.channels)
        self._patcher.start()
        self.addCleanup(self._patcher.stop)
        self.addCleanup(self._tmp.cleanup)

        self.manifest = _load_manifest()
        self.manifest_path = MANIFEST

    def _stable_pointer(self, release_id: str) -> None:
        (self.channels / "stable.json").write_text(json.dumps({
            "schema_version": "openkara.catalog/channel-v1",
            "channel": "stable",
            "generation": 1,
            "release_id": release_id,
            "release_manifest_url": "https://example.com/releases/download/infra-x/release-manifest.json",
            "release_manifest_sha256": "0" * 64,
            "release_manifest_size": 1,
            "updated_at": "2026-07-20T00:00:00Z",
            "previous_release_id": None,
        }))

    def test_candidate_pointer_never_touches_stable(self):
        self._stable_pointer("2026-01-01-001")
        stable_before = (self.channels / "stable.json").read_bytes()

        errors = pub._advance_channel_pointer(
            self.manifest, self.manifest_path, "candidate"
        )
        self.assertEqual(errors, [])

        candidate = json.loads((self.channels / "candidate.json").read_text())
        self.assertEqual(candidate["channel"], "candidate")
        self.assertEqual(candidate["release_id"], self.manifest["release_id"])
        # First candidate pointer inherits its lineage from stable.
        self.assertEqual(candidate["previous_release_id"], "2026-01-01-001")
        self.assertEqual(
            (self.channels / "stable.json").read_bytes(),
            stable_before,
            "candidate publication must not modify the stable pointer",
        )

    def test_stable_pointer_still_written_to_stable_file(self):
        self._stable_pointer("2026-01-01-001")
        errors = pub._advance_channel_pointer(
            self.manifest, self.manifest_path, "stable"
        )
        self.assertEqual(errors, [])
        stable = json.loads((self.channels / "stable.json").read_text())
        self.assertEqual(stable["channel"], "stable")
        self.assertEqual(stable["release_id"], self.manifest["release_id"])
        self.assertEqual(stable["previous_release_id"], "2026-01-01-001")
        self.assertFalse((self.channels / "candidate.json").exists())

    def test_candidate_pointer_tracks_previous_candidate(self):
        self._stable_pointer("2026-01-01-001")
        errors = pub._advance_channel_pointer(
            self.manifest, self.manifest_path, "candidate"
        )
        self.assertEqual(errors, [])
        # Re-advance: previous_release_id now comes from the candidate file.
        # (Same release id keeps previous unchanged; a differing id would
        # replace it — emulate by rewriting the candidate's release_id.)
        candidate_path = self.channels / "candidate.json"
        first = json.loads(candidate_path.read_text())
        first["release_id"] = "2026-01-02-001"
        candidate_path.write_text(json.dumps(first))

        errors = pub._advance_channel_pointer(
            self.manifest, self.manifest_path, "candidate"
        )
        self.assertEqual(errors, [])
        second = json.loads(candidate_path.read_text())
        self.assertEqual(second["previous_release_id"], "2026-01-02-001")


if __name__ == "__main__":
    unittest.main()
