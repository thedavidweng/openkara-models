"""Tests for the dependency automation health monitor."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def test_check_ort_lock_stale_current() -> None:
    """When the lock matches the latest stable, status is 'current'."""
    import check_dep_automation_health as h
    import detect_ort_release
    # Mock detect_latest_stable to return the lock's current values.
    lock = json.loads((ROOT / "ort" / "source-lock.json").read_text())
    original = detect_ort_release.detect_latest_stable
    detect_ort_release.detect_latest_stable = lambda token=None: {
        "tag": lock["upstream"]["tag"],
        "commit_sha": lock["upstream"]["commit_sha"],
        "release_url": "", "published_at": "", "name": "", "body": "",
    }
    try:
        result = h.check_ort_lock_stale(None)
        assert result["status"] == "current"
        assert result["update_available"] is False
    finally:
        detect_ort_release.detect_latest_stable = original


def test_check_ort_lock_stale_behind() -> None:
    """When the lock is behind the latest stable, status is 'stale'."""
    import check_dep_automation_health as h
    import detect_ort_release
    lock = json.loads((ROOT / "ort" / "source-lock.json").read_text())
    original = detect_ort_release.detect_latest_stable
    detect_ort_release.detect_latest_stable = lambda token=None: {
        "tag": "v99.99.99",
        "commit_sha": "new" * 13 + "1",
        "release_url": "", "published_at": "", "name": "", "body": "",
    }
    try:
        result = h.check_ort_lock_stale(None)
        assert result["status"] == "stale"
        assert result["update_available"] is True
    finally:
        detect_ort_release.detect_latest_stable = original


def test_check_ort_lock_stale_api_error() -> None:
    import check_dep_automation_health as h
    import detect_ort_release
    original = detect_ort_release.detect_latest_stable
    def raise_error(token=None):
        raise RuntimeError("API down")
    detect_ort_release.detect_latest_stable = raise_error
    try:
        result = h.check_ort_lock_stale(None)
        assert result["status"] == "api_error"
    finally:
        detect_ort_release.detect_latest_stable = original


def test_check_workflow_recency_no_token() -> None:
    import check_dep_automation_health as h
    result = h.check_workflow_recency(None, max_days=14)
    assert result["status"] == "skipped"


def test_check_duplicate_candidates_no_token() -> None:
    import check_dep_automation_health as h
    result = h.check_duplicate_candidates(None)
    assert result["status"] == "skipped"


def test_cli_help() -> None:
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "check_dep_automation_health.py"), "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert "--max-stale-days" in r.stdout


def test_cli_json() -> None:
    """The CLI should output JSON with --json (may exit non-zero if stale)."""
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "check_dep_automation_health.py"), "--json"],
        capture_output=True, text=True,
    )
    # Exit code may be 0 (current), 1 (stale), or 3 (API error without token).
    # The important thing is that JSON is produced.
    assert r.stdout.strip().startswith("{")
    data = json.loads(r.stdout)
    assert "ort_lock" in data
