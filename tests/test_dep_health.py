"""Tests for the dependency automation health monitor."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

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


def test_check_workflow_recency_current() -> None:
    """Workflow ran recently — status should be 'current'."""
    import check_dep_automation_health as h
    recent = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    mock_response = {"workflow_runs": [{"created_at": recent}]}
    with patch.object(h, "_github_get", return_value=mock_response):
        result = h.check_workflow_recency("fake-token", max_days=14)
    assert result["status"] == "current"
    assert result["age_days"] <= 2


def test_check_workflow_recency_stale() -> None:
    """Workflow has not run recently — status should be 'stale'."""
    import check_dep_automation_health as h
    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    mock_response = {"workflow_runs": [{"created_at": old}]}
    with patch.object(h, "_github_get", return_value=mock_response):
        result = h.check_workflow_recency("fake-token", max_days=14)
    assert result["status"] == "stale"
    assert result["age_days"] > 14


def test_check_workflow_recency_no_runs() -> None:
    """No workflow runs found — status should be 'no_runs'."""
    import check_dep_automation_health as h
    with patch.object(h, "_github_get", return_value={"workflow_runs": []}):
        result = h.check_workflow_recency("fake-token", max_days=14)
    assert result["status"] == "no_runs"


def test_check_workflow_recency_api_error() -> None:
    """API failure during workflow recency check — status should be 'api_error'."""
    import check_dep_automation_health as h
    with patch.object(h, "_github_get", side_effect=RuntimeError("API down")):
        result = h.check_workflow_recency("fake-token", max_days=14)
    assert result["status"] == "api_error"


def test_check_duplicate_candidates_no_token() -> None:
    import check_dep_automation_health as h
    result = h.check_duplicate_candidates(None)
    assert result["status"] == "skipped"


def test_check_duplicate_candidates_clean() -> None:
    """No duplicate candidate branches — status should be 'clean'."""
    import check_dep_automation_health as h
    mock_branches = [
        {"name": "main"},
        {"name": "deps/candidate-ort-v1.28.0"},
        {"name": "deps/candidate-htdemucs-abc123"},
    ]
    with patch.object(h, "_github_get", return_value=mock_branches):
        result = h.check_duplicate_candidates("fake-token")
    assert result["status"] == "clean"
    assert len(result["candidates"]) == 2


def test_check_duplicate_candidates_duplicates() -> None:
    """Duplicate candidate branches for the same version — status should be 'duplicates'."""
    import check_dep_automation_health as h
    mock_branches = [
        {"name": "main"},
        {"name": "deps/candidate-ort-v1.28.0"},
        {"name": "deps/candidate-ort-v1.28.0-old"},
    ]
    with patch.object(h, "_github_get", return_value=mock_branches):
        result = h.check_duplicate_candidates("fake-token")
    assert result["status"] == "duplicates"
    assert "ort-v1.28.0" in result["duplicates"]


def test_check_duplicate_candidates_api_error() -> None:
    """API failure during duplicate check — status should be 'api_error'."""
    import check_dep_automation_health as h
    with patch.object(h, "_github_get", side_effect=RuntimeError("API down")):
        result = h.check_duplicate_candidates("fake-token")
    assert result["status"] == "api_error"


def test_check_duplicate_candidates_permission_failure() -> None:
    """Bot permission failure (403) — status should be 'api_error'."""
    import check_dep_automation_health as h
    with patch.object(h, "_github_get", side_effect=RuntimeError("GitHub API error 403: Forbidden")):
        result = h.check_duplicate_candidates("fake-token")
    assert result["status"] == "api_error"
    assert "403" in result["error"]


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


def test_main_no_runs_is_failure() -> None:
    """Regression: when the detection workflow has never run (no_runs), the
    health check must NOT report 'automation healthy'. Previously no_runs
    fell through all failure checks and returned 0 with 'OK'."""
    import check_dep_automation_health as h
    from unittest.mock import patch

    # Mock all three checks: ort current, workflow no_runs, duplicates clean.
    with patch.object(h, "check_ort_lock_stale",
                      return_value={"status": "current", "lock_tag": "v1.28.0",
                                    "latest_tag": "v1.28.0",
                                    "update_available": False}), \
         patch.object(h, "check_workflow_recency",
                      return_value={"status": "no_runs",
                                    "reason": "no workflow runs found"}), \
         patch.object(h, "check_duplicate_candidates",
                      return_value={"status": "clean", "candidates": [],
                                    "duplicates": {}}):
        # main() calls argparse and reads GITHUB_TOKEN; patch sys.argv and env.
        import sys as _sys
        old_argv = _sys.argv
        _sys.argv = ["check_dep_automation_health.py"]
        try:
            exit_code = h.main()
        finally:
            _sys.argv = old_argv
    assert exit_code != 0, "no_runs must not be reported as healthy (exit 0)"
    assert exit_code == 2, f"no_runs should be automation failure (exit 2), got {exit_code}"


def test_main_no_runs_not_healthy_message() -> None:
    """The non-JSON output must not say 'OK: automation healthy' when
    no_runs is present."""
    import check_dep_automation_health as h
    from unittest.mock import patch
    import io

    mock_stdout = io.StringIO()
    mock_stderr = io.StringIO()
    with patch.object(h, "check_ort_lock_stale",
                      return_value={"status": "current", "lock_tag": "v1.28.0",
                                    "latest_tag": "v1.28.0",
                                    "update_available": False}), \
         patch.object(h, "check_workflow_recency",
                      return_value={"status": "no_runs",
                                    "reason": "no workflow runs found"}), \
         patch.object(h, "check_duplicate_candidates",
                      return_value={"status": "clean", "candidates": [],
                                    "duplicates": {}}), \
         patch("sys.stdout", mock_stdout), \
         patch("sys.stderr", mock_stderr):
        import sys as _sys
        old_argv = _sys.argv
        _sys.argv = ["check_dep_automation_health.py"]
        try:
            exit_code = h.main()
        finally:
            _sys.argv = old_argv
    assert exit_code == 2
    assert "healthy" not in mock_stdout.getvalue().lower()
    assert "NO_RUNS" in mock_stderr.getvalue()
