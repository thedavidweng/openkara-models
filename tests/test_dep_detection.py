"""Tests for the dependency detection scripts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def test_is_prerelease_filters_rc() -> None:
    import detect_ort_release as d
    assert d._is_prerelease("v1.28.0-rc1") is True
    assert d._is_prerelease("v1.28.0-alpha") is True
    assert d._is_prerelease("v1.28.0-beta1") is True
    assert d._is_prerelease("v1.28.0-preview") is True
    assert d._is_prerelease("v1.28.0-dev") is True
    assert d._is_prerelease("v1.28.0-pre") is True
    assert d._is_prerelease("v1.28.0-pre.1") is True
    assert d._is_prerelease("v2.0.0-alpha.2") is True


def test_is_prerelease_no_false_positives() -> None:
    """Tags containing prerelease markers as substrings of other words."""
    import detect_ort_release as d
    assert d._is_prerelease("v1.28.0-architecture") is False
    assert d._is_prerelease("v2.0.0-march") is False
    assert d._is_prerelease("v2.0.0-mrc") is False
    assert d._is_prerelease("v1.0.0-arch") is False


def test_is_prerelease_passes_stable() -> None:
    import detect_ort_release as d
    assert d._is_prerelease("v1.27.1") is False
    assert d._is_prerelease("v1.28.0") is False


def test_compare_with_lock_no_update() -> None:
    import detect_ort_release as d
    latest = {"tag": "v1.27.1", "commit_sha": "abc123"}
    lock = {"upstream": {"tag": "v1.27.1", "commit_sha": "abc123"}}
    c = d.compare_with_lock(latest, lock)
    assert c["update_available"] is False


def test_compare_with_lock_tag_update() -> None:
    import detect_ort_release as d
    latest = {"tag": "v1.28.0", "commit_sha": "def456"}
    lock = {"upstream": {"tag": "v1.27.1", "commit_sha": "abc123"}}
    c = d.compare_with_lock(latest, lock)
    assert c["update_available"] is True
    assert c["latest_tag"] == "v1.28.0"


def test_compare_with_lock_commit_update() -> None:
    """Same tag but different commit SHA (force-push or rebase)."""
    import detect_ort_release as d
    latest = {"tag": "v1.27.1", "commit_sha": "new789"}
    lock = {"upstream": {"tag": "v1.27.1", "commit_sha": "abc123"}}
    c = d.compare_with_lock(latest, lock)
    assert c["update_available"] is True


def test_detect_ort_release_help() -> None:
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "detect_ort_release.py"), "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert "--json" in r.stdout


def test_detect_model_weight_revision_help() -> None:
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "detect_model_weight_revision.py"), "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert "--model" in r.stdout


def test_detect_model_weight_revision_unknown_model() -> None:
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "detect_model_weight_revision.py"),
         "--model", "nonexistent"],
        capture_output=True, text=True,
    )
    # argparse choices should reject it.
    assert r.returncode != 0


def test_hf_models_mapping() -> None:
    import detect_model_weight_revision as d
    assert "htdemucs" in d.HF_MODELS
    assert "htdemucs_ft" in d.HF_MODELS
    assert d.HF_MODELS["htdemucs"] == "adefossez/HTDemucs"
    assert d.HF_MODELS["htdemucs_ft"] == "adefossez/HTDemucs-ft"
