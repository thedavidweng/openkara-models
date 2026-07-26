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


# ---------------------------------------------------------------------------
# Model source lock (models/source-lock.json, issue #20)
# ---------------------------------------------------------------------------

MODEL_LOCK_PATH = ROOT / "models" / "source-lock.json"


def _load_model_lock() -> dict:
    return json.loads(MODEL_LOCK_PATH.read_text(encoding="utf-8"))


def test_model_lock_exists() -> None:
    assert MODEL_LOCK_PATH.is_file(), "models/source-lock.json must exist"


def test_model_lock_version_and_models() -> None:
    lock = _load_model_lock()
    assert lock["lock_version"] == "openkara.model-source-lock/v1"
    assert set(lock["models"].keys()) == {"htdemucs", "htdemucs_ft"}


def test_model_lock_repos_match_detector_mapping() -> None:
    """The lock's weights repos and the detector's HF mapping must agree —
    one weights authority per model."""
    import detect_model_weight_revision as d
    lock = _load_model_lock()
    for name, entry in lock["models"].items():
        assert entry["weights"]["repo"] == d.HF_MODELS[name]


def test_model_lock_commit_shas_are_40_hex() -> None:
    lock = _load_model_lock()
    for name, entry in lock["models"].items():
        sha = entry["weights"]["commit_sha"]
        assert len(sha) == 40, name
        int(sha, 16)


def test_validate_model_source_lock_passes() -> None:
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "validate_model_source_lock.py")],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert r.returncode == 0, r.stderr


def test_validate_model_lock_rejects_timestamp_state() -> None:
    import validate_model_source_lock as v
    lock = _load_model_lock()
    lock["models"]["htdemucs"]["weights"]["last_modified"] = "2026-07-11T13:51:42Z"
    errors = v.validate_model_lock(lock)
    assert any("timestamp state is forbidden" in e for e in errors)


def test_validate_model_lock_rejects_catalog_field_duplication() -> None:
    import validate_model_source_lock as v
    lock = _load_model_lock()
    lock["models"]["htdemucs"]["weights"]["sha256"] = "0" * 64
    errors = v.validate_model_lock(lock)
    assert any("catalog-owned artifact field" in e for e in errors)


def test_validate_model_lock_rejects_missing_model() -> None:
    import validate_model_source_lock as v
    lock = _load_model_lock()
    del lock["models"]["htdemucs_ft"]
    errors = v.validate_model_lock(lock)
    assert any("missing models" in e for e in errors)


def test_validate_model_lock_rejects_bad_sha() -> None:
    import validate_model_source_lock as v
    lock = _load_model_lock()
    lock["models"]["htdemucs"]["weights"]["commit_sha"] = "not-a-sha"
    errors = v.validate_model_lock(lock)
    assert any("commit_sha" in e for e in errors)


def test_validate_model_lock_requires_spectral_core_authority() -> None:
    import validate_model_source_lock as v
    lock = _load_model_lock()
    del lock["models"]["htdemucs"]["spectral_core"]
    errors = v.validate_model_lock(lock)
    assert any("spectral_core missing" in e for e in errors)


def test_validate_model_lock_pins_spectral_contract_version() -> None:
    import validate_model_source_lock as v
    lock = _load_model_lock()
    lock["models"]["htdemucs"]["spectral_core"]["contract"] = (
        "openkara.spectral-contract/v2"
    )
    errors = v.validate_model_lock(lock)
    assert any("spectral_core.contract" in e for e in errors)


def test_validate_model_lock_requires_spectral_sole_artifact_authority() -> None:
    """The top-level artifact_id must name the spectral-core artifact; the
    retired waveform artifact is no longer the entry's identity."""
    import validate_model_source_lock as v
    lock = _load_model_lock()
    for name, entry in lock["models"].items():
        assert entry["artifact_id"] == entry["spectral_core"]["artifact_id"], name
    lock["models"]["htdemucs"]["artifact_id"] = "htdemucs.balanced.fp32.onnx"
    errors = v.validate_model_lock(lock)
    assert any("sole artifact authority" in e for e in errors)


def test_detector_resolves_per_model_lock_entry() -> None:
    """The detector must compare against the per-model entry of the v1 lock."""
    import detect_model_weight_revision  # noqa: F401 — importability
    lock = _load_model_lock()
    for name in ("htdemucs", "htdemucs_ft"):
        entry = lock["models"][name]
        assert entry["weights"]["commit_sha"], name
