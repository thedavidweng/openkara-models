"""Tests for the quality gate enforcer."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _quality_report(mse: float = 1e-6, has_nan: bool = False, shape_match: bool = True) -> dict:
    return {
        "schema_version": "openkara.quality-report/v1",
        "model": "htdemucs",
        "onnx_path": "models/htdemucs.onnx",
        "tier": "pr",
        "mse_threshold": 1e-4,
        "results": [{
            "fixture_id": "synth-silence-343980", "category": "silence", "tier": "pr",
            "onnx_shape": [1, 4, 2, 343980], "pytorch_shape": [1, 4, 2, 343980],
            "shape_match": shape_match, "expected_shape_match": shape_match,
            "onnx_has_nan": has_nan, "onnx_has_inf": False,
            "pytorch_has_nan": False, "pytorch_has_inf": False,
            "mse": mse, "mae": mse / 10, "max_abs_error": mse * 10,
            "onnx_output_digest": "a" * 64, "pytorch_output_digest": "b" * 64,
        }],
    }


def _runtime_report(rtf: float = 0.14, cold_load: float = 1.5, peak_rss: int = 500000) -> dict:
    return {
        "schema_version": "openkara.runtime-quality-report/v1",
        "runtime_archive": {"name": "test.tar.gz", "sha256": "a" * 64, "size": 50000000, "installed_size": 150000000},
        "onnx_path": "models/htdemucs.onnx", "tier": "pr", "rtf_threshold": 10.0,
        "results": [{
            "fixture_id": "synth-silence-343980", "category": "silence", "tier": "pr",
            "frames": 343980, "audio_duration_s": 7.8,
            "cold_load_s": cold_load, "first_window_s": 2.3,
            "warm_median_s": 1.1, "warm_p95_s": 1.3, "warm_iters": 10,
            "rtf_first": 0.29, "rtf_warm": rtf,
            "peak_rss_kb": peak_rss, "rss_delta_kb": 100000,
            "providers": ["CPUExecutionProvider"], "fallback_node_count": 0,
            "output_shape": [[1, 4, 2, 343980]], "shape_errors": [],
        }],
    }


def test_enforce_gates_rejects_release_with_unfrozen_baseline() -> None:
    """Release tier must FAIL when baseline is not frozen — release gates
    are NOT landed and no artifact can be promoted without a real baseline."""
    import enforce_quality_gates as g
    errors = g.enforce_gates(
        "htdemucs.balanced.fp32.onnx",
        _quality_report(mse=1e-6),
        _runtime_report(rtf=0.14),
        tier="release",
    )
    assert len(errors) == 1, errors
    assert "NOT FROZEN" in errors[0]


def test_enforce_gates_pr_tier_passes_with_unfrozen_baseline() -> None:
    """PR tier uses absolute thresholds only — no frozen baseline required."""
    import enforce_quality_gates as g
    errors = g.enforce_gates(
        "htdemucs.balanced.fp32.onnx",
        _quality_report(mse=1e-6),
        _runtime_report(rtf=0.14),
        tier="pr",
    )
    assert errors == [], errors


def test_is_baseline_frozen_rejects_pending() -> None:
    """_is_baseline_frozen returns False for pending-pr4-freeze and null."""
    import enforce_quality_gates as g
    assert g._is_baseline_frozen({"frozen_at": None}) is False
    assert g._is_baseline_frozen({"frozen_at": "pending"}) is False
    assert g._is_baseline_frozen({
        "frozen_at": "2026-07-21T00:00:00Z",
        "baseline_quality_report_id": "pending-pr4-freeze",
        "baseline_runtime_report_id": "real-id",
    }) is False
    assert g._is_baseline_frozen({
        "frozen_at": "2026-07-21T00:00:00Z",
        "baseline_quality_report_id": None,
        "baseline_runtime_report_id": "real-id",
    }) is False


def test_is_baseline_frozen_accepts_real() -> None:
    """_is_baseline_frozen returns True when all required fields are real."""
    import enforce_quality_gates as g
    assert g._is_baseline_frozen({
        "frozen_at": "2026-07-21T00:00:00Z",
        "baseline_quality_report_id": "real-quality-id",
        "baseline_runtime_report_id": "real-runtime-id",
    }) is True


def test_enforce_gates_rejects_nan() -> None:
    import enforce_quality_gates as g
    errors = g.enforce_gates(
        "htdemucs.balanced.fp32.onnx",
        _quality_report(has_nan=True),
        None,
        tier="pr",
    )
    assert any("onnx_has_nan" in e for e in errors)


def test_enforce_gates_rejects_shape_mismatch() -> None:
    import enforce_quality_gates as g
    errors = g.enforce_gates(
        "htdemucs.balanced.fp32.onnx",
        _quality_report(shape_match=False),
        None,
        tier="pr",
    )
    assert any("shape_match" in e for e in errors)


def test_enforce_gates_rejects_high_mse() -> None:
    import enforce_quality_gates as g
    errors = g.enforce_gates(
        "htdemucs.balanced.fp32.onnx",
        _quality_report(mse=1e-3),
        None,
        tier="pr",
    )
    assert any("mse" in e for e in errors)


def test_enforce_gates_rejects_unknown_artifact() -> None:
    import enforce_quality_gates as g
    errors = g.enforce_gates("nonexistent.onnx", _quality_report(), None, tier="pr")
    assert any("no baseline" in e for e in errors)


def test_enforce_gates_rejects_no_reports() -> None:
    import enforce_quality_gates as g
    errors = g.enforce_gates("htdemucs.balanced.fp32.onnx", None, None, tier="pr")
    assert any("no candidate" in e for e in errors)


def test_pareto_check_dominated() -> None:
    import enforce_quality_gates as g
    budgets = json.loads((ROOT / "quality" / "budgets.json").read_text())["budgets"]
    candidate = {"rtf_warm": 0.3, "cold_load_s": 2.0, "warm_median_s": 1.5}
    baseline = {"rtf_warm": 0.14, "cold_load_s": 1.5, "warm_median_s": 1.1}
    err = g._check_pareto(candidate, baseline, budgets)
    assert err is not None
    assert "dominated" in err


def test_pareto_check_not_dominated() -> None:
    import enforce_quality_gates as g
    budgets = json.loads((ROOT / "quality" / "budgets.json").read_text())["budgets"]
    # Candidate is worse on RTF but better on cold_load.
    candidate = {"rtf_warm": 0.3, "cold_load_s": 1.0, "warm_median_s": 1.5}
    baseline = {"rtf_warm": 0.14, "cold_load_s": 1.5, "warm_median_s": 1.1}
    err = g._check_pareto(candidate, baseline, budgets)
    assert err is None


def test_cli_help() -> None:
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "enforce_quality_gates.py"), "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert "--artifact-id" in r.stdout
