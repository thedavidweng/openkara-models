"""Tests for the quality report validator and correctness metrics."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _valid_report() -> dict:
    return {
        "schema_version": "openkara.quality-report/v1",
        "model": "htdemucs",
        "onnx_path": "models/htdemucs.onnx",
        "tier": "pr",
        "mse_threshold": 1e-4,
        "results": [
            {
                "fixture_id": "synth-silence-343980",
                "category": "silence",
                "tier": "pr",
                "onnx_shape": [1, 4, 2, 343980],
                "pytorch_shape": [1, 4, 2, 343980],
                "shape_match": True,
                "expected_shape": [1, 4, 2, 343980],
                "expected_shape_match": True,
                "onnx_has_nan": False,
                "onnx_has_inf": False,
                "pytorch_has_nan": False,
                "pytorch_has_inf": False,
                "mse": 1e-6,
                "mae": 1e-7,
                "max_abs_error": 1e-5,
                "onnx_output_digest": "a" * 64,
                "pytorch_output_digest": "b" * 64,
            }
        ],
    }


def test_validate_valid_report() -> None:
    import validate_quality_report as v
    assert v.validate_report(_valid_report()) == []


def test_validate_rejects_nan() -> None:
    import validate_quality_report as v
    report = _valid_report()
    report["results"][0]["onnx_has_nan"] = True
    errors = v.validate_report(report)
    assert any("NaN" in e for e in errors)


def test_validate_rejects_shape_mismatch() -> None:
    import validate_quality_report as v
    report = _valid_report()
    report["results"][0]["shape_match"] = False
    errors = v.validate_report(report)
    assert any("shape mismatch" in e for e in errors)


def test_validate_rejects_mse_exceeding_threshold() -> None:
    import validate_quality_report as v
    report = _valid_report()
    report["results"][0]["mse"] = 1e-3
    report["mse_threshold"] = 1e-4
    errors = v.validate_report(report)
    assert any("MSE" in e for e in errors)


def test_validate_rejects_bad_schema_version() -> None:
    import validate_quality_report as v
    report = _valid_report()
    report["schema_version"] = "wrong"
    errors = v.validate_report(report)
    assert any("schema" in e for e in errors)


def test_correctness_metrics_shape_match() -> None:
    import run_quality_suite as q
    a = np.zeros((1, 4, 2, 100), dtype=np.float32)
    b = np.zeros((1, 4, 2, 100), dtype=np.float32)
    m = q._correctness_metrics(a, b, [1, 4, 2, 100])
    assert m["shape_match"] is True
    assert m["expected_shape_match"] is True
    assert m["mse"] == 0.0
    assert m["mae"] == 0.0
    assert m["max_abs_error"] == 0.0


def test_correctness_metrics_shape_mismatch() -> None:
    import run_quality_suite as q
    a = np.zeros((1, 4, 2, 100), dtype=np.float32)
    b = np.zeros((1, 4, 2, 200), dtype=np.float32)
    m = q._correctness_metrics(a, b, [1, 4, 2, 100])
    assert m["shape_match"] is False
    assert m["mse"] is None


def test_correctness_metrics_nan() -> None:
    import run_quality_suite as q
    a = np.zeros((1, 4, 2, 100), dtype=np.float32)
    b = np.full((1, 4, 2, 100), np.nan, dtype=np.float32)
    m = q._correctness_metrics(a, b, [1, 4, 2, 100])
    assert m["onnx_has_nan"] is True
    assert m["mse"] is None


def test_output_digest_deterministic() -> None:
    import run_quality_suite as q
    a = np.zeros((1, 4, 2, 100), dtype=np.float32)
    assert q._output_digest(a) == q._output_digest(a)
    b = np.ones((1, 4, 2, 100), dtype=np.float32)
    assert q._output_digest(a) != q._output_digest(b)


def test_gate_failures_on_shape_mismatch_without_expected_shape() -> None:
    """Regression: gate must fail when onnx vs pytorch shapes disagree, even
    if expected_shape is not set. Previously the runner returned success in
    this case but the validator rejected the report."""
    import run_quality_suite as q
    results = [{
        "fixture_id": "fx",
        "onnx_has_nan": False,
        "onnx_has_inf": False,
        "shape_match": False,  # onnx vs pytorch mismatch
        # expected_shape_match intentionally absent
        "mse": None,
    }]
    failures = q.gate_failures(results, mse_threshold=1e-4)
    assert any("shape mismatch" in f for f in failures)


def test_gate_failures_on_expected_shape_mismatch() -> None:
    import run_quality_suite as q
    results = [{
        "fixture_id": "fx",
        "onnx_has_nan": False,
        "onnx_has_inf": False,
        "shape_match": True,
        "expected_shape_match": False,
        "mse": 0.0,
    }]
    failures = q.gate_failures(results, mse_threshold=1e-4)
    assert any("expected_output_shape" in f for f in failures)


def test_gate_failures_clean() -> None:
    import run_quality_suite as q
    results = [{
        "fixture_id": "fx",
        "onnx_has_nan": False,
        "onnx_has_inf": False,
        "shape_match": True,
        "expected_shape_match": True,
        "mse": 1e-6,
    }]
    assert q.gate_failures(results, mse_threshold=1e-4) == []


def test_gate_failures_aligned_with_validator() -> None:
    """Any report that passes the gate must also pass the validator, and
    vice versa. This is the contract that makes runner + validator coherent."""
    import run_quality_suite as q
    import validate_quality_report as v
    base = _valid_report()
    # _valid_report() already passes both gate and validator.
    assert q.gate_failures(base["results"], base["mse_threshold"]) == []
    assert v.validate_report(base) == []


def test_validate_cli() -> None:
    report = _valid_report()
    p = Path("/tmp/test-quality-report.json")
    p.write_text(json.dumps(report))
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "validate_quality_report.py"), str(p)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout
    p.unlink()


def test_quality_suite_help() -> None:
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_quality_suite.py"), "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert "--tier" in r.stdout
    assert "--onnx" in r.stdout
