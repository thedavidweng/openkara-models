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
        "pytorch_has_nan": False,
        "pytorch_has_inf": False,
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
        "pytorch_has_nan": False,
        "pytorch_has_inf": False,
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
        "pytorch_has_nan": False,
        "pytorch_has_inf": False,
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


@pytest.mark.parametrize('flag', ['pytorch_has_nan', 'pytorch_has_inf'])
def test_pytorch_nonfinite_rejected_by_runner_and_validator(flag):
    import run_quality_suite as q
    import validate_quality_report as v
    report = _valid_report()
    report['results'][0][flag] = True
    assert q.gate_failures(report['results'], report['mse_threshold'])
    assert v.validate_report(report)


def test_ensemble_reference_uses_per_source_weights(monkeypatch) -> None:
    """Regression: htdemucs_ft is a four-model BagOfModels with identity weights.

    Upstream ``apply_model`` multiplies each sub-model's source output by the
    pinned weight matrix (4x4 identity: one sub-model per source) and divides
    by the per-source total. The strict quality reference must do the same;
    using demucs_loader.load() unwraps to sub-model 0, and a plain average is
    also a different model.
    """
    import contextlib
    import sys
    import types

    import demucs_loader
    import run_quality_suite as q

    class Tensor:
        def __init__(self, arr):
            self.arr = np.asarray(arr, dtype=np.float32)

        def unsqueeze(self, axis):
            return Tensor(np.expand_dims(self.arr, axis))

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.arr

    class Model:
        def __init__(self, value):
            self.value = value

        def __call__(self, tensor):
            frames = tensor.arr.shape[-1]
            return Tensor(np.full((1, 4, 2, frames), self.value, dtype=np.float32))

    fake_torch = types.SimpleNamespace(
        from_numpy=lambda arr: Tensor(arr),
        no_grad=lambda: contextlib.nullcontext(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        demucs_loader, "iter_sub_models",
        lambda _name: (
            2, 4,
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0]],
            iter([(Model(2.0), 4, 0), (Model(4.0), 4, 1)]),
        ),
    )
    monkeypatch.setattr(
        demucs_loader, "load",
        lambda _name: pytest.fail("ensemble path must not unwrap to one sub-model"),
    )

    out = q._compute_pytorch_reference("htdemucs_ft", np.zeros((2, 4), np.float32))
    assert out.shape == (1, 4, 2, 4)
    np.testing.assert_allclose(out[0, :, 0, 0], [2.0, 4.0, 4.0, 4.0])


def test_normalize_weighted_rejects_zero_source_total() -> None:
    """A source with zero total weight is undefined, never NaN."""
    import demucs_loader

    with pytest.raises(ValueError, match="positive total"):
        demucs_loader.normalize_weighted(
            np.zeros((1, 4, 2, 4), dtype=np.float64),
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        )
