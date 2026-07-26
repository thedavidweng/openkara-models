"""Tests for the native smoke harness wrapper and benchmark model resolution.

Covers:
  - run_native_smoke.py CLI help, the spectral-core shape gate, and report
    assembly (without building the C++ harness, which requires an ORT source
    checkout).
  - run_runtime_benchmarks.py model/catalog resolution: a bare --model is
    accepted (the spectral-core output shapes are asserted intrinsically), and
    an unknown --model-artifact-id is rejected.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_native_smoke  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def test_run_native_smoke_cli_help() -> None:
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_native_smoke.py"), "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert "--runtime" in r.stdout
    assert "--model" in r.stdout
    assert "--provider" in r.stdout
    assert "--target" in r.stdout


def test_run_native_smoke_requires_args() -> None:
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_native_smoke.py"),
         "--runtime", "/nonexistent", "--model", "/nonexistent",
         "--target", "x86_64-unknown-linux-gnu", "--provider", "cpu",
         "--ort-source", "/nonexistent", "--report", "/tmp/x.json"],
        capture_output=True, text=True,
    )
    # Should fail because the runtime archive / source do not exist.
    assert r.returncode != 0


SPECTRAL_SHAPE = "[1,4,2,2,2048,336];[1,4,2,343980]"


def _valid_smoke_report(provider: str) -> dict[str, object]:
    return {
        "requested_provider": provider,
        "harness_exit_code": 0,
        "session_creation": "ok",
        "inference": "ok",
        "finite_output": True,
        "output_shape": SPECTRAL_SHAPE,
        "used_fallback": False,
        "provider_assignment": provider,
        "provider_node_count": 1,
        "total_node_count": 10,
    }


def test_strict_smoke_gate_accepts_requested_provider() -> None:
    assert run_native_smoke.validation_failures(
        _valid_smoke_report("directml"), "directml"
    ) == []


def test_strict_smoke_gate_rejects_whole_session_cpu_fallback() -> None:
    report = _valid_smoke_report("directml")
    report["provider_assignment"] = "cpu"
    report["used_fallback"] = True
    report["provider_node_count"] = 0
    failures = run_native_smoke.validation_failures(report, "directml")
    assert any("fallback" in failure for failure in failures)
    assert any("assignment mismatch" in failure for failure in failures)
    assert any("provider_node_count" in failure for failure in failures)


def test_strict_smoke_gate_rejects_wrong_shape() -> None:
    report = _valid_smoke_report("xnnpack")
    report["output_shape"] = "[1,3,2,343980]"  # not the spectral two-output shape
    failures = run_native_smoke.validation_failures(report, "xnnpack")
    assert any("output_shape" in failure for failure in failures)


def test_strict_smoke_gate_accepts_spectral_output_shape() -> None:
    """The spectral-core two-output shape string is the default (and only)
    expectation."""
    report = _valid_smoke_report("cpu")
    assert run_native_smoke.validation_failures(report, "cpu") == []


def test_spectral_output_shape_constant() -> None:
    assert run_native_smoke.SPECTRAL_OUTPUT_SHAPE == SPECTRAL_SHAPE


def test_strict_smoke_gate_rejects_zero_provider_nodes() -> None:
    report = _valid_smoke_report("coreml")
    report["provider_node_count"] = 0
    failures = run_native_smoke.validation_failures(report, "coreml")
    assert any("provider_node_count" in failure for failure in failures)


def test_strict_smoke_gate_accepts_skipped_for_hardware_provider() -> None:
    """A 'skipped' session_creation is accepted for hardware-dependent
    providers (directml, coreml) when the hardware is absent."""
    for provider in ("directml", "coreml"):
        report = {
            "requested_provider": provider,
            "harness_exit_code": 1,
            "session_creation": "skipped",
            "inference": "skipped",
            "finite_output": False,
            "output_shape": "",
            "used_fallback": False,
            "provider_assignment": "",
            "provider_node_count": None,
            "total_node_count": None,
        }
        assert run_native_smoke.validation_failures(report, provider) == []


def test_strict_smoke_gate_rejects_skipped_for_cpu() -> None:
    """CPU is never skippable — a 'skipped' status must fail the gate."""
    report = {
        "requested_provider": "cpu",
        "harness_exit_code": 1,
        "session_creation": "skipped",
        "inference": "skipped",
        "finite_output": False,
        "output_shape": "",
        "used_fallback": False,
        "provider_assignment": "",
        "provider_node_count": None,
        "total_node_count": None,
    }
    failures = run_native_smoke.validation_failures(report, "cpu")
    assert any("session creation" in f for f in failures)


def test_strict_smoke_gate_rejects_skipped_for_xnnpack() -> None:
    """XNNPACK is not hardware-dependent — a 'skipped' status must fail."""
    report = {
        "requested_provider": "xnnpack",
        "harness_exit_code": 1,
        "session_creation": "skipped",
        "inference": "skipped",
        "finite_output": False,
        "output_shape": "",
        "used_fallback": False,
        "provider_assignment": "",
        "provider_node_count": None,
        "total_node_count": None,
    }
    failures = run_native_smoke.validation_failures(report, "xnnpack")
    assert any("session creation" in f for f in failures)


def test_benchmark_bare_model_not_rejected_for_missing_shape(tmp_path: Path) -> None:
    """A bare --model is accepted: a spectral-core model's output shapes are
    asserted intrinsically, so there is no expected-shape bypass guard. The
    run fails later (fake model), but never for a missing shape argument."""
    import io
    import tarfile
    archive = tmp_path / "onnxruntime-1.27.1-openkara-x86_64-unknown-linux-gnu.tar.gz"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = b"x"
        info = tarfile.TarInfo(name="libonnxruntime.so")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    archive.write_bytes(buf.getvalue())
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake")
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_runtime_benchmarks.py"),
         "--runtime", str(archive), "--model", str(model),
         "--target", "x86_64-unknown-linux-gnu",
         "--report", str(tmp_path / "report.json")],
        capture_output=True, text=True,
    )
    # No expected-shape argument exists anymore; the failure is never about one.
    assert "expected-output-shape" not in (r.stderr + r.stdout)


def test_benchmark_rejects_unknown_model_artifact_id(tmp_path: Path) -> None:
    """--catalog with an unknown --model-artifact-id must fail with a clear
    error, not silently benchmark nothing."""
    import io
    import tarfile
    archive = tmp_path / "onnxruntime-1.27.1-openkara-x86_64-unknown-linux-gnu.tar.gz"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = b"x"
        info = tarfile.TarInfo(name="libonnxruntime.so")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    archive.write_bytes(buf.getvalue())
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({
        "artifacts": {"models": [
            {"artifact_id": "htdemucs.balanced.fp32.onnx",
             "filename": "htdemucs.onnx", "download_url": "http://example.com/m.onnx",
             "archive_digest": "0" * 64,
             "model": {"output_semantics": "[1,4,2,343980]"}},
        ]},
        "compatibility": [],
    }))
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_runtime_benchmarks.py"),
         "--runtime", str(archive), "--catalog", str(catalog),
         "--model-artifact-id", "nonexistent.onnx",
         "--target", "x86_64-unknown-linux-gnu",
         "--report", str(tmp_path / "report.json")],
        capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "nonexistent.onnx" in r.stderr or "nonexistent.onnx" in r.stdout


def test_benchmark_catalog_resolves_model_artifact_id(tmp_path: Path) -> None:
    """--catalog with a valid --model-artifact-id must resolve the model and
    proceed past the catalog resolution stage (it will fail later at
    download/inference because the URL is fake, but the error should not
    mention model-artifact-id)."""
    import io
    import tarfile
    archive = tmp_path / "onnxruntime-1.27.1-openkara-x86_64-unknown-linux-gnu.tar.gz"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = b"x"
        info = tarfile.TarInfo(name="libonnxruntime.so")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    archive.write_bytes(buf.getvalue())
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({
        "artifacts": {"models": [
            {"artifact_id": "htdemucs.balanced.fp32.onnx",
             "filename": "htdemucs.onnx", "download_url": "http://example.com/m.onnx",
             "archive_digest": "0" * 64,
             "model": {"output_semantics": "[1,4,2,343980]"}},
        ]},
        "compatibility": [],
    }))
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_runtime_benchmarks.py"),
         "--runtime", str(archive), "--catalog", str(catalog),
         "--model-artifact-id", "htdemucs.balanced.fp32.onnx",
         "--target", "x86_64-unknown-linux-gnu",
         "--report", str(tmp_path / "report.json")],
        capture_output=True, text=True,
    )
    # Should fail at download (fake URL), not at model resolution.
    assert r.returncode != 0
    assert "not found in catalog" not in r.stderr
