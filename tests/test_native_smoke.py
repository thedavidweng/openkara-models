"""Tests for the native smoke harness wrapper and benchmark shape-validation
bypass rejection.

Covers:
  - run_native_smoke.py CLI help and report assembly (without building the
    C++ harness, which requires an ORT source checkout).
  - run_runtime_benchmarks.py rejects a bare --model without
    --expected-output-shape or --catalog (prevents bypassing output-shape
    validation).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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


def test_benchmark_rejects_bare_model_without_shape(tmp_path: Path) -> None:
    """A bare --model without --expected-output-shape or --catalog must be
    rejected so output-shape validation is not bypassed."""
    # Create a fake runtime archive so the --runtime check passes; the script
    # will fail at the model-bypass check before extracting.
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
         "--target", "x86_64-unknown-linux-gnu"],
        capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "expected-output-shape" in r.stderr or "expected-output-shape" in r.stdout


def test_benchmark_accepts_model_with_explicit_shape(tmp_path: Path) -> None:
    """--model with --expected-output-shape should pass the bypass check
    (it will fail later at extraction/inference, but not at the bypass guard)."""
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
         "--expected-output-shape", "1,4,2,343980",
         "--target", "x86_64-unknown-linux-gnu",
         "--report", str(tmp_path / "report.json")],
        capture_output=True, text=True,
    )
    # It should get past the bypass guard; it will fail at inference because
    # the model is fake, but the error should NOT mention expected-output-shape.
    assert "expected-output-shape" not in r.stderr
