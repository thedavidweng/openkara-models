"""Tests for the runtime quality report validator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _valid_report() -> dict:
    return {
        "schema_version": "openkara.runtime-quality-report/v1",
        "runtime_archive": {
            "name": "onnxruntime-1.27.1-openkara-x86_64-unknown-linux-gnu.tar.gz",
            "sha256": "a" * 64,
            "size": 50000000,
            "installed_size": 150000000,
        },
        "onnx_path": "models/htdemucs.onnx",
        "tier": "pr",
        "rtf_threshold": 10.0,
        "results": [
            {
                "fixture_id": "synth-silence-343980",
                "category": "silence",
                "tier": "pr",
                "frames": 343980,
                "audio_duration_s": 7.8,
                "cold_load_s": 1.5,
                "first_window_s": 2.3,
                "warm_median_s": 1.1,
                "warm_p95_s": 1.3,
                "warm_iters": 10,
                "rtf_first": 0.29,
                "rtf_warm": 0.14,
                "peak_rss_kb": 500000,
                "rss_delta_kb": 100000,
                "providers": ["CPUExecutionProvider"],
                "fallback_node_count": 0,
                "output_shape": [[1, 4, 2, 343980]],
                "shape_errors": [],
            }
        ],
    }


def test_validate_valid_report() -> None:
    import validate_runtime_quality_report as v
    assert v.validate_report(_valid_report()) == []


def test_validate_rejects_shape_errors() -> None:
    import validate_runtime_quality_report as v
    report = _valid_report()
    report["results"][0]["shape_errors"] = ["output[0] contains NaN"]
    errors = v.validate_report(report)
    assert any("shape errors" in e for e in errors)


def test_validate_rejects_rtf_exceeding_threshold() -> None:
    import validate_runtime_quality_report as v
    report = _valid_report()
    report["results"][0]["rtf_warm"] = 15.0
    report["rtf_threshold"] = 10.0
    errors = v.validate_report(report)
    assert any("RTF" in e for e in errors)


def test_validate_rejects_bad_schema() -> None:
    import validate_runtime_quality_report as v
    report = _valid_report()
    report["schema_version"] = "wrong"
    errors = v.validate_report(report)
    assert any("schema" in e for e in errors)


def test_validate_cli() -> None:
    report = _valid_report()
    p = Path("/tmp/test-runtime-quality-report.json")
    p.write_text(json.dumps(report))
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "validate_runtime_quality_report.py"), str(p)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout
    p.unlink()


def test_runtime_quality_suite_help() -> None:
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_runtime_quality_suite.py"), "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert "--runtime" in r.stdout
    assert "--onnx" in r.stdout
    assert "--tier" in r.stdout


def test_safe_extract_tar_rejects_path_traversal(tmp_path: Path) -> None:
    """Regression: tar member with .. must be rejected, not extracted.

    Extraction now goes through archive_utils.safe_extract (the per-script
    _safe_extract_tar helper was removed in favor of the shared module)."""
    import tarfile
    import io
    import archive_utils
    # Build a tar with a path-traversal member.
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = b"evil"
        info = tarfile.TarInfo(name="../evil.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    archive = tmp_path / "evil.tar.gz"
    archive.write_bytes(buf.getvalue())
    dest = tmp_path / "dest"
    with pytest.raises(ValueError, match="unsafe"):
        archive_utils.safe_extract(archive, dest)
    # Confirm nothing escaped dest.
    assert not (tmp_path / "evil.txt").exists()


def test_safe_extract_zip_rejects_path_traversal(tmp_path: Path) -> None:
    """Regression: zip member with .. must be rejected, not extracted."""
    import zipfile
    import archive_utils
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../evil.txt", "evil")
    dest = tmp_path / "dest"
    with pytest.raises(ValueError, match="unsafe"):
        archive_utils.safe_extract(archive, dest)
    assert not (tmp_path / "evil.txt").exists()


def test_safe_extract_tar_allows_normal_members(tmp_path: Path) -> None:
    import tarfile
    import io
    import archive_utils
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = b"ok"
        info = tarfile.TarInfo(name="lib/libonnxruntime.so")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    archive = tmp_path / "ok.tar.gz"
    archive.write_bytes(buf.getvalue())
    dest = tmp_path / "dest"
    archive_utils.safe_extract(archive, dest)
    assert (dest / "lib" / "libonnxruntime.so").read_bytes() == b"ok"


def _make_minimal_runtime_archive(path: Path) -> None:
    """Write a tar.gz with a zero-byte libonnxruntime.so for arg-validation tests."""
    import io
    import tarfile
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="libonnxruntime.so")
        info.size = 0
        tar.addfile(info, io.BytesIO(b""))
    path.write_bytes(buf.getvalue())


def test_iters_zero_rejected(tmp_path: Path) -> None:
    """Regression: --iters 0 must produce a controlled error, not an
    unhandled StatisticsError from median of an empty list."""
    archive = tmp_path / "rt.tar.gz"
    _make_minimal_runtime_archive(archive)
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"fake")
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_runtime_quality_suite.py"),
         "--runtime", str(archive), "--onnx", str(onnx), "--iters", "0"],
        capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "--iters must be >= 1" in r.stderr
    # The controlled error must surface, not an unhandled traceback.
    assert "Traceback" not in r.stderr
    assert "StatisticsError" not in r.stderr


def test_warmup_negative_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "rt.tar.gz"
    _make_minimal_runtime_archive(archive)
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"fake")
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_runtime_quality_suite.py"),
         "--runtime", str(archive), "--onnx", str(onnx), "--warmup", "-1"],
        capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "--warmup must be >= 0" in r.stderr
