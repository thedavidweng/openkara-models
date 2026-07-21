"""Tests for the runtime benchmark report validator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _valid_report() -> dict:
    return {
        "schema_version": "openkara.runtime-benchmark/v1",
        "runtime_archive": {
            "name": "onnxruntime-1.27.1-openkara-x86_64-unknown-linux-gnu.tar.gz",
            "sha256": "a" * 64,
            "size": 1000000,
        },
        "target": "x86_64-unknown-linux-gnu",
        "frames": 343980,
        "warmup": 3,
        "iters": 10,
        "results": [
            {
                "model": "htdemucs.onnx",
                "model_artifact_id": "htdemucs.balanced.fp32.onnx",
                "cold_load_s": 1.5,
                "first_window_s": 2.3,
                "warm_median_s": 1.1,
                "warm_p95_s": 1.3,
                "warm_iters": 10,
                "peak_rss_kb": 500000,
                "rss_delta_kb": 100000,
                "providers": ["CPUExecutionProvider"],
                "fallback_node_count": 0,
                "output_shape": [[1, 4, 2, 343980]],
                "shape_errors": [],
                "frames": 343980,
            }
        ],
    }


def test_validate_valid_report(tmp_path: Path) -> None:
    sys.path.insert(0, str(SCRIPTS))
    import validate_benchmark_report as v
    report = _valid_report()
    assert v.validate_report(report, tier="pr") == []
    assert v.validate_report(report, tier="release") == []


def test_validate_rejects_shape_errors(tmp_path: Path) -> None:
    sys.path.insert(0, str(SCRIPTS))
    import validate_benchmark_report as v
    report = _valid_report()
    report["results"][0]["shape_errors"] = ["output[0] contains NaN"]
    errors = v.validate_report(report)
    assert any("shape errors" in e for e in errors)


def test_validate_rejects_fallback_mismatch(tmp_path: Path) -> None:
    sys.path.insert(0, str(SCRIPTS))
    import validate_benchmark_report as v
    report = _valid_report()
    report["results"][0]["fallback_node_count_mismatch"] = {"baseline": 0, "actual": 5}
    errors = v.validate_report(report)
    assert any("fallback node count" in e for e in errors)


def test_validate_rejects_non_finite_latency(tmp_path: Path) -> None:
    sys.path.insert(0, str(SCRIPTS))
    import validate_benchmark_report as v
    report = _valid_report()
    report["results"][0]["warm_median_s"] = float("inf")
    errors = v.validate_report(report)
    assert any("warm_median_s" in e for e in errors)


def test_validate_release_tier_requires_full_window(tmp_path: Path) -> None:
    sys.path.insert(0, str(SCRIPTS))
    import validate_benchmark_report as v
    report = _valid_report()
    report["frames"] = 1000
    report["results"][0]["frames"] = 1000
    errors = v.validate_report(report, tier="release")
    assert any("frames=343980" in e for e in errors)
    # PR tier allows any frame count.
    assert v.validate_report(report, tier="pr") == []


def test_validate_rejects_bad_schema_version(tmp_path: Path) -> None:
    sys.path.insert(0, str(SCRIPTS))
    import validate_benchmark_report as v
    report = _valid_report()
    report["schema_version"] = "wrong"
    errors = v.validate_report(report)
    assert any("schema" in e for e in errors)


def test_validate_cli(tmp_path: Path) -> None:
    report = _valid_report()
    p = tmp_path / "report.json"
    p.write_text(json.dumps(report))
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "validate_benchmark_report.py"), str(p), "--tier", "release"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_benchmark_script_help() -> None:
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_runtime_benchmarks.py"), "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert "--frames" in r.stdout
    assert "--runtime" in r.stdout
