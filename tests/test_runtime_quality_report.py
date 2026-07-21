"""Tests for the runtime quality report validator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

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
