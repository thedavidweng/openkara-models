"""Tests for the trend report generator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _quality_report(mse: float = 1e-6) -> dict:
    return {
        "schema_version": "openkara.quality-report/v1",
        "model": "htdemucs", "onnx_path": "m.onnx", "tier": "pr",
        "mse_threshold": 1e-4,
        "results": [{
            "fixture_id": "f", "category": "silence", "tier": "pr",
            "onnx_shape": [1, 4, 2, 100], "pytorch_shape": [1, 4, 2, 100],
            "shape_match": True, "expected_shape_match": True,
            "onnx_has_nan": False, "onnx_has_inf": False,
            "pytorch_has_nan": False, "pytorch_has_inf": False,
            "mse": mse, "mae": mse / 10, "max_abs_error": mse * 10,
            "onnx_output_digest": "a" * 64, "pytorch_output_digest": "b" * 64,
        }],
    }


def test_trend_with_pending_baseline() -> None:
    import generate_trend_report as t
    trend = t.generate_trend("htdemucs.balanced.fp32.onnx", _quality_report(), None)
    assert "error" not in trend
    assert trend["schema_version"] == "openkara.trend-report/v1"
    # With pending baseline, deltas are None but trends are still listed.
    mse_trend = next(tr for tr in trend["trends"] if tr["metric"] == "mse")
    assert mse_trend["candidate"] == 1e-6


def test_trend_unknown_artifact() -> None:
    import generate_trend_report as t
    trend = t.generate_trend("nonexistent.onnx", _quality_report(), None)
    assert "error" in trend


def test_trend_markdown() -> None:
    import generate_trend_report as t
    trend = t.generate_trend("htdemucs.balanced.fp32.onnx", _quality_report(), None)
    md = t._markdown(trend)
    assert "Trend report" in md
    assert "mse" in md


def test_cli() -> None:
    q = Path("/tmp/test-trend-quality.json")
    q.write_text(json.dumps(_quality_report()))
    out = Path("/tmp/test-trend.json")
    md = Path("/tmp/test-trend.md")
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "generate_trend_report.py"),
         "--artifact-id", "htdemucs.balanced.fp32.onnx",
         "--quality-report", str(q), "--output", str(out), "--markdown", str(md)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert out.is_file()
    assert md.is_file()
    trend = json.loads(out.read_text())
    assert trend["schema_version"] == "openkara.trend-report/v1"
    q.unlink(); out.unlink(); md.unlink()


def test_cli_help() -> None:
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "generate_trend_report.py"), "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert "--artifact-id" in r.stdout
