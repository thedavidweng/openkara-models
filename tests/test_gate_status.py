"""Tests for the candidate gate status generator.

These tests verify the requirements from issue #20 PR 3/4:

  1. Suite returns non-zero → gate status is ``failed``, overall is not ``passed``.
  2. Model missing → no false success (gate is ``not_run`` or ``unavailable``,
     overall is ``incomplete``).
  3. Required report missing → PR body (Markdown) explicitly shows ``INCOMPLETE``.
  4. ORT candidate missing runtime/model quality evidence → cannot be promoted
     (overall is ``incomplete`` or ``failed``, exit code is non-zero).
  5. Stable pointer is never modified by the gate status generator.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

# A script that always exists, used as a stand-in for quality scripts that
# may not be present on every branch (the quality harness lives on a
# separate stack). Tests monkeypatch the module-level script paths to this.
_EXISTING_SCRIPT = SCRIPTS / "generate_gate_status.py"


@pytest.fixture(autouse=True)
def _stub_quality_scripts(monkeypatch):
    """Point quality-suite script paths to an existing file so that gates
    are classified as available/unavailable based on report presence, not
    on whether the quality harness has been merged into this branch."""
    import generate_gate_status as g
    monkeypatch.setattr(g, "QUALITY_SUITE_SCRIPT", _EXISTING_SCRIPT)
    monkeypatch.setattr(g, "RUNTIME_QUALITY_SUITE_SCRIPT", _EXISTING_SCRIPT)
    monkeypatch.setattr(g, "GATE_ENFORCER_SCRIPT", _EXISTING_SCRIPT)


def _make_report(path: Path, status: str | None = "passed", results: list | None = None) -> Path:
    """Write a minimal report JSON to ``path``.

    If ``status`` is ``None``, the ``status`` field is omitted so the
    classifier falls back to inspecting ``results``.
    """
    data: dict = {}
    if status is not None:
        data["status"] = status
    if results is not None:
        data["results"] = results
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


class TestClassifyReport:
    def test_passed_report(self, tmp_path: Path) -> None:
        import generate_gate_status as g
        report = _make_report(tmp_path / "q.json", status="passed")
        assert g._classify_report(report, _EXISTING_SCRIPT) == "passed"

    def test_failed_report(self, tmp_path: Path) -> None:
        import generate_gate_status as g
        report = _make_report(tmp_path / "q.json", status="failed")
        assert g._classify_report(report, _EXISTING_SCRIPT) == "failed"

    def test_missing_report_is_not_run(self, tmp_path: Path) -> None:
        import generate_gate_status as g
        assert g._classify_report(tmp_path / "nonexistent.json", _EXISTING_SCRIPT) == "not_run"

    def test_missing_script_is_unavailable(self, tmp_path: Path) -> None:
        import generate_gate_status as g
        report = _make_report(tmp_path / "q.json", status="passed")
        script = tmp_path / "nonexistent_script.py"
        assert g._classify_report(report, script) == "unavailable"

    def test_corrupt_report_is_failed(self, tmp_path: Path) -> None:
        import generate_gate_status as g
        report = tmp_path / "q.json"
        report.write_text("not valid json {{{", encoding="utf-8")
        assert g._classify_report(report, _EXISTING_SCRIPT) == "failed"

    def test_report_with_results_no_errors_is_passed(self, tmp_path: Path) -> None:
        import generate_gate_status as g
        report = _make_report(tmp_path / "q.json", status=None, results=[{"mse": 0.001}, {"mse": 0.002}])
        assert g._classify_report(report, _EXISTING_SCRIPT) == "passed"

    def test_report_with_error_in_results_is_failed(self, tmp_path: Path) -> None:
        import generate_gate_status as g
        report = _make_report(tmp_path / "q.json", status=None, results=[{"mse": 0.001}, {"error": "NaN"}])
        assert g._classify_report(report, _EXISTING_SCRIPT) == "failed"


class TestOrtCandidate:
    def test_all_pass(self, tmp_path: Path) -> None:
        import generate_gate_status as g
        q = _make_report(tmp_path / "quality.json", status="passed")
        r = _make_report(tmp_path / "runtime.json", status="passed")
        gr = _make_report(tmp_path / "gate.json")
        gr.write_text(json.dumps({"overall": "passed"}) + "\n", encoding="utf-8")
        rt_dir = tmp_path / "runtimes"
        rt_dir.mkdir()
        for i in range(5):
            (rt_dir / f"onnxruntime-target-{i}.tar.gz").write_bytes(b"x")
        status = g.generate_gate_status(
            "ort", q, r, gr, rt_dir,
        )
        assert status["overall_status"] == "passed"
        assert status["incomplete"] is False

    def test_missing_runtime_artifacts_is_incomplete(self, tmp_path: Path) -> None:
        import generate_gate_status as g
        q = _make_report(tmp_path / "quality.json", status="passed")
        r = _make_report(tmp_path / "runtime.json", status="passed")
        gr = tmp_path / "gate.json"
        gr.write_text(json.dumps({"overall": "passed"}) + "\n", encoding="utf-8")
        status = g.generate_gate_status(
            "ort", q, r, gr, None,
        )
        assert status["overall_status"] == "incomplete"
        assert status["incomplete"] is True
        assert status["gates"]["runtime_builds"]["status"] == "not_run"

    def test_partial_runtime_targets_is_failed(self, tmp_path: Path) -> None:
        import generate_gate_status as g
        q = _make_report(tmp_path / "quality.json", status="passed")
        r = _make_report(tmp_path / "runtime.json", status="passed")
        gr = tmp_path / "gate.json"
        gr.write_text(json.dumps({"overall": "passed"}) + "\n", encoding="utf-8")
        rt_dir = tmp_path / "runtimes"
        rt_dir.mkdir()
        for i in range(3):  # only 3 of 5
            (rt_dir / f"onnxruntime-target-{i}.tar.gz").write_bytes(b"x")
        status = g.generate_gate_status(
            "ort", q, r, gr, rt_dir, expected_targets=5,
        )
        assert status["gates"]["runtime_builds"]["status"] == "failed"
        assert status["overall_status"] == "failed"

    def test_missing_quality_evidence_is_incomplete(self, tmp_path: Path) -> None:
        """ORT candidate with no quality/runtime reports cannot be promoted."""
        import generate_gate_status as g
        rt_dir = tmp_path / "runtimes"
        rt_dir.mkdir()
        for i in range(5):
            (rt_dir / f"onnxruntime-target-{i}.tar.gz").write_bytes(b"x")
        status = g.generate_gate_status(
            "ort", None, None, None, rt_dir,
        )
        assert status["overall_status"] == "incomplete"
        assert status["incomplete"] is True
        assert status["gates"]["model_quality"]["status"] == "not_run"
        assert status["gates"]["runtime_quality"]["status"] == "not_run"

    def test_failed_quality_is_failed(self, tmp_path: Path) -> None:
        """Suite returns non-zero (failed status) → overall is failed."""
        import generate_gate_status as g
        q = _make_report(tmp_path / "quality.json", status="failed")
        r = _make_report(tmp_path / "runtime.json", status="passed")
        gr = tmp_path / "gate.json"
        gr.write_text(json.dumps({"overall": "passed"}) + "\n", encoding="utf-8")
        rt_dir = tmp_path / "runtimes"
        rt_dir.mkdir()
        for i in range(5):
            (rt_dir / f"onnxruntime-target-{i}.tar.gz").write_bytes(b"x")
        status = g.generate_gate_status(
            "ort", q, r, gr, rt_dir,
        )
        assert status["overall_status"] == "failed"
        assert status["gates"]["model_quality"]["status"] == "failed"


class TestModelCandidate:
    def test_all_pass(self, tmp_path: Path) -> None:
        import generate_gate_status as g
        q = _make_report(tmp_path / "quality.json", status="passed")
        r = _make_report(tmp_path / "runtime.json", status="passed")
        model = tmp_path / "htdemucs.onnx"
        model.write_bytes(b"model")
        baseline = tmp_path / "baseline.json"
        baseline.write_text(json.dumps({"verdict": "no_regression"}) + "\n", encoding="utf-8")
        status = g.generate_gate_status(
            "model", q, r, None, None,
            converted_model=model, baseline_comparison=baseline,
        )
        assert status["overall_status"] == "passed"

    def test_missing_conversion_is_incomplete(self, tmp_path: Path) -> None:
        import generate_gate_status as g
        q = _make_report(tmp_path / "quality.json", status="passed")
        r = _make_report(tmp_path / "runtime.json", status="passed")
        baseline = tmp_path / "baseline.json"
        baseline.write_text(json.dumps({"verdict": "no_regression"}) + "\n", encoding="utf-8")
        status = g.generate_gate_status(
            "model", q, r, None, None,
            converted_model=None, baseline_comparison=baseline,
        )
        assert status["overall_status"] == "incomplete"
        assert status["gates"]["model_conversion"]["status"] == "not_run"

    def test_missing_baseline_comparison_is_incomplete(self, tmp_path: Path) -> None:
        """Model-weight update must not pass with only a summary update."""
        import generate_gate_status as g
        q = _make_report(tmp_path / "quality.json", status="passed")
        r = _make_report(tmp_path / "runtime.json", status="passed")
        model = tmp_path / "htdemucs.onnx"
        model.write_bytes(b"model")
        status = g.generate_gate_status(
            "model", q, r, None, None,
            converted_model=model, baseline_comparison=None,
        )
        assert status["overall_status"] == "incomplete"
        assert status["gates"]["baseline_comparison"]["status"] == "not_run"

    def test_regressed_baseline_is_failed(self, tmp_path: Path) -> None:
        import generate_gate_status as g
        q = _make_report(tmp_path / "quality.json", status="passed")
        r = _make_report(tmp_path / "runtime.json", status="passed")
        model = tmp_path / "htdemucs.onnx"
        model.write_bytes(b"model")
        baseline = tmp_path / "baseline.json"
        baseline.write_text(json.dumps({"verdict": "regressed"}) + "\n", encoding="utf-8")
        status = g.generate_gate_status(
            "model", q, r, None, None,
            converted_model=model, baseline_comparison=baseline,
        )
        assert status["overall_status"] == "failed"
        assert status["gates"]["baseline_comparison"]["status"] == "failed"


class TestMarkdownRendering:
    def test_incomplete_markdown_shows_incomplete(self, tmp_path: Path) -> None:
        """Required report missing → PR body explicitly shows INCOMPLETE."""
        import generate_gate_status as g
        status = g.generate_gate_status("ort", None, None, None, None)
        md = g.render_markdown(status)
        assert "INCOMPLETE" in md
        assert "NOT verified" in md
        assert "NOT ready" in md

    def test_passed_markdown_shows_passed(self, tmp_path: Path) -> None:
        import generate_gate_status as g
        q = _make_report(tmp_path / "quality.json", status="passed")
        r = _make_report(tmp_path / "runtime.json", status="passed")
        gr = tmp_path / "gate.json"
        gr.write_text(json.dumps({"overall": "passed"}) + "\n", encoding="utf-8")
        rt_dir = tmp_path / "runtimes"
        rt_dir.mkdir()
        for i in range(5):
            (rt_dir / f"onnxruntime-target-{i}.tar.gz").write_bytes(b"x")
        status = g.generate_gate_status("ort", q, r, gr, rt_dir)
        md = g.render_markdown(status)
        assert "PASSED" in md
        assert "INCOMPLETE" not in md

    def test_failed_markdown_shows_failed(self, tmp_path: Path) -> None:
        import generate_gate_status as g
        q = _make_report(tmp_path / "quality.json", status="failed")
        status = g.generate_gate_status("ort", q, None, None, None)
        md = g.render_markdown(status)
        assert "FAILED" in md

    def test_incomplete_markdown_mentions_stable_pointer_not_modified(self, tmp_path: Path) -> None:
        import generate_gate_status as g
        status = g.generate_gate_status("ort", None, None, None, None)
        md = g.render_markdown(status)
        assert "stable pointer" in md.lower()
        assert "NOT modified" in md


class TestStablePointerProtection:
    def test_generate_gate_status_does_not_touch_stable_pointer(self, tmp_path: Path, monkeypatch) -> None:
        """The gate status generator must never modify the stable pointer."""
        import generate_gate_status as g
        # Create a fake stable pointer in a temp dir.
        fake_catalog = tmp_path / "catalog" / "channels"
        fake_catalog.mkdir(parents=True)
        stable = fake_catalog / "stable.json"
        stable.write_text(json.dumps({"channel": "stable", "generation": 1}) + "\n", encoding="utf-8")
        original = stable.read_text(encoding="utf-8")
        # Monkeypatch the module-level path to point to our temp file.
        monkeypatch.setattr(g, "STABLE_POINTER_PATH", stable)
        # Run the generator — it should not touch the stable pointer.
        status = g.generate_gate_status("ort", None, None, None, None)
        assert status["incomplete"] is True
        assert stable.read_text(encoding="utf-8") == original


class TestCLI:
    def test_cli_incomplete_returns_nonzero(self, tmp_path: Path) -> None:
        """CLI returns non-zero when gates are incomplete."""
        r = subprocess.run(
            [sys.executable, str(SCRIPTS / "generate_gate_status.py"),
             "--candidate-type", "ort",
             "--output", str(tmp_path / "status.json"),
             "--markdown", str(tmp_path / "status.md")],
            capture_output=True, text=True,
        )
        assert r.returncode == 1
        status = json.loads((tmp_path / "status.json").read_text())
        assert status["incomplete"] is True
        md = (tmp_path / "status.md").read_text()
        assert "INCOMPLETE" in md

    def test_cli_help(self) -> None:
        r = subprocess.run(
            [sys.executable, str(SCRIPTS / "generate_gate_status.py"), "--help"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "--candidate-type" in r.stdout

    def test_cli_invalid_type(self) -> None:
        r = subprocess.run(
            [sys.executable, str(SCRIPTS / "generate_gate_status.py"),
             "--candidate-type", "invalid"],
            capture_output=True, text=True,
        )
        assert r.returncode != 0
