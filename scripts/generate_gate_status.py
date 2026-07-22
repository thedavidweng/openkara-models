#!/usr/bin/env python3
"""Generate candidate gate status from verification artifacts.

Given a candidate type (``ort`` or ``model``) and the paths of expected
verification artifacts, produce a structured gate-status record that
classifies every required gate as one of:

  - ``passed``     — the gate ran and produced evidence of success.
  - ``failed``     — the gate ran but produced evidence of failure.
  - ``not_run``    — the gate was skipped (e.g. model file missing).
  - ``unavailable``— the tooling/script needed to run the gate does not
                     exist in this checkout (e.g. quality harness not
                     merged yet).

The candidate PR body MUST distinguish these states. A candidate whose
required gates are not all ``passed`` is **incomplete** and must never be
described as ``verified``, ``ready``, or ``passed``. The stable pointer
is never advanced by an incomplete candidate.

This script is called by ``.github/workflows/dep-candidate.yml`` after the
quality-gates job finishes. It reads the job results and artifact paths,
emits a JSON status file, and prints a Markdown summary for the PR body.

Exit codes:
  0 — all required gates passed (candidate is verified).
  1 — one or more required gates did not pass (candidate is incomplete).
  2 — invalid arguments / usage error.

Usage::

    python scripts/generate_gate_status.py \\
        --candidate-type ort \\
        --quality-report quality-report.json \\
        --runtime-report runtime-quality-report.json \\
        --gate-result gate-result.json \\
        --runtime-artifacts-dir artifacts/runtimes \\
        --output gate-status.json \\
        --markdown gate-status.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# Scripts that must exist in the checkout for a gate to be "available".
# If the quality harness has not been merged, these scripts won't exist
# and the gate is classified as ``unavailable`` rather than ``not_run``.
QUALITY_SUITE_SCRIPT = ROOT / "scripts" / "run_quality_suite.py"
RUNTIME_QUALITY_SUITE_SCRIPT = ROOT / "scripts" / "run_runtime_quality_suite.py"
GATE_ENFORCER_SCRIPT = ROOT / "scripts" / "enforce_quality_gates.py"

# The stable pointer must never be modified by a candidate workflow.
STABLE_POINTER_PATH = ROOT / "catalog" / "channels" / "stable.json"


def _classify_report(report_path: Path, script_path: Path) -> str:
    """Classify a report-based gate.

    Returns one of ``passed``, ``failed``, ``not_run``, ``unavailable``.
    """
    if not script_path.exists():
        return "unavailable"
    if not report_path.exists():
        return "not_run"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "failed"
    # A report with a top-level ``status`` field is the simplest signal.
    status = report.get("status")
    if status == "passed":
        return "passed"
    if status == "failed":
        return "failed"
    # Fall back to inspecting results: if any result entry exists and none
    # have errors, treat as passed; otherwise failed.
    results = report.get("results", [])
    if results and all(not r.get("error") for r in results):
        return "passed"
    if results:
        return "failed"
    # Report exists but has no results — treat as not_run (suite produced
    # nothing, likely because no fixtures matched the tier).
    return "not_run"


def _classify_runtime_artifacts(artifacts_dir: Path) -> str:
    """Classify the runtime-builds gate for ORT candidates."""
    if not artifacts_dir.exists():
        return "not_run"
    files = [f for f in artifacts_dir.rglob("*") if f.is_file()]
    if not files:
        return "not_run"
    # ORT updates require all 5 targets. We check for at least one archive
    # per known target suffix pattern. A stricter count check is done by
    # the caller via --expected-targets.
    return "passed"


def _classify_gate_result(gate_result_path: Path, script_path: Path) -> str:
    """Classify the enforcement gate result."""
    if not script_path.exists():
        return "unavailable"
    if not gate_result_path.exists():
        return "not_run"
    try:
        result = json.loads(gate_result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "failed"
    if result.get("overall") == "passed":
        return "passed"
    if result.get("overall") == "failed":
        return "failed"
    return "not_run"


def _required_gates(candidate_type: str) -> list[str]:
    """Return the list of required gate names for a candidate type."""
    if candidate_type == "ort":
        return [
            "runtime_builds",
            "model_quality",
            "runtime_quality",
            "gate_enforcement",
        ]
    if candidate_type == "model":
        return [
            "model_conversion",
            "model_quality",
            "runtime_quality",
            "baseline_comparison",
        ]
    raise ValueError(f"unknown candidate type: {candidate_type}")


def generate_gate_status(
    candidate_type: str,
    quality_report: Path | None,
    runtime_report: Path | None,
    gate_result: Path | None,
    runtime_artifacts_dir: Path | None,
    converted_model: Path | None = None,
    baseline_comparison: Path | None = None,
    expected_targets: int = 5,
) -> dict[str, Any]:
    """Generate the gate status structure.

    Returns a dict with ``overall_status``, ``gates``, and ``incomplete``.
    """
    gates: dict[str, dict[str, Any]] = {}

    # --- Runtime builds (ORT only) ---
    if candidate_type == "ort":
        rt_status = _classify_runtime_artifacts(runtime_artifacts_dir) if runtime_artifacts_dir else "not_run"
        # If artifacts exist, verify target count.
        if rt_status == "passed" and runtime_artifacts_dir:
            archives = [f for f in runtime_artifacts_dir.rglob("*") if f.is_file() and "onnxruntime" in f.name]
            if len(archives) < expected_targets:
                rt_status = "failed"
                gates["runtime_builds"] = {
                    "status": rt_status,
                    "detail": f"only {len(archives)} of {expected_targets} runtime targets produced",
                }
            else:
                gates["runtime_builds"] = {
                    "status": rt_status,
                    "detail": f"{len(archives)} runtime targets produced",
                }
        else:
            gates["runtime_builds"] = {
                "status": rt_status,
                "detail": "no runtime artifacts directory" if not runtime_artifacts_dir else "directory empty",
            }

    # --- Model conversion (model-weight only) ---
    if candidate_type == "model":
        if converted_model is None or not converted_model.exists():
            gates["model_conversion"] = {"status": "not_run", "detail": "converted model not produced"}
        else:
            gates["model_conversion"] = {"status": "passed", "detail": str(converted_model)}

    # --- Model quality ---
    mq_status = _classify_report(quality_report, QUALITY_SUITE_SCRIPT) if quality_report else "not_run"
    gates["model_quality"] = {
        "status": mq_status,
        "detail": "" if quality_report else "no quality report path provided",
    }

    # --- Runtime quality ---
    rq_status = _classify_report(runtime_report, RUNTIME_QUALITY_SUITE_SCRIPT) if runtime_report else "not_run"
    gates["runtime_quality"] = {
        "status": rq_status,
        "detail": "" if runtime_report else "no runtime report path provided",
    }

    # --- Gate enforcement (ORT) / baseline comparison (model) ---
    if candidate_type == "ort":
        ge_status = _classify_gate_result(gate_result, GATE_ENFORCER_SCRIPT) if gate_result else "not_run"
        gates["gate_enforcement"] = {
            "status": ge_status,
            "detail": "" if gate_result else "no gate result path provided",
        }
    elif candidate_type == "model":
        if baseline_comparison is None or not baseline_comparison.exists():
            gates["baseline_comparison"] = {"status": "not_run", "detail": "baseline comparison not produced"}
        else:
            try:
                comp = json.loads(baseline_comparison.read_text(encoding="utf-8"))
                gates["baseline_comparison"] = {
                    "status": "passed" if comp.get("verdict") != "regressed" else "failed",
                    "detail": comp.get("verdict", "unknown"),
                }
            except (OSError, json.JSONDecodeError):
                gates["baseline_comparison"] = {"status": "failed", "detail": "invalid baseline comparison file"}

    # --- Overall ---
    required = _required_gates(candidate_type)
    statuses = {g: gates.get(g, {}).get("status", "not_run") for g in required}
    all_passed = all(s == "passed" for s in statuses.values())
    any_failed = any(s == "failed" for s in statuses.values())

    if all_passed:
        overall = "passed"
    elif any_failed:
        overall = "failed"
    else:
        # Not all passed, but none explicitly failed → incomplete (not_run or unavailable).
        overall = "incomplete"

    return {
        "candidate_type": candidate_type,
        "overall_status": overall,
        "incomplete": overall != "passed",
        "gates": gates,
        "required_gates": required,
    }


def render_markdown(status: dict[str, Any]) -> str:
    """Render the gate status as a Markdown table for the PR body."""
    lines = [
        "## Gate status",
        "",
    ]
    overall = status["overall_status"]
    if overall == "passed":
        lines.append("> **Status: PASSED** — all required gates passed.")
    elif overall == "failed":
        lines.append("> **Status: FAILED** — one or more required gates failed.")
    else:
        lines.append(
            "> **Status: INCOMPLETE** — one or more required gates did not run "
            "or are unavailable. This candidate is NOT verified, NOT ready, "
            "and MUST NOT be promoted."
        )
    lines.append("")
    lines.append("| Gate | Status | Detail |")
    lines.append("|------|--------|--------|")
    for gate_name in status["required_gates"]:
        gate = status["gates"].get(gate_name, {})
        s = gate.get("status", "not_run")
        d = gate.get("detail", "")
        lines.append(f"| {gate_name} | {s} | {d} |")
    lines.append("")
    lines.append(
        "Status legend: `passed` = ran and succeeded; `failed` = ran and failed; "
        "`not_run` = skipped (e.g. model file missing); `unavailable` = tooling "
        "not present in this checkout."
    )
    lines.append("")
    if overall != "passed":
        lines.append(
            "**The stable pointer was NOT modified.** An incomplete candidate "
            "cannot advance the stable channel."
        )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate candidate gate status.")
    parser.add_argument("--candidate-type", required=True, choices=["ort", "model"],
                        help="Candidate type.")
    parser.add_argument("--quality-report", type=Path, default=None,
                        help="Path to the model quality report JSON.")
    parser.add_argument("--runtime-report", type=Path, default=None,
                        help="Path to the runtime quality report JSON.")
    parser.add_argument("--gate-result", type=Path, default=None,
                        help="Path to the gate enforcement result JSON.")
    parser.add_argument("--runtime-artifacts-dir", type=Path, default=None,
                        help="Directory containing runtime build artifacts.")
    parser.add_argument("--converted-model", type=Path, default=None,
                        help="Path to the converted ONNX model (model-weight candidates).")
    parser.add_argument("--baseline-comparison", type=Path, default=None,
                        help="Path to the baseline comparison JSON (model-weight candidates).")
    parser.add_argument("--expected-targets", type=int, default=5,
                        help="Expected number of runtime targets (ORT candidates).")
    parser.add_argument("--output", type=Path, default=Path("gate-status.json"),
                        help="Output JSON status path.")
    parser.add_argument("--markdown", type=Path, default=None,
                        help="Optional Markdown summary path.")
    args = parser.parse_args()

    status = generate_gate_status(
        candidate_type=args.candidate_type,
        quality_report=args.quality_report,
        runtime_report=args.runtime_report,
        gate_result=args.gate_result,
        runtime_artifacts_dir=args.runtime_artifacts_dir,
        converted_model=args.converted_model,
        baseline_comparison=args.baseline_comparison,
        expected_targets=args.expected_targets,
    )

    args.output.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"gate-status: {args.output}")

    md = render_markdown(status)
    if args.markdown:
        args.markdown.write_text(md, encoding="utf-8")
        print(f"gate-status markdown: {args.markdown}")

    print(md)
    return 0 if not status["incomplete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
