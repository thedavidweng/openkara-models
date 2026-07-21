#!/usr/bin/env python3
"""Generate a trend report comparing a candidate's reports against the baseline.

Given a candidate's quality report + runtime quality report and the baseline
metrics for the artifact it replaces, produces a Markdown + JSON trend report
showing the delta for every metric. This is displayed on candidate PRs so
reviewers can see at a glance whether the candidate regresses or improves.

The JSON trend report is stored as a release asset (not committed to the
repo) to avoid growing the Git history with per-PR reports.

Usage::

    python scripts/generate_trend_report.py \\
        --artifact-id htdemucs.balanced.fp32.onnx \\
        --quality-report quality-report.json \\
        --runtime-report runtime-quality-report.json \\
        --output trend.json --markdown trend.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
BASELINES_PATH = ROOT / "quality" / "baselines.json"
BUDGETS_PATH = ROOT / "quality" / "budgets.json"

import enforce_quality_gates as gates


def _delta(candidate: float | None, baseline: float | None) -> dict[str, Any]:
    if candidate is None or baseline is None:
        return {"candidate": candidate, "baseline": baseline, "delta": None,
                "delta_pct": None, "direction": "unknown"}
    delta = candidate - baseline
    pct = (delta / baseline * 100) if baseline != 0 else None
    return {"candidate": candidate, "baseline": baseline, "delta": delta,
            "delta_pct": pct, "improved": delta < 0}  # lower-is-better assumption


def generate_trend(
    artifact_id: str, quality_report: dict[str, Any] | None,
    runtime_report: dict[str, Any] | None,
) -> dict[str, Any]:
    baselines = json.loads(BASELINES_PATH.read_text(encoding="utf-8"))
    budgets = json.loads(BUDGETS_PATH.read_text(encoding="utf-8"))
    baseline_entry = next(
        (b for b in baselines["stable_baselines"] if b["artifact_id"] == artifact_id),
        None,
    )
    if baseline_entry is None:
        return {"error": f"no baseline for {artifact_id}"}
    baseline_metrics = baseline_entry.get("baseline_metrics", {})
    candidate: dict[str, Any] = {}
    if quality_report:
        candidate.update(gates._aggregate_quality(quality_report))
    if runtime_report:
        candidate.update(gates._aggregate_runtime(runtime_report))

    budget_map = {b["metric"]: b for b in budgets["budgets"]}
    trends: list[dict[str, Any]] = []
    for metric, budget in budget_map.items():
        if metric not in candidate:
            continue
        cval = candidate[metric]
        bval = baseline_metrics.get(metric)
        if budget["direction"] == "equality":
            trends.append({
                "metric": metric, "category": budget["category"],
                "direction": "equality", "candidate": cval, "baseline": bval,
                "delta": None, "status": "pass" if cval == budget.get("required_value") else "fail",
            })
        else:
            d = _delta(cval, bval)
            status = "improved" if d.get("improved") else ("regressed" if d.get("delta", 0) > 0 else "unchanged")
            trends.append({
                "metric": metric, "category": budget["category"],
                "direction": budget["direction"],
                "candidate": cval, "baseline": bval,
                "delta": d["delta"], "delta_pct": d["delta_pct"],
                "status": status,
            })
    return {
        "schema_version": "openkara.trend-report/v1",
        "artifact_id": artifact_id,
        "baseline_report_id": baseline_entry.get("baseline_quality_report_id"),
        "trends": trends,
        "gate_errors": gates.enforce_gates(artifact_id, quality_report, runtime_report, "release"),
    }


def _markdown(trend: dict[str, Any]) -> str:
    if "error" in trend:
        return f"# Trend report\n\nERROR: {trend['error']}\n"
    lines = [
        f"# Trend report — {trend['artifact_id']}",
        "",
        f"**Baseline report:** `{trend.get('baseline_report_id', 'n/a')}`",
        "",
        "| Metric | Category | Candidate | Baseline | Delta | Status |",
        "|--------|----------|-----------|----------|-------|--------|",
    ]
    for t in trend["trends"]:
        delta = f"{t['delta']:.4f}" if t.get("delta") is not None else "n/a"
        cand = f"{t['candidate']:.4f}" if isinstance(t.get("candidate"), float) else str(t.get("candidate"))
        base = f"{t['baseline']:.4f}" if isinstance(t.get("baseline"), float) else str(t.get("baseline"))
        lines.append(f"| {t['metric']} | {t['category']} | {cand} | {base} | {delta} | {t['status']} |")
    if trend.get("gate_errors"):
        lines.append("")
        lines.append("## Gate errors")
        for e in trend["gate_errors"]:
            lines.append(f"- {e}")
    else:
        lines.append("")
        lines.append("All gates passed.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a trend report.")
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--quality-report", type=Path, default=None)
    parser.add_argument("--runtime-report", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("trend.json"))
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()

    quality = json.loads(args.quality_report.read_text()) if args.quality_report and args.quality_report.is_file() else None
    runtime = json.loads(args.runtime_report.read_text()) if args.runtime_report and args.runtime_report.is_file() else None

    trend = generate_trend(args.artifact_id, quality, runtime)
    args.output.write_text(json.dumps(trend, indent=2, sort_keys=True) + "\n")
    print(f"trend: {args.output}")
    if args.markdown:
        args.markdown.write_text(_markdown(trend))
        print(f"markdown: {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
