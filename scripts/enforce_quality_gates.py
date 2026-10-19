#!/usr/bin/env python3
"""Enforce quality gates against a candidate artifact's reports.

Given a candidate's quality report + runtime quality report + the baseline
for the artifact it replaces, checks that:
  - The required reports are present (not "pending-pr4-freeze").
  - No correctness metric regresses beyond its budget.
  - No runtime metric regresses beyond its budget.
  - Shape/NaN/Inf equality checks pass.
  - The Pareto requirement is satisfied: the candidate is not dominated by
    the baseline on all measured metrics (it must improve at least one
    metric within budget).

Fails (non-zero exit) if any gate fails. The publisher uses the shared frozen
baseline prerequisite. Binding immutable candidate reports to every catalog
model/runtime/provider edge remains required before release gates are complete.

Usage::

    python scripts/enforce_quality_gates.py \\
        --artifact-id htdemucs.spectral.fp32.onnx \\
        --quality-report quality-report.json \\
        --runtime-report runtime-quality-report.json \\
        --tier release
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASELINES_PATH = ROOT / "quality" / "baselines.json"
BUDGETS_PATH = ROOT / "quality" / "budgets.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate_quality(report: dict[str, Any]) -> dict[str, Any]:
    """Aggregate quality report results into a single metrics dict."""
    results = report.get("results", [])
    if not results:
        return {}
    # Use the worst-case (max) across fixtures for error metrics. Fixtures
    # carrying a per-fixture MSE budget (corpus manifest mse_budget, recorded
    # per result as mse_threshold by run_quality_suite) are gated against
    # their OWN budget in _per_fixture_budget_violations and excluded from
    # the flat-budget aggregate — otherwise a fixture whose calibrated budget
    # exceeds the tier budget (the impulse fixture measures 1.03e-4 on the
    # shipped stable artifact) would fail every healthy candidate.
    tier_default = report.get("mse_threshold")
    mse_values = [
        r["mse"]
        for r in results
        if r.get("mse") is not None
        and r.get("mse_threshold", tier_default) == tier_default
    ]
    mae_values = [r["mae"] for r in results if r.get("mae") is not None]
    max_abs_values = [r["max_abs_error"] for r in results if r.get("max_abs_error") is not None]
    return {
        "mse": max(mse_values) if mse_values else None,
        "mae": max(mae_values) if mae_values else None,
        "max_abs_error": max(max_abs_values) if max_abs_values else None,
        "shape_match": all(r.get("shape_match", False) for r in results),
        "onnx_has_nan": any(r.get("onnx_has_nan", False) for r in results),
        "onnx_has_inf": any(r.get("onnx_has_inf", False) for r in results),
    }


def _per_fixture_budget_violations(report: dict[str, Any]) -> list[str]:
    """Enforce per-fixture MSE budgets, aligned with run_quality_suite's gate."""
    errors: list[str] = []
    tier_default = report.get("mse_threshold")
    for r in report.get("results", []):
        mse = r.get("mse")
        threshold = r.get("mse_threshold", tier_default)
        if mse is None or threshold is None:
            continue
        if r.get("mse_threshold", tier_default) != tier_default and mse > threshold:
            errors.append(
                f"{r.get('fixture_id', '?')}: mse {mse} exceeds its "
                f"per-fixture budget {threshold}"
            )
    return errors


def _aggregate_runtime(report: dict[str, Any]) -> dict[str, Any]:
    """Aggregate runtime quality report results."""
    results = report.get("results", [])
    if not results:
        return {}
    return {
        "rtf_warm": max(r["rtf_warm"] for r in results),
        "cold_load_s": max(r["cold_load_s"] for r in results),
        "warm_median_s": max(r["warm_median_s"] for r in results),
        "peak_rss_kb": max(r["peak_rss_kb"] for r in results),
        "installed_size": report.get("runtime_archive", {}).get("installed_size"),
        "shape_match": all(not r.get("shape_errors") for r in results),
    }


def _check_budget(
    metric: str, candidate_value: float | None, baseline_value: float | None,
    budget: dict[str, Any], tier: str,
) -> str | None:
    """Check a single metric against its budget. Returns error string or None."""
    if budget["direction"] == "equality":
        required = budget.get("required_value")
        if type(candidate_value) is not type(required) or candidate_value != required:
            return f"{metric}: expected {required}, got {candidate_value}"
        return None
    if type(candidate_value) not in (int, float) or not math.isfinite(candidate_value):
        return f"{metric}: candidate value must be finite"
    if baseline_value is None:
        # No baseline — use absolute threshold.
        key = f"{tier}_max_absolute"
        if key in budget and candidate_value > budget[key]:
            return f"{metric}: {candidate_value} > {key}={budget[key]}"
        return None
    # Relative budget.
    factor_key = f"{tier}_max_factor"
    absolute_key = f"{tier}_max_absolute"
    if factor_key in budget:
        threshold = baseline_value * budget[factor_key]
        if candidate_value > threshold:
            return f"{metric}: {candidate_value} > baseline({baseline_value}) * {budget[factor_key]} = {threshold}"
    elif absolute_key in budget:
        if candidate_value > budget[absolute_key]:
            return f"{metric}: {candidate_value} > {absolute_key}={budget[absolute_key]}"
    return None


def _check_pareto(
    candidate: dict[str, Any], baseline: dict[str, Any], budgets: list[dict[str, Any]],
) -> str | None:
    """Check that the candidate is not Pareto-dominated by the baseline.

    The candidate must improve at least one metric (be strictly better than
    the baseline on at least one lower-is-better metric, or equal on all
    and better on one). If the candidate is worse on every measured metric,
    it is dominated and fails the Pareto requirement.
    """
    if not baseline:
        return None  # No baseline — no Pareto check.
    budget_map = {b["metric"]: b for b in budgets}
    improvements = 0
    regressions = 0
    for metric, budget in budget_map.items():
        if metric not in candidate or metric not in baseline:
            continue
        if candidate[metric] is None or baseline[metric] is None:
            continue
        if budget["direction"] == "lower-is-better":
            if candidate[metric] < baseline[metric]:
                improvements += 1
            elif candidate[metric] > baseline[metric]:
                regressions += 1
        elif budget["direction"] == "higher-is-better":
            if candidate[metric] > baseline[metric]:
                improvements += 1
            elif candidate[metric] < baseline[metric]:
                regressions += 1
    if regressions > 0 and improvements == 0:
        return f"Pareto: candidate is dominated by baseline (regressed {regressions} metric(s), improved 0)"
    return None


def _is_baseline_frozen(entry: dict[str, Any]) -> bool:
    """Require all declared freeze fields, including measured metric values."""
    def populated(value: Any) -> bool:
        if isinstance(value, str):
            value = value.strip().lower()
            return bool(value) and not value.startswith("pending") and value not in {
                "not_run", "unavailable",
            }
        if isinstance(value, dict):
            return bool(value) and all(populated(v) for v in value.values())
        if isinstance(value, list):
            return bool(value) and all(populated(v) for v in value)
        return isinstance(value, (int, float)) and math.isfinite(value)

    required = _load_json(BASELINES_PATH)["required_fields_for_freeze"]
    if not all(populated(entry.get(field)) for field in required):
        return False
    if any(
        not isinstance(entry[field], str)
        for field in required
        if field not in {"artifact_size_bytes", "baseline_metrics", "benchmark_environment"}
    ):
        return False
    for field in ("model_digest", "baseline_quality_report_digest", "baseline_runtime_report_digest"):
        if not isinstance(entry[field], str) or not re.fullmatch(r"[0-9a-f]{64}", entry[field]):
            return False
    if type(entry["artifact_size_bytes"]) is not int or entry["artifact_size_bytes"] <= 0:
        return False
    for field in ("frozen_at", "generation_time"):
        try:
            timestamp = datetime.datetime.fromisoformat(entry[field].replace("Z", "+00:00"))
        except (ValueError, TypeError, AttributeError):
            return False
        if timestamp.tzinfo is None:
            return False
    metrics = entry["baseline_metrics"]
    if not isinstance(metrics, dict):
        return False
    for budget in _load_json(BUDGETS_PATH)["budgets"]:
        value = metrics.get(budget["metric"])
        if budget["direction"] == "equality":
            if type(value) is not bool or value != budget["required_value"]:
                return False
        elif type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            return False
    return True


def frozen_baseline_errors(artifact_ids: list[str]) -> list[str]:
    """Stable publication requires one complete baseline per shipping model."""
    entries = _load_json(BASELINES_PATH)["stable_baselines"]
    errors = []
    for artifact_id in artifact_ids:
        matches = [entry for entry in entries if entry["artifact_id"] == artifact_id]
        if len(matches) != 1 or not _is_baseline_frozen(matches[0]):
            errors.append(f"baseline for {artifact_id} is missing, duplicate or NOT FROZEN")
    return errors


def enforce_gates(
    artifact_id: str, quality_report: dict[str, Any] | None,
    runtime_report: dict[str, Any] | None, tier: str,
) -> list[str]:
    errors: list[str] = []
    baselines = _load_json(BASELINES_PATH)
    budgets_doc = _load_json(BUDGETS_PATH)
    budgets = budgets_doc["budgets"]

    # Find baseline for this artifact.
    baseline_entry = next(
        (b for b in baselines["stable_baselines"] if b["artifact_id"] == artifact_id),
        None,
    )
    if baseline_entry is None:
        errors.append(f"no baseline found for artifact_id={artifact_id}")
        return errors

    frozen = _is_baseline_frozen(baseline_entry)
    if not frozen and tier == "release":
        # Release tier requires a frozen baseline. Without one, release gates
        # are NOT landed and no artifact can be promoted to the stable channel.
        errors.append(
            f"baseline for {artifact_id} is NOT FROZEN — release-tier gates "
            f"cannot pass without a real measured baseline. Run the quality "
            f"and runtime suites on the stable artifact and freeze the baseline."
        )
        return errors

    if not frozen:
        # PR tier: use absolute thresholds only (no baseline comparison).
        baseline_metrics: dict[str, Any] = {}
    else:
        baseline_metrics = baseline_entry.get("baseline_metrics", {})

    from validate_quality_report import validate_report as validate_quality
    from validate_runtime_quality_report import validate_report as validate_runtime

    for label, report, validate in (
        ("quality", quality_report, validate_quality),
        ("runtime", runtime_report, validate_runtime),
    ):
        if report is not None:
            errors.extend(f"{label}: {error}" for error in validate(report))
            if tier == "release" and report.get("tier") != "release":
                errors.append(f"{label}: release-tier report required")
        elif tier == "release":
            errors.append(f"{label}: release-tier report required")
    if errors:
        return errors

    # Aggregate candidate metrics.
    candidate: dict[str, Any] = {}
    if quality_report:
        candidate.update(_aggregate_quality(quality_report))
        errors.extend(_per_fixture_budget_violations(quality_report))
    if runtime_report:
        candidate.update(_aggregate_runtime(runtime_report))

    if not candidate:
        errors.append("no candidate metrics: provide --quality-report and/or --runtime-report")
        return errors

    # Check each budget.
    for budget in budgets:
        metric = budget["metric"]
        if metric not in candidate:
            if tier == "release":
                errors.append(f"{metric}: missing release metric")
            continue
        err = _check_budget(
            metric, candidate[metric], baseline_metrics.get(metric), budget, tier,
        )
        if err:
            errors.append(err)

    # Pareto check (release tier only).
    if tier == "release" and baseline_metrics:
        pareto_err = _check_pareto(candidate, baseline_metrics, budgets)
        if pareto_err:
            errors.append(pareto_err)

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Enforce quality gates for a candidate artifact.")
    parser.add_argument("--artifact-id", required=True, help="Candidate artifact ID.")
    parser.add_argument("--quality-report", type=Path, default=None)
    parser.add_argument("--runtime-report", type=Path, default=None)
    parser.add_argument("--tier", choices=["pr", "release"], default="release")
    args = parser.parse_args()

    quality = _load_json(args.quality_report) if args.quality_report and args.quality_report.is_file() else None
    runtime = _load_json(args.runtime_report) if args.runtime_report and args.runtime_report.is_file() else None

    if not quality and not runtime:
        print("ERROR: provide at least one of --quality-report or --runtime-report", file=sys.stderr)
        return 1

    errors = enforce_gates(args.artifact_id, quality, runtime, args.tier)
    if errors:
        print(f"FAIL: {len(errors)} gate error(s) for {args.artifact_id}:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 2

    print(f"OK: quality gates passed for {args.artifact_id} (tier={args.tier})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
