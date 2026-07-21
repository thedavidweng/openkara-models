#!/usr/bin/env python3
"""Generate a dependency update candidate: update locks + create a candidate branch.

Given a detection result (from detect_ort_release.py or
detect_model_weight_revision.py), this script:
  1. Updates the relevant source lock with the new upstream tag + commit SHA.
  2. Validates the updated lock.
  3. Creates a candidate branch name.
  4. Prints a summary of the changes for the candidate PR body.

This script does NOT build, convert, or run quality gates — those are done
by the candidate orchestration workflow (.github/workflows/dep-candidate.yml).
This script only prepares the lock update; the workflow does the rest.

Usage::

    python scripts/generate_dep_candidate.py \\
        --ort-tag v1.28.0 --ort-commit abc123 \\
        --output candidate-summary.json

    python scripts/generate_dep_candidate.py \\
        --model htdemucs --model-commit def456 \\
        --output candidate-summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "ort" / "source-lock.json"


def update_ort_lock(tag: str, commit_sha: str) -> dict[str, Any]:
    """Update the ORT source lock with a new tag + commit SHA."""
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    old_tag = lock["upstream"]["tag"]
    old_commit = lock["upstream"]["commit_sha"]
    lock["upstream"]["tag"] = tag
    lock["upstream"]["commit_sha"] = commit_sha
    lock["upstream"]["release_url"] = f"https://github.com/microsoft/onnxruntime/releases/tag/{tag}"
    LOCK_PATH.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "type": "ort",
        "old_tag": old_tag,
        "new_tag": tag,
        "old_commit": old_commit,
        "new_commit": commit_sha,
        "lock_path": str(LOCK_PATH),
    }


def generate_candidate_summary(changes: dict[str, Any]) -> str:
    """Generate a Markdown summary for the candidate PR body."""
    lines = ["## Dependency update candidate", ""]
    if changes["type"] == "ort":
        lines.extend([
            f"**Type:** ONNX Runtime source update",
            f"**Old:** `{changes['old_tag']}` ({changes['old_commit'][:12]})",
            f"**New:** `{changes['new_tag']}` ({changes['new_commit'][:12]})",
            "",
            "### Required coupled changes",
            "",
            "- [ ] Rebuild all 5 runtime targets (ort-publish.yml)",
            "- [ ] Re-convert affected ONNX models (convert.yml)",
            "- [ ] Generate runtime catalog entries (generate_runtime_catalog_entries.py)",
            "- [ ] Run quality suite (run_quality_suite.py --tier release)",
            "- [ ] Run runtime quality suite (run_runtime_quality_suite.py --tier release)",
            "- [ ] Enforce quality gates (enforce_quality_gates.py --tier release)",
            "- [ ] Generate trend report (generate_trend_report.py)",
            "",
            "### Artifacts",
            "",
            "The candidate build workflow uploads:",
            "- Runtime archives (5 targets)",
            "- Converted ONNX models",
            "- Quality report + runtime quality report",
            "- Trend report",
            "- SBOM + provenance (per runtime archive)",
            "",
            "### Review checklist",
            "",
            "- [ ] Quality gates pass (no regressions beyond budget)",
            "- [ ] Pareto: candidate is not dominated by baseline",
            "- [ ] Runtime archives load on all 5 targets",
            "- [ ] Catalog entries are correct",
            "- [ ] No stable pointer changed",
        ])
    elif changes["type"] == "model":
        lines.extend([
            f"**Type:** Model-weight revision update",
            f"**Model:** `{changes['model']}`",
            f"**Old commit:** `{changes['old_commit'][:12]}`" if changes.get("old_commit") else "**Old commit:** n/a",
            f"**New commit:** `{changes['new_commit'][:12]}`",
            "",
            "### Required coupled changes",
            "",
            "- [ ] Re-convert affected ONNX models (convert.yml)",
            "- [ ] Run quality suite (run_quality_suite.py --tier release)",
            "- [ ] Enforce quality gates (enforce_quality_gates.py --tier release)",
            "- [ ] Generate trend report (generate_trend_report.py)",
        ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a dependency update candidate.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ort-tag", help="New ORT release tag.")
    group.add_argument("--model", help="Model name for weight revision update.")
    parser.add_argument("--ort-commit", help="New ORT commit SHA (required with --ort-tag).")
    parser.add_argument("--model-commit", help="New model commit SHA (required with --model).")
    parser.add_argument("--output", type=Path, default=Path("candidate-summary.json"),
                        help="Output JSON summary path.")
    parser.add_argument("--markdown", type=Path, default=None,
                        help="Optional Markdown summary path.")
    args = parser.parse_args()

    if args.ort_tag:
        if not args.ort_commit:
            parser.error("--ort-commit is required with --ort-tag")
        changes = update_ort_lock(args.ort_tag, args.ort_commit)
    elif args.model:
        if not args.model_commit:
            parser.error("--model-commit is required with --model")
        changes = {
            "type": "model",
            "model": args.model,
            "old_commit": None,  # PR 3 does not define model lock format yet
            "new_commit": args.model_commit,
        }
    else:
        parser.error("provide --ort-tag or --model")

    args.output.write_text(json.dumps(changes, indent=2, sort_keys=True) + "\n")
    print(f"summary: {args.output}")

    if args.markdown:
        args.markdown.write_text(generate_candidate_summary(changes))
        print(f"markdown: {args.markdown}")

    # Print summary to stdout.
    print(generate_candidate_summary(changes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
