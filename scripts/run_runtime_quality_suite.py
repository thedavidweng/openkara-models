#!/usr/bin/env python3
"""Run the runtime benchmark harness against the corpus manifest.

For each fixture in the corpus manifest (filtered by --tier), loads the ONNX
model via the built ORT shared library and runs inference. Records the full
runtime metric set:

  - session cold-load time
  - first-window latency
  - warm median / p95 latency
  - end-to-end real-time factor (RTF = processing_time / audio_duration)
  - peak RSS
  - provider assignment + fallback node count
  - output shape + finiteness
  - compressed download bytes + installed bytes (from the runtime archive)

Emits a JSON report conforming to quality/runtime-quality-report-v1.json
plus a concise Markdown summary.

Fails (non-zero exit) if:
  - the model cannot be loaded;
  - output contains NaN or Inf;
  - output shape does not match expected_output_shape;
  - RTF exceeds the tier budget (pr_max=10.0, release_max=5.0 — initial
    budgets; PR 4 freezes empirical baselines).

Usage::

    python scripts/run_runtime_quality_suite.py \\
        --runtime ort/packages/<archive> \\
        --onnx models/htdemucs.onnx \\
        --tier pr \\
        --report report.json --summary summary.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import archive_utils  # noqa: E402

CORPUS_MANIFEST = ROOT / "quality" / "corpus-manifest.json"
REPORT_SCHEMA = ROOT / "quality" / "runtime-quality-report-v1.json"
GENERATOR_VERSION = "openkara.runtime-quality-report/v1"

# Initial RTF budgets: PR tier is lenient, release tier is strict.
# PR 4 freezes empirical baselines and may tighten these.
RTF_PR_MAX = 10.0
RTF_RELEASE_MAX = 5.0


def _sha256_file(path: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return size, h.hexdigest()


def _load_corpus(tier: str | None) -> list[dict[str, Any]]:
    manifest = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    import jsonschema
    jsonschema.validate(manifest, json.loads((ROOT / "quality/corpus-manifest-v1.json").read_text()))
    fixtures = manifest["fixtures"]
    if tier and tier != "release":
        fixtures = [f for f in fixtures if f["tier"] == tier]
    return fixtures


def _installed_size(archive: Path) -> int:
    """Sum of extracted file bytes."""
    return archive_utils.installed_size(archive)


def run_fixture(native, fixture: dict[str, Any], onnx_path: str,
                warmup: int, iters: int, corpus_dir: Path | None = None) -> dict[str, Any]:
    from corpus_audio import load_fixture
    inp, _ = load_fixture(fixture, corpus_dir)
    output, metrics = native.measure(Path(onnx_path), inp, fixture["sample_rate"], warmup, iters)
    expected = fixture.get("expected_output_shape")
    if expected is not None and list(output.shape) != expected:
        metrics["shape_errors"].append("native waveform output differs from expected shape")
    return metrics | {"fixture_id": fixture["fixture_id"], "category": fixture["category"], "tier": fixture["tier"]}


def _markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        f"# Runtime quality report — {report['tier']} tier",
        "",
        f"**Runtime:** `{report['runtime_archive']['name']}`",
        f"**Model:** `{report['onnx_path']}`",
        f"**Fixtures:** {len(report['results'])}",
        f"**RTF budget:** {report['rtf_threshold']}",
        "",
        "| Fixture | Category | Cold load (s) | First window (s) | RTF (warm) | Peak RSS (KB) | Errors |",
        "|---------|----------|---------------|------------------|------------|---------------|--------|",
    ]
    for r in report["results"]:
        errors = "YES" if r["shape_errors"] else "no"
        lines.append(
            f"| {r['fixture_id']} | {r['category']} | {r['cold_load_s']:.3f} | "
            f"{r['first_window_s']:.3f} | {r['rtf_warm']:.2f} | "
            f"{r['peak_rss_kb']} | {errors} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the runtime quality suite.")
    parser.add_argument("--runtime", required=True, type=Path, help="Runtime archive.")
    parser.add_argument("--onnx", required=True, type=Path, help="ONNX model path.")
    parser.add_argument("--target", default=None, help="Target triple (default: native host).")
    parser.add_argument("--provider", choices=["cpu", "coreml", "xnnpack", "directml"], default="cpu")
    parser.add_argument("--ort-source", type=Path, help="Optional source header cross-check; uses packaged headers.")
    parser.add_argument("--corpus-dir", type=Path)
    parser.add_argument("--tier", choices=["pr", "release"], default="pr")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--report", type=Path, default=Path("runtime-quality-report.json"))
    parser.add_argument("--summary", type=Path, default=None, help="Optional Markdown summary path.")
    args = parser.parse_args()

    if not args.runtime.is_file():
        print(f"ERROR: runtime archive not found: {args.runtime}", file=sys.stderr)
        return 1
    if not args.onnx.is_file():
        print(f"ERROR: ONNX model not found: {args.onnx}", file=sys.stderr)
        return 1
    # Validate iteration counts up front so empty-iterable median errors and
    # divide-by-zero RTF errors never surface as unhandled exceptions.
    if args.warmup < 0:
        print(f"ERROR: --warmup must be >= 0 (got {args.warmup})", file=sys.stderr)
        return 1
    if args.iters < 1:
        print(f"ERROR: --iters must be >= 1 (got {args.iters}); "
              f"at least one measured iteration is required for median/p95",
              file=sys.stderr)
        return 1

    fixtures = _load_corpus(args.tier)
    if not fixtures:
        print(f"ERROR: no {args.tier} fixtures in corpus manifest", file=sys.stderr)
        return 1

    archive_size, archive_sha = _sha256_file(args.runtime)
    installed = _installed_size(args.runtime)
    rtf_threshold = RTF_PR_MAX if args.tier == "pr" else RTF_RELEASE_MAX

    with tempfile.TemporaryDirectory() as tmpd:
        from native_quality import NativeRuntime
        from run_runtime_benchmarks import _detect_target
        native = NativeRuntime(args.runtime, args.ort_source, args.target or _detect_target(), args.provider, Path(tmpd))
        print(f"Running {len(fixtures)} {args.tier}-tier fixture(s)...")
        results: list[dict[str, Any]] = []
        for fixture in fixtures:
            print(f"  {fixture['fixture_id']}...", end=" ", flush=True)
            r = run_fixture(native, fixture, str(args.onnx), args.warmup, args.iters, args.corpus_dir)
            results.append(r)
            print(f"rtf_warm={r['rtf_warm']:.2f}")

    model_size, model_sha = _sha256_file(args.onnx)
    report = {
        "schema_version": GENERATOR_VERSION,
        "measurement_backend": "native-c-api",
        "measurement_scope": "native-per-window-warm; end-to-end-first-including-process-io-dsp",
        "target": args.target or _detect_target(),
        "provider": args.provider,
        "model_digest": {"sha256": model_sha, "size": model_size},
        "runtime_archive": {
            "name": args.runtime.name,
            "sha256": archive_sha,
            "size": archive_size,
            "installed_size": installed,
        },
        "onnx_path": str(args.onnx),
        "tier": args.tier,
        "rtf_threshold": rtf_threshold,
        "results": results,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"\nreport: {args.report}")

    if args.summary:
        args.summary.write_text(_markdown_summary(report))
        print(f"summary: {args.summary}")

    # Gate.
    failed = False
    for r in results:
        if r["shape_errors"]:
            print(f"FAIL: {r['fixture_id']}: shape errors", file=sys.stderr)
            failed = True
        if r["rtf_warm"] > rtf_threshold:
            print(f"FAIL: {r['fixture_id']}: RTF {r['rtf_warm']} > {rtf_threshold}", file=sys.stderr)
            failed = True
    if failed:
        return 2
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
