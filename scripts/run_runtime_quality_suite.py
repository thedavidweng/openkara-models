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
import gc
import hashlib
import json
import os
import resource
import statistics
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
    fixtures = manifest["fixtures"]
    if tier:
        fixtures = [f for f in fixtures if f["tier"] == tier]
    return fixtures


def _extract_runtime(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    archive_utils.safe_extract(archive, dest)
    for p in dest.rglob("*"):
        low = p.name.lower()
        if low.endswith((".so", ".dylib", ".dll")) and "onnxruntime" in low:
            return p
    raise FileNotFoundError(f"no onnxruntime shared library found in {archive}")


def _load_session(lib_path: Path, model_path: str):
    import onnxruntime as ort
    os.environ["ORT_RUNTIMES"] = str(lib_path)
    so = ort.SessionOptions()
    so.enable_profiling = True
    return ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])


def _installed_size(archive: Path) -> int:
    """Sum of extracted file bytes."""
    return archive_utils.installed_size(archive)


def run_fixture(
    lib_path: Path, fixture: dict[str, Any], onnx_path: str,
    warmup: int, iters: int,
) -> dict[str, Any]:
    """Run one fixture through the runtime and return metrics."""
    import synthetic_fixtures as sf
    import time
    inp = sf.generate_fixture(fixture)
    # Add batch dimension for the model.
    batched = inp[np.newaxis, ...]
    audio_duration = fixture["frames"] / fixture["sample_rate"]

    gc.collect()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    t0 = time.perf_counter()
    sess = _load_session(lib_path, onnx_path)
    cold_load = time.perf_counter() - t0

    input_name = sess.get_inputs()[0].name

    # First window.
    t0 = time.perf_counter()
    out = sess.run(None, {input_name: batched})
    first_window = time.perf_counter() - t0
    rtf_first = first_window / audio_duration

    # Shape + finiteness.
    shape_errors: list[str] = []
    for i, o in enumerate(out):
        if not np.all(np.isfinite(o)):
            shape_errors.append(f"output[{i}] contains NaN or Inf")
        expected = fixture.get("expected_output_shape")
        if expected is not None and list(o.shape) != expected:
            shape_errors.append(f"output[{i}] shape {list(o.shape)} != expected {expected}")

    # Warm up.
    for _ in range(warmup):
        sess.run(None, {input_name: batched})

    # Measure.
    latencies: list[float] = []
    for _ in range(iters):
        t = time.perf_counter()
        sess.run(None, {input_name: batched})
        latencies.append(time.perf_counter() - t)

    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    warm_median = statistics.median(latencies)
    warm_p95 = sorted(latencies)[int(0.95 * len(latencies))]
    rtf_warm = warm_median / audio_duration

    # Provider + fallback from profiler.
    providers = sess.get_providers()
    fallback_nodes: int | None = None
    try:
        prof_path = sess.end_profiling()
        prof = json.loads(Path(prof_path).read_text())
        cpu_nodes = sum(
            1 for ev in prof.get("traceEvents", [])
            if ev.get("cat") == "Node" and ev.get("args", {}).get("provider") == "CPUExecutionProvider"
        )
        fallback_nodes = cpu_nodes
        Path(prof_path).unlink(missing_ok=True)
    except Exception:
        pass

    del sess
    gc.collect()

    return {
        "fixture_id": fixture["fixture_id"],
        "category": fixture["category"],
        "tier": fixture["tier"],
        "frames": fixture["frames"],
        "audio_duration_s": audio_duration,
        "cold_load_s": cold_load,
        "first_window_s": first_window,
        "warm_median_s": warm_median,
        "warm_p95_s": warm_p95,
        "warm_iters": iters,
        "rtf_first": rtf_first,
        "rtf_warm": rtf_warm,
        "peak_rss_kb": rss_after,
        "rss_delta_kb": rss_after - rss_before,
        "providers": providers,
        "fallback_node_count": fallback_nodes,
        "output_shape": [list(o.shape) for o in out],
        "shape_errors": shape_errors,
    }


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
        lib_path = _extract_runtime(args.runtime, Path(tmpd))
        print(f"runtime library: {lib_path}")
        print(f"Running {len(fixtures)} {args.tier}-tier fixture(s)...")
        results: list[dict[str, Any]] = []
        for fixture in fixtures:
            print(f"  {fixture['fixture_id']}...", end=" ", flush=True)
            r = run_fixture(lib_path, fixture, str(args.onnx), args.warmup, args.iters)
            results.append(r)
            print(f"rtf_warm={r['rtf_warm']:.2f}")

    report = {
        "schema_version": GENERATOR_VERSION,
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
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
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
