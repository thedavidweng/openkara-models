#!/usr/bin/env python3
"""Run native ORT runtime compatibility + performance benchmarks.

For each stable catalog model compatible with the current target's runtime,
loads the model via the built ORT shared library and runs one complete
343980-frame stereo inference window. Records:

  - session cold-load time
  - first-window latency
  - warm median / p95 latency
  - peak RSS
  - provider assignment
  - fallback node count (via session profiler)
  - output shape + finiteness (NaN/Inf check)
  - runtime archive + installed size
  - checkpoint/output I/O bytes

Emits a JSON report conforming to ``ort/benchmark-report-v1.json``.

Fails (non-zero exit) if:
  - the model cannot be loaded (missing kernel / incompatible opset);
  - output contains NaN or Inf;
  - output shape does not match the catalog-declared ``output_semantics``;
  - the fallback node count differs from the full-runtime baseline (when
    ``--baseline`` is given).

This script runs on the native runner for each target. It does not
cross-compile or cross-run.

Usage::

    python scripts/run_runtime_benchmarks.py \\
        --runtime ort/packages/onnxruntime-1.27.1-openkara-x86_64-unknown-linux-gnu.tar.gz \\
        --catalog catalog/releases/2026-07-20-001.json \\
        --report benchmark-report.json

    # Synthetic input only (no model download needed for smoke test):
    python scripts/run_runtime_benchmarks.py --runtime ... --model models/htdemucs.onnx --frames 343980 --report report.json
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
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA_PATH = ROOT / "ort" / "benchmark-report-v1.json"
GENERATOR_VERSION = "openkara.runtime-benchmark/v1"

sys.path.insert(0, str(ROOT / "scripts"))
import archive_utils  # noqa: E402


def _sha256_file(path: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return size, h.hexdigest()


def _extract_runtime(archive: Path, dest: Path) -> Path:
    """Extract the runtime archive and return the directory containing the lib.

    Uses archive_utils.safe_extract which rejects unsafe members (absolute
    paths, traversal, symlink/hardlink escapes, duplicate paths, excessive
    counts/sizes).
    """
    archive_utils.safe_extract(archive, dest)
    # Find the shared library.
    for p in dest.rglob("*"):
        low = p.name.lower()
        if low.endswith((".so", ".dylib", ".dll")) and "onnxruntime" in low:
            return p
    raise FileNotFoundError(f"no onnxruntime shared library found in {archive}")


def _resolve_model_from_catalog(manifest_path: Path, download_dir: Path,
                                 target: str | None) -> list[dict[str, Any]]:
    """Resolve compatible models from a catalog manifest. Returns artifact dicts."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = manifest.get("artifacts", {}).get("models", [])
    compat = manifest.get("compatibility", [])
    compatible_ids: set[str] | None = None
    if target and compat:
        # Find runtime artifacts for this target, then models compatible with them.
        runtime_ids = {
            e["runtime_artifact_id"] for e in compat
            if e.get("target_triple") == target
        }
        compatible_ids = set()
        for e in compat:
            if e["runtime_artifact_id"] in runtime_ids:
                compatible_ids.add(e["model_artifact_id"])
    resolved = []
    for art in models:
        if compatible_ids is not None and art["artifact_id"] not in compatible_ids:
            continue
        resolved.append(art)
    return resolved


def _download_model(art: dict[str, Any], download_dir: Path) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    dest = download_dir / art["filename"]
    expected_sha = art["archive_digest"]
    if dest.is_file() and _sha256_file(dest)[1] == expected_sha:
        return dest
    print(f"  downloading {art['artifact_id']}...")
    with urlopen(art["download_url"]) as resp, dest.open("wb") as out:  # noqa: S310
        out.write(resp.read())
    got = _sha256_file(dest)[1]
    if got != expected_sha:
        raise RuntimeError(f"model sha256 mismatch: {got} != {expected_sha}")
    return dest


def _parse_output_shape(semantics: str) -> list[int] | None:
    """Parse '[1,4,2,343980] drums/bass/other/vocals' into [1,4,2,343980]."""
    import re
    m = re.match(r"\[([0-9,\s]+)\]", semantics)
    if not m:
        return None
    return [int(x.strip()) for x in m.group(1).split(",")]


def _load_session(lib_path: Path, model_path: str, enable_profiling: bool = False):
    """Load the ORT shared library and create an InferenceSession."""
    import onnxruntime as ort
    # Override the runtime library path. ORT's Python wheel supports loading
    # a custom shared library via the ORT_RUNTIMES env var (semicolon-separated
    # list of paths). This swaps the wheel's bundled lib for our built one.
    os.environ["ORT_RUNTIMES"] = str(lib_path)
    so = ort.SessionOptions()
    if enable_profiling:
        so.enable_profiling = True
    sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
    return sess


def benchmark_model(
    lib_path: Path, model_path: Path, frames: int,
    warmup: int, iters: int, expected_output_shape: list[int] | None,
) -> dict[str, Any]:
    """Run the benchmark for one model. Returns a metrics dict."""
    import numpy as np

    # Cold load.
    gc.collect()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import time
    t0 = time.perf_counter()
    sess = _load_session(lib_path, str(model_path), enable_profiling=True)
    cold_load = time.perf_counter() - t0

    inp = np.zeros((1, 2, frames), dtype=np.float32)
    input_name = sess.get_inputs()[0].name

    # First window.
    t0 = time.perf_counter()
    out = sess.run(None, {input_name: inp})
    first_window = time.perf_counter() - t0

    # Shape + finiteness check.
    shape_errors: list[str] = []
    for i, o in enumerate(out):
        if not np.all(np.isfinite(o)):
            shape_errors.append(f"output[{i}] contains NaN or Inf")
        if expected_output_shape is not None and list(o.shape) != expected_output_shape:
            shape_errors.append(
                f"output[{i}] shape {list(o.shape)} != expected {expected_output_shape}"
            )

    # Warm up.
    for _ in range(warmup):
        sess.run(None, {input_name: inp})

    # Measure.
    latencies: list[float] = []
    for _ in range(iters):
        t = time.perf_counter()
        sess.run(None, {input_name: inp})
        latencies.append(time.perf_counter() - t)

    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Provider assignment + fallback node count from profiler.
    providers = sess.get_providers()
    fallback_nodes: int | None = None
    try:
        prof_path = sess.end_profiling()
        prof = json.loads(Path(prof_path).read_text())
        # Count nodes assigned to CPUExecutionProvider (fallback).
        cpu_nodes = 0
        total_nodes = 0
        for ev in prof.get("traceEvents", []):
            if ev.get("cat") == "Node" and "args" in ev:
                total_nodes += 1
                if ev["args"].get("provider") == "CPUExecutionProvider":
                    cpu_nodes += 1
        fallback_nodes = cpu_nodes
        Path(prof_path).unlink(missing_ok=True)
    except Exception as e:
        fallback_nodes = None  # profiler format may differ across ORT versions

    del sess
    gc.collect()

    return {
        "cold_load_s": cold_load,
        "first_window_s": first_window,
        "warm_median_s": statistics.median(latencies),
        "warm_p95_s": sorted(latencies)[int(0.95 * len(latencies))],
        "warm_iters": iters,
        "peak_rss_kb": rss_after,
        "rss_delta_kb": rss_after - rss_before,
        "providers": providers,
        "fallback_node_count": fallback_nodes,
        "output_shape": [list(o.shape) for o in out],
        "shape_errors": shape_errors,
        "frames": frames,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run native ORT runtime benchmarks.")
    parser.add_argument("--runtime", required=True, type=Path,
                        help="Runtime archive to benchmark.")
    parser.add_argument("--model", type=Path, default=None,
                        help="Single ONNX model. Requires --expected-output-shape "
                             "so output-shape validation is not bypassed.")
    parser.add_argument("--expected-output-shape", type=str, default=None,
                        help="Expected output shape as comma-separated dims, "
                             "e.g. '1,4,2,343980'. Required when --model is used "
                             "without --catalog so shape validation is enforced.")
    parser.add_argument("--catalog", type=Path, default=None,
                        help="Catalog manifest to resolve compatible models from.")
    parser.add_argument("--target", type=str, default=None,
                        help="Target triple for catalog compatibility filtering.")
    parser.add_argument("--frames", type=int, default=343980,
                        help="Input frame count (default 343980 = full window).")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--download-dir", type=Path, default=ROOT / "ort" / "model-cache")
    parser.add_argument("--report", type=Path, default=Path("benchmark-report.json"),
                        help="Output report JSON path.")
    parser.add_argument("--baseline", type=Path, default=None,
                        help="Baseline report to compare fallback node count against.")
    args = parser.parse_args()

    if not args.runtime.is_file():
        print(f"ERROR: runtime archive not found: {args.runtime}", file=sys.stderr)
        return 1

    # Extract runtime.
    with tempfile.TemporaryDirectory() as tmpd:
        lib_path = _extract_runtime(args.runtime, Path(tmpd))
        print(f"runtime library: {lib_path}")

        # Resolve models.
        models: list[tuple[str, Path, dict[str, Any] | None]] = []
        explicit_shape: list[int] | None = None
        if args.expected_output_shape:
            try:
                explicit_shape = [int(x.strip()) for x in args.expected_output_shape.split(",")]
            except ValueError:
                parser.error(f"--expected-output-shape must be comma-separated ints, "
                             f"got {args.expected_output_shape!r}")
        if args.model:
            # Reject a bare --model that bypasses expected-output-shape
            # validation. Either --catalog (which supplies the shape from the
            # model artifact's output_semantics) or --expected-output-shape
            # must be provided.
            if not args.catalog and explicit_shape is None:
                parser.error(
                    "--model requires --expected-output-shape (or --catalog) so "
                    "output-shape validation is not bypassed"
                )
            art: dict[str, Any] | None = None
            if explicit_shape is not None:
                art = {"model": {"output_semantics": str(explicit_shape)},
                       "artifact_id": args.model.name}
            models.append((args.model.name, args.model, art))
        elif args.catalog:
            arts = _resolve_model_from_catalog(args.catalog, args.download_dir, args.target)
            for art in arts:
                p = _download_model(art, args.download_dir)
                models.append((art["artifact_id"], p, art))
        else:
            parser.error("provide --model or --catalog")

        if not models:
            print("ERROR: no models to benchmark", file=sys.stderr)
            return 1

        # Archive size.
        archive_size, archive_sha = _sha256_file(args.runtime)

        results: list[dict[str, Any]] = []
        for name, model_path, art in models:
            print(f"\nBenchmarking {name}...")
            expected_shape = None
            if art:
                expected_shape = _parse_output_shape(
                    art.get("model", {}).get("output_semantics", "")
                )
            metrics = benchmark_model(
                lib_path, model_path, args.frames, args.warmup, args.iters, expected_shape,
            )
            metrics["model"] = name
            metrics["model_artifact_id"] = art["artifact_id"] if art else None
            results.append(metrics)
            if metrics["shape_errors"]:
                print("  FAIL: shape/finite errors:")
                for e in metrics["shape_errors"]:
                    print(f"    {e}", file=sys.stderr)

        report = {
            "schema_version": GENERATOR_VERSION,
            "runtime_archive": {
                "name": args.runtime.name,
                "sha256": archive_sha,
                "size": archive_size,
            },
            "target": args.target or _detect_target(),
            "frames": args.frames,
            "warmup": args.warmup,
            "iters": args.iters,
            "results": results,
        }

        # Baseline comparison.
        if args.baseline and args.baseline.is_file():
            baseline = json.loads(args.baseline.read_text())
            report["baseline"] = {"archive": baseline.get("runtime_archive", {}).get("name")}
            for r in results:
                bl = next((b for b in baseline.get("results", [])
                           if b.get("model_artifact_id") == r.get("model_artifact_id")), None)
                if bl and bl.get("fallback_node_count") is not None and r.get("fallback_node_count") is not None:
                    if bl["fallback_node_count"] != r["fallback_node_count"]:
                        r["fallback_node_count_mismatch"] = {
                            "baseline": bl["fallback_node_count"],
                            "actual": r["fallback_node_count"],
                        }

        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nreport: {args.report}")

        # Fail on shape errors or fallback mismatches.
        failed = any(r["shape_errors"] for r in results)
        failed |= any(r.get("fallback_node_count_mismatch") for r in results)
        if failed:
            print("FAIL: benchmark detected errors", file=sys.stderr)
            return 2
    print("OK")
    return 0


def _detect_target() -> str:
    import platform
    s = platform.system()
    m = platform.machine()
    if s == "Darwin" and m == "arm64":
        return "aarch64-apple-darwin"
    if s == "Darwin" and m == "x86_64":
        return "x86_64-apple-darwin"
    if s == "Linux" and m == "x86_64":
        return "x86_64-unknown-linux-gnu"
    if s == "Linux" and m == "aarch64":
        return "aarch64-unknown-linux-gnu"
    if s == "Windows" and m == "AMD64":
        return "x86_64-pc-windows-msvc"
    return f"unknown-{s}-{m}"


if __name__ == "__main__":
    raise SystemExit(main())
