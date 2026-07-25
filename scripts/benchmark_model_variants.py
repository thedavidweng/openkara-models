#!/usr/bin/env python3
"""Benchmark model variants against each other (issue #22 evidence).

For each given ONNX model, measures in an isolated subprocess (so peak RSS
is per-model, not cumulative):

  - file size
  - session cold-load time
  - first-window latency
  - warm median latency over N runs
  - peak RSS
  - output tensor bytes per window

Uses the pip onnxruntime with the requested execution provider. This is a
model-variant comparison tool for derivation evidence; per-target runtime
benchmarks against the packaged ORT builds stay in
``run_runtime_benchmarks.py``.

Usage::

    python scripts/benchmark_model_variants.py \\
        --models a.onnx b.onnx --provider cpu --warm-runs 3 --report bench.json
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

FRAMES = 343980

PROVIDER_MAP = {
    "cpu": ["CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
}


def _bench_single(model_path: Path, provider: str, warm_runs: int) -> dict:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3

    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(model_path), sess_options=so,
                                providers=PROVIDER_MAP[provider])
    load_s = time.perf_counter() - t0

    in_name = sess.get_inputs()[0].name
    rng = np.random.default_rng(42)
    audio = (rng.standard_normal((1, 2, FRAMES)) * 0.1).astype(np.float32)

    t0 = time.perf_counter()
    out = sess.run(None, {in_name: audio})[0]
    first_s = time.perf_counter() - t0

    warm = []
    for _ in range(warm_runs):
        t0 = time.perf_counter()
        out = sess.run(None, {in_name: audio})[0]
        warm.append(time.perf_counter() - t0)

    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024

    return {
        "model": str(model_path),
        "provider": provider,
        "file_bytes": model_path.stat().st_size,
        "load_s": round(load_s, 3),
        "first_window_s": round(first_s, 3),
        "warm_median_s": round(statistics.median(warm), 3) if warm else None,
        "warm_runs": warm_runs,
        "peak_rss_mb": rss_kb // 1024,
        "output_shape": list(out.shape),
        "output_bytes_per_window": int(out.nbytes),
        "output_finite": bool(np.isfinite(out).all()),
        "session_providers": sess.get_providers(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", type=Path, required=True)
    parser.add_argument("--provider", choices=list(PROVIDER_MAP), default="cpu")
    parser.add_argument("--warm-runs", type=int, default=3)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--single", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.single:
        print(json.dumps(_bench_single(args.single, args.provider, args.warm_runs)))
        return 0

    results = []
    for model in args.models:
        if not model.is_file():
            print(f"ERROR: not found: {model}", file=sys.stderr)
            return 1
        r = subprocess.run(
            [sys.executable, __file__, "--single", str(model),
             "--provider", args.provider, "--warm-runs", str(args.warm_runs),
             "--models", str(model)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"ERROR benchmarking {model}: {r.stderr}", file=sys.stderr)
            return 1
        entry = json.loads(r.stdout.strip().splitlines()[-1])
        results.append(entry)
        print(f"{model.name} [{args.provider}]: "
              f"file={entry['file_bytes']/1e6:.0f}MB load={entry['load_s']}s "
              f"first={entry['first_window_s']}s warm={entry['warm_median_s']}s "
              f"rss={entry['peak_rss_mb']}MB out={entry['output_bytes_per_window']/1e6:.1f}MB")

    if args.report:
        args.report.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
