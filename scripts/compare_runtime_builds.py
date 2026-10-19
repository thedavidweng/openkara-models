#!/usr/bin/env python3
"""Compare a full and a reduced ORT runtime build.

Produces a structured comparison report covering:

  * archive size (compressed bytes)
  * installed size (sum of extracted file bytes)
  * file list delta (files only in one build)
  * runtime library size
  * session cold-load time (when ``--model`` and a loadable runtime are given)
  * first-window inference latency
  * warm median/p95 inference latency
  * peak RSS during inference
  * provider assignment + fallback node count
  * output numerical equivalence vs the full runtime (max abs / MSE)

The size/file comparison runs without a model or runtime. The inference
comparison requires ``--model`` (an ONNX model) and loads each runtime archive's
shared library directly through the native C API (not a Python wheel). On CI
this runs on the native runner for each target.

Fails (non-zero exit) if:
  - the reduced runtime cannot load the model (a required kernel was pruned);
  - the fallback node count differs from the full runtime (partitioning changed);
  - output max-abs error exceeds ``--tolerance`` (default 0, i.e. bit-identical;
    raise for FP16/quantized variants in #22).

Usage::

    python scripts/compare_runtime_builds.py \\
        --full ort/packages/onnxruntime-1.27.1-openkara-x86_64-unknown-linux-gnu.tar.gz \\
        --reduced ort/packages/onnxruntime-1.27.1-openkara-x86_64-unknown-linux-gnu-reduced.tar.gz \\
        --report report.json

    # With inference comparison (native runner only):
    python scripts/compare_runtime_builds.py --full ... --reduced ... \\
        --model models/htdemucs.spectral.onnx --frames 343980 --report report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import archive_utils  # noqa: E402


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _extract_archive(archive: Path) -> dict[str, bytes]:
    return archive_utils.safe_read_archive(archive)


def _find_lib(files: dict[str, bytes]) -> str | None:
    """Return the archive member name of the runtime shared library."""
    for name in sorted(files):
        low = name.lower()
        if Path(low).name in {"libonnxruntime.so", "libonnxruntime.dylib", "onnxruntime.dll"}:
            return name
    return None


def size_comparison(full: Path, reduced: Path) -> dict[str, Any]:
    full_files = _extract_archive(full)
    red_files = _extract_archive(reduced)
    full_size = sum(len(b) for b in full_files.values())
    red_size = sum(len(b) for b in red_files.values())
    full_lib = _find_lib(full_files)
    red_lib = _find_lib(red_files)
    return {
        "full": {
            "archive_bytes": full.stat().st_size,
            "installed_bytes": full_size,
            "file_count": len(full_files),
            "library": {"name": full_lib, "bytes": len(full_files[full_lib]) if full_lib else None},
        },
        "reduced": {
            "archive_bytes": reduced.stat().st_size,
            "installed_bytes": red_size,
            "file_count": len(red_files),
            "library": {"name": red_lib, "bytes": len(red_files[red_lib]) if red_lib else None},
        },
        "archive_size_delta_bytes": reduced.stat().st_size - full.stat().st_size,
        "installed_size_delta_bytes": red_size - full_size,
        "library_size_delta_bytes": (
            len(red_files[red_lib]) - len(full_files[full_lib])
            if full_lib and red_lib else None
        ),
        "files_only_in_full": sorted(set(full_files) - set(red_files)),
        "files_only_in_reduced": sorted(set(red_files) - set(full_files)),
    }


def inference_comparison(
    full: Path, reduced: Path, model_path: Path, frames: int,
    warmup: int, iters: int, tolerance: float,
    target: str | None = None, provider: str = "cpu", source: Path | None = None,
) -> dict[str, Any]:
    """Measure the actual two archives and compare every raw output tensor."""
    import math
    import numpy as np
    from native_quality import NativeRuntime
    from run_runtime_benchmarks import _detect_target

    if frames != 343980 or warmup < 0 or iters < 1 or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("comparison requires one full 343980-frame window, positive iters and finite non-negative warmup/tolerance")
    mix = np.random.default_rng(42).normal(0, 0.01, (2, frames)).astype(np.float32)
    with tempfile.TemporaryDirectory() as directory:
        runtimes = [NativeRuntime(archive, source, target or _detect_target(), provider,
                                  Path(directory) / name)
                    for archive, name in ((full, "full"), (reduced, "reduced"))]
        metrics = [rt.measure(model_path, mix, 44100, warmup, iters)[1] for rt in runtimes]
        full_out, red_out = [rt.raw_outputs for rt in runtimes]
        if len(full_out) != len(red_out) or not full_out:
            raise ValueError("native output counts differ or are empty")
        max_abs, squared_error, count = 0.0, 0.0, 0
        for fo, ro in zip(full_out, red_out):
            if fo.shape != ro.shape or not np.isfinite(fo).all() or not np.isfinite(ro).all():
                raise ValueError("native output shape/finite validation failed")
            diff = fo.astype(np.float64) - ro.astype(np.float64)
            max_abs = max(max_abs, float(np.abs(diff).max()))
            squared_error += float((diff ** 2).sum())
            count += diff.size
        return {
            "measurement_backend": "native-c-api",
            "full": metrics[0] | {"runtime_archive": runtimes[0].archive},
            "reduced": metrics[1] | {"runtime_archive": runtimes[1].archive},
            "output_max_abs": max_abs, "output_mse": squared_error / count,
            "tolerance": tolerance,
            "passed": max_abs <= tolerance and
                      metrics[0]["fallback_node_count"] == metrics[1]["fallback_node_count"],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare full vs reduced ORT builds.")
    parser.add_argument("--full", required=True, type=Path, help="Full build archive.")
    parser.add_argument("--reduced", required=True, type=Path, help="Reduced build archive.")
    parser.add_argument("--model", type=Path, default=None,
                        help="ONNX model for inference comparison (optional).")
    parser.add_argument("--frames", type=int, default=343980,
                        help="Input frame count for inference (default 343980).")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=0.0,
                        help="Max acceptable output max-abs error (default 0 = bit-identical).")
    parser.add_argument("--report", type=Path, default=Path("runtime-comparison-report.json"),
                        help="Output report JSON path.")
    parser.add_argument("--target")
    parser.add_argument("--provider", choices=["cpu", "coreml", "xnnpack", "directml"], default="cpu")
    parser.add_argument("--ort-source", type=Path, help="Optional source header cross-check.")
    args = parser.parse_args()

    if not args.full.is_file() or not args.reduced.is_file():
        print("ERROR: archive not found", file=sys.stderr)
        return 1

    print("Comparing archive sizes...")
    report: dict[str, Any] = {"schema_version": "openkara.runtime-comparison/v1"}
    report["sizes"] = size_comparison(args.full, args.reduced)
    sz = report["sizes"]
    print(f"  full:     {sz['full']['archive_bytes']} bytes archive, "
          f"{sz['full']['installed_bytes']} installed")
    print(f"  reduced:  {sz['reduced']['archive_bytes']} bytes archive, "
          f"{sz['reduced']['installed_bytes']} installed")
    print(f"  archive delta:  {sz['archive_size_delta_bytes']} bytes")
    print(f"  installed delta: {sz['installed_size_delta_bytes']} bytes")

    if args.model:
        if not args.model.is_file():
            print(f"ERROR: model not found: {args.model}", file=sys.stderr)
            return 1
        print("Running inference comparison...")
        report["inference"] = inference_comparison(
            args.full, args.reduced, args.model, args.frames,
            args.warmup, args.iters, args.tolerance, args.target, args.provider, args.ort_source,
        )
        inf = report["inference"]
        print(f"  output max-abs: {inf['output_max_abs']} (tolerance {inf['tolerance']})")
        if not inf["passed"]:
            print("FAIL: reduced runtime output diverges beyond tolerance", file=sys.stderr)
            args.report.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
            return 2

    args.report.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"OK: report written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
