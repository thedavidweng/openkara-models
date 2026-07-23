#!/usr/bin/env python3
"""Run the native ORT runtime smoke harness and assemble the JSON report.

Builds (or reuses) the compiled C++ smoke harness in ``ort/smoke/``, invokes
it against the ORT shared library built by the current job, and writes one
JSON report per target. The harness loads the runtime via the platform
dynamic loader and uses the ORT C API directly — it never uses the
onnxruntime Python wheel as proof that the new library works.

The report contains:
  - runtime_digest   (SHA-256 of the runtime shared library)
  - model_digest     (SHA-256 of the ONNX model)
  - target
  - requested_provider
  - available_providers
  - output_shape
  - finite_output
  - session_creation
  - inference
  - provider_assignment
  - fallback_node_count

Usage::

    python scripts/run_native_smoke.py \\
        --runtime ort/packages/onnxruntime-1.27.1-openkara-aarch64-apple-darwin.tar.gz \\
        --model models/htdemucs.onnx \\
        --target aarch64-apple-darwin \\
        --provider coreml \\
        --ort-source ort/source \\
        --report smoke-aarch64-apple-darwin-coreml.json

The runtime archive is extracted with archive_utils.safe_extract. The harness
binary is built into ``ort/smoke/build/`` via CMake.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import archive_utils  # noqa: E402

SMOKE_DIR = ROOT / "ort" / "smoke"


def _sha256_file(path: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return size, h.hexdigest()


def _find_lib(dest: Path) -> Path:
    for p in dest.rglob("*"):
        low = p.name.lower()
        if low.endswith((".so", ".dylib", ".dll")) and "onnxruntime" in low:
            return p
    raise FileNotFoundError(f"no onnxruntime shared library found in extracted runtime")


def _build_harness(ort_source: Path, build_dir: Path) -> Path:
    """Build the C++ smoke harness via CMake. Returns the binary path."""
    build_dir.mkdir(parents=True, exist_ok=True)
    # Pass an absolute ORT_SOURCE_DIR so cmake's get_filename_component(...
    # ABSOLUTE) resolves the include directory correctly regardless of the
    # cmake working directory / CMAKE_CURRENT_SOURCE_DIR.
    ort_source_abs = ort_source.resolve()
    configure = ["cmake", "-S", str(SMOKE_DIR), "-B", str(build_dir),
                 f"-DORT_SOURCE_DIR={ort_source_abs}", "-DCMAKE_BUILD_TYPE=Release"]
    r = subprocess.run(configure, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cmake configure failed:\n{r.stderr}")
    build = ["cmake", "--build", str(build_dir), "--config", "Release",
             "--parallel", str(os.cpu_count() or 4)]
    r = subprocess.run(build, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cmake build failed:\n{r.stderr}")
    binary = build_dir / ("ort_smoke.exe" if os.name == "nt" else "ort_smoke")
    if not binary.is_file():
        # Windows may place it under Release/.
        alt = build_dir / "Release" / "ort_smoke.exe"
        if alt.is_file():
            return alt
        raise FileNotFoundError(f"ort_smoke binary not found under {build_dir}")
    return binary


def run_smoke(
    lib_path: Path, model_path: Path, target: str, provider: str,
    harness: Path,
) -> dict[str, Any]:
    """Invoke the harness and return its parsed JSON output."""
    cmd = [
        str(harness),
        "--lib", str(lib_path),
        "--model", str(model_path),
        "--provider", provider,
        "--target", target,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    # The harness prints JSON to stdout. Parse it.
    out = r.stdout.strip()
    if not out:
        stderr = r.stderr.strip()
        return {
            "target": target,
            "requested_provider": provider,
            "session_creation": "failed",
            "session_creation_error": (
                f"no stdout from harness (exit={r.returncode})"
                + (f"; stderr: {stderr[:500]}" if stderr else "")
            ),
            "inference": "not_attempted",
            "available_providers": "",
            "output_shape": "",
            "finite_output": False,
            "provider_assignment": "",
            "fallback_node_count": None,
            "harness_exit_code": r.returncode,
        }
    try:
        data = json.loads(out)
    except json.JSONDecodeError as e:
        return {
            "target": target,
            "requested_provider": provider,
            "session_creation": "failed",
            "session_creation_error": f"harness stdout not JSON: {e}; stdout={out[:200]}",
            "inference": "not_attempted",
            "available_providers": "",
            "output_shape": "",
            "finite_output": False,
            "provider_assignment": "",
            "fallback_node_count": None,
            "harness_exit_code": r.returncode,
        }
    data["harness_exit_code"] = r.returncode
    return data


def validation_failures(report: dict[str, Any], requested_provider: str) -> list[str]:
    """Return strict native-smoke gate failures.

    CPU and accelerated providers are tested in separate invocations. A
    non-CPU invocation must use the requested provider for at least one node;
    creating a replacement CPU session is a failure.
    """
    failures: list[str] = []
    if report.get("requested_provider") != requested_provider:
        failures.append(
            "requested provider identity mismatch: "
            f"report={report.get('requested_provider')!r}, expected={requested_provider!r}"
        )
    if report.get("harness_exit_code") != 0:
        failures.append(f"harness_exit_code={report.get('harness_exit_code')}")
    if report.get("session_creation") != "ok":
        failures.append("session creation did not succeed")
    if report.get("inference") != "ok":
        failures.append("inference did not succeed")
    if report.get("finite_output") is not True:
        failures.append("output contains NaN/Inf or was not produced")
    if report.get("output_shape") != "[1,4,2,343980]":
        failures.append(f"unexpected output_shape={report.get('output_shape')!r}")
    if report.get("used_fallback") is True:
        failures.append("whole-session CPU fallback was used")
    if report.get("provider_assignment") != requested_provider:
        failures.append(
            "provider assignment mismatch: "
            f"requested={requested_provider!r}, assigned={report.get('provider_assignment')!r}"
        )
    provider_nodes = report.get("provider_node_count")
    if not isinstance(provider_nodes, int) or provider_nodes <= 0:
        failures.append(f"provider_node_count must be > 0, got {provider_nodes!r}")
    total_nodes = report.get("total_node_count")
    if not isinstance(total_nodes, int) or total_nodes <= 0:
        failures.append(f"total_node_count must be > 0, got {total_nodes!r}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the native ORT smoke harness.")
    parser.add_argument("--runtime", required=True, type=Path,
                        help="Runtime archive (tar.gz or zip) built by the current job.")
    parser.add_argument("--model", required=True, type=Path,
                        help="ONNX model to run inference on.")
    parser.add_argument("--target", required=True,
                        help="Target triple (e.g. aarch64-apple-darwin).")
    parser.add_argument("--provider", required=True,
                        choices=["cpu", "coreml", "xnnpack", "directml"],
                        help="Requested execution provider.")
    parser.add_argument("--ort-source", type=Path, default=ROOT / "ort" / "source",
                        help="ORT source checkout (for the C API header).")
    parser.add_argument("--harness-build-dir", type=Path, default=SMOKE_DIR / "build",
                        help="Where to build the C++ harness.")
    parser.add_argument("--harness", type=Path, default=None,
                        help="Pre-built harness binary (skip CMake build).")
    parser.add_argument("--report", required=True, type=Path,
                        help="Output JSON report path.")
    args = parser.parse_args()

    if not args.runtime.is_file():
        print(f"ERROR: runtime archive not found: {args.runtime}", file=sys.stderr)
        return 1
    if not args.model.is_file():
        print(f"ERROR: model not found: {args.model}", file=sys.stderr)
        return 1
    if not args.ort_source.is_dir():
        print(f"ERROR: ORT source not found: {args.ort_source}", file=sys.stderr)
        return 1

    # Build or reuse the harness.
    if args.harness and args.harness.is_file():
        harness = args.harness
    else:
        print("Building native smoke harness...")
        harness = _build_harness(args.ort_source, args.harness_build_dir)
    print(f"  harness: {harness}")

    # Extract the runtime archive safely and locate the shared library.
    with tempfile.TemporaryDirectory() as tmpd:
        dest = Path(tmpd) / "runtime"
        archive_utils.safe_extract(args.runtime, dest)
        lib_path = _find_lib(dest)
        print(f"  runtime library: {lib_path}")

        # Compute digests.
        runtime_size, runtime_sha = _sha256_file(lib_path)
        model_size, model_sha = _sha256_file(args.model)

        # Run the harness.
        print(f"Running smoke: target={args.target} provider={args.provider}...")
        result = run_smoke(lib_path, args.model, args.target, args.provider, harness)

    # Assemble the final report.
    report = {
        "schema_version": "openkara.native-smoke-report/v1",
        "runtime_digest": {
            "sha256": runtime_sha,
            "size": runtime_size,
            "archive_name": args.runtime.name,
        },
        "model_digest": {
            "sha256": model_sha,
            "size": model_size,
            "filename": args.model.name,
        },
        "target": args.target,
        "requested_provider": args.provider,
        "available_providers": result.get("available_providers", ""),
        "output_shape": result.get("output_shape", ""),
        "finite_output": result.get("finite_output", False),
        "session_creation": result.get("session_creation", "failed"),
        "session_creation_error": result.get("session_creation_error"),
        "inference": result.get("inference", "not_attempted"),
        "inference_error": result.get("inference_error"),
        "provider_assignment": result.get("provider_assignment", ""),
        "total_node_count": result.get("total_node_count"),
        "cpu_node_count": result.get("cpu_node_count"),
        "provider_node_count": result.get("provider_node_count"),
        "fallback_node_count": result.get("fallback_node_count"),
        "used_fallback": result.get("used_fallback", False),
        "harness_exit_code": result.get("harness_exit_code"),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"report: {args.report}")
    print(f"  session_creation: {report['session_creation']}")
    print(f"  inference: {report['inference']}")
    print(f"  finite_output: {report['finite_output']}")
    print(f"  provider_assignment: {report['provider_assignment']}")

    failures = validation_failures(report, args.provider)
    if failures:
        print("FAIL: native smoke harness detected errors", file=sys.stderr)
        if report.get("session_creation_error"):
            print(f"  session_creation_error: {report['session_creation_error']}",
                  file=sys.stderr)
        if report.get("inference_error"):
            print(f"  inference_error: {report['inference_error']}",
                  file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 2
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
