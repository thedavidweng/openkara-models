#!/usr/bin/env python3
"""
Shared checks for OpenKara "standard" ONNX releases vs official ONNX Runtime builds.

Standard release artifacts must load on official ORT CPU for Linux x64, Windows x64,
macOS x64, and macOS arm64 without custom-built ORT (no reliance on NCHWc layout ops).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx

FORBIDDEN_OP_DOMAINS = frozenset({"com.microsoft.nchwc"})

MODEL_CACHE_KEY_METADATA = "openkara.model_cache_key"
MODEL_OPTIMIZED_BY_METADATA = "openkara.optimized_by"


def collect_op_domains(model: onnx.ModelProto) -> dict[str, set[str]]:
    """Map domain -> set of op_type seen in that domain."""
    domains: dict[str, set[str]] = {}
    for node in model.graph.node:
        domain = node.domain or ""
        domains.setdefault(domain, set()).add(node.op_type)
    return domains


def forbidden_domain_violations(model: onnx.ModelProto) -> list[tuple[str, set[str]]]:
    domains = collect_op_domains(model)
    out: list[tuple[str, set[str]]] = []
    for d in sorted(domains.keys()):
        if d in FORBIDDEN_OP_DOMAINS:
            out.append((d, domains[d]))
    return out


def assert_release_onnx_compatible_with_official_ort(onnx_path: Path) -> None:
    """
    Fail fast if the graph uses operator domains not supported by official ORT
    on all OpenKara target platforms (see docs/runtime-contract.md).
    """
    model = onnx.load(str(onnx_path), load_external_data=True)
    violations = forbidden_domain_violations(model)
    if violations:
        lines = [f"  domain={dom!r} ops={sorted(ops)}" for dom, ops in violations]
        raise RuntimeError(
            "ONNX release violates OpenKara runtime contract (forbidden operator domains):\n"
            + "\n".join(lines)
            + f"\nSee docs/runtime-contract.md. File: {onnx_path}"
        )


def make_contract_compliant_session(onnx_path: Path, optimized_model_filepath=None):
    """Create an ORT InferenceSession that satisfies the runtime contract.

    Uses ORT_ENABLE_EXTENDED (not ORT_ENABLE_ALL) and CPUExecutionProvider so
    layout passes that emit com.microsoft.nchwc ops are not applied. This is
    the single source of truth for contract-compliant session creation — see
    docs/runtime-contract.md.

    Log severity is set to ERROR (3, not the default WARNING=2) to suppress
    non-essential ORT warnings at the session level. This does NOT suppress
    the known device_discovery.cc GetPciBusId warning that ORT 1.24+ emits on
    GitHub Linux runners (microsoft/onnxruntime#27268), because that warning
    comes from a statically-initialized logger in the pybind module that
    hardcodes WARNING level and bypasses all Python API and env var control
    (microsoft/onnxruntime#27092). That warning is harmless — it comes from
    GPU device discovery for TRT-RTX EP, which is irrelevant since we only
    use CPUExecutionProvider. The only way to silence it would be to redirect
    fd 2 during `import onnxruntime`, but that hides ALL stderr output
    including real errors, so we accept the warning.

    If optimized_model_filepath is set, ORT writes the optimized graph to that
    path before returning (used by the conversion pipeline to emit the final
    optimized ONNX artifact).
    """
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    so.log_severity_level = 3  # ORT LogLevel.ERROR: 0=Verbose 1=Info 2=Warning 3=Error 4=Fatal
    if optimized_model_filepath is not None:
        so.optimized_model_filepath = str(optimized_model_filepath)
    return ort.InferenceSession(
        str(onnx_path),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )


def verify_ort_cpu_session(onnx_path: Path) -> None:
    """Create and discard a CPU EP InferenceSession (matches portable official ORT usage)."""
    make_contract_compliant_session(onnx_path)


def run_self_test() -> None:
    """Ensure the domain gate catches com.microsoft.nchwc nodes."""
    from onnx import helper, TensorProto

    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    bad_node = helper.make_node("Identity", ["x"], ["y"], domain="com.microsoft.nchwc")
    graph = helper.make_graph([bad_node], "g", [inp], [out])
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft.nchwc", 1),
        ],
    )
    violations = forbidden_domain_violations(model)
    if not violations:
        raise RuntimeError("self-test expected violations for com.microsoft.nchwc Identity")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify ONNX artifacts meet OpenKara official-ORT runtime contract."
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Path to ONNX file to check (protobuf scan + optional ORT session).",
    )
    parser.add_argument(
        "--ort-session",
        action="store_true",
        help="Also create an ORT CPU InferenceSession (requires onnxruntime).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run internal sanity check for the domain gate; no model file needed.",
    )
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        print("runtime-contract self-test: OK")
        return 0

    if not args.model:
        parser.error("provide --model or --self-test")

    path = args.model.resolve()
    if not path.is_file():
        print(f"ERROR: model not found: {path}", file=sys.stderr)
        return 1

    try:
        assert_release_onnx_compatible_with_official_ort(path)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1

    print(f"Protobuf domain check OK: {path}")

    if args.ort_session:
        try:
            verify_ort_cpu_session(path)
        except Exception as e:
            print(f"ERROR: ORT CPU InferenceSession failed: {e}", file=sys.stderr)
            return 1
        print(f"ORT CPU session OK: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
