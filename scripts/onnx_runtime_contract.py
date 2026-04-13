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
from typing import Iterable

import onnx
from onnx import NodeProto

# Domains that must not appear in default OpenKara standard models. Extend carefully.
FORBIDDEN_OP_DOMAINS = frozenset({"com.microsoft.nchwc"})


def iter_graph_nodes(model: onnx.ModelProto) -> Iterable[NodeProto]:
    for node in model.graph.node:
        yield node
    for fn in getattr(model, "functions", ()):
        for node in fn.node:
            yield node


def collect_op_domains(model: onnx.ModelProto) -> dict[str, set[str]]:
    """Map domain -> set of op_type seen in that domain."""
    domains: dict[str, set[str]] = {}
    for node in iter_graph_nodes(model):
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


def load_onnx_for_contract(path: Path) -> onnx.ModelProto:
    """Load ONNX; resolve external data if present."""
    model = onnx.load(str(path), load_external_data=True)
    return model


def assert_release_onnx_compatible_with_official_ort(onnx_path: Path) -> None:
    """
    Fail fast if the graph uses operator domains not supported by official ORT
    on all OpenKara target platforms (see docs/runtime-contract.md).
    """
    model = load_onnx_for_contract(onnx_path)
    violations = forbidden_domain_violations(model)
    if violations:
        lines = [f"  domain={dom!r} ops={sorted(ops)}" for dom, ops in violations]
        raise RuntimeError(
            "ONNX release violates OpenKara runtime contract (forbidden operator domains):\n"
            + "\n".join(lines)
            + f"\nSee docs/runtime-contract.md. File: {onnx_path}"
        )


def verify_ort_cpu_session(onnx_path: Path) -> None:
    """Create a CPU EP InferenceSession (matches portable official ORT usage)."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    ort.InferenceSession(
        str(onnx_path),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )


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
