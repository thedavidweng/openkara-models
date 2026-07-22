"""Tests for the required-operator config generator and reduced-build support."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import onnx
import pytest
from onnx import helper, TensorProto

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _make_model(path: Path, ops_with_domains: list[tuple[str, str]]) -> None:
    """Build a tiny ONNX model with the given (domain, op_type) nodes."""
    inputs = [helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 2]),
              helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 2])]
    outputs = [helper.make_tensor_value_info("C", TensorProto.FLOAT, [2, 2])]
    nodes = []
    prev = "A"
    for i, (domain, op) in enumerate(ops_with_domains):
        out = f"T{i}" if i < len(ops_with_domains) - 1 else "C"
        # First node consumes A and B; later nodes consume prev output and B.
        ins = [prev, "B"] if op in ("Add", "Mul", "Sub", "Div") else [prev]
        nodes.append(helper.make_node(op, ins, [out], domain=domain))
        prev = out
    graph = helper.make_graph(nodes, "test", inputs, outputs)
    domains = {d if d else "": 17 for d, _ in ops_with_domains}
    domains.setdefault("", 17)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid(d, v) for d, v in domains.items()],
    )
    onnx.save(model, str(path))


def test_extract_operators_single_domain(tmp_path: Path) -> None:
    sys.path.insert(0, str(SCRIPTS))
    import generate_required_operators as gen
    m = tmp_path / "m.onnx"
    _make_model(m, [("", "Add"), ("", "Relu"), ("", "Mul")])
    ops, opsets = gen.extract_operators(m)
    assert ops == {"": {"Add", "Relu", "Mul"}}
    assert opsets == {"": {17}}


def test_extract_operators_multiple_domains(tmp_path: Path) -> None:
    sys.path.insert(0, str(SCRIPTS))
    import generate_required_operators as gen
    m = tmp_path / "m.onnx"
    _make_model(m, [("", "Add"), ("com.microsoft", "Mul"), ("", "Relu")])
    ops, opsets = gen.extract_operators(m)
    assert ops == {"": {"Add", "Relu"}, "com.microsoft": {"Mul"}}
    assert opsets == {"": {17}, "com.microsoft": {17}}


def test_merge_operator_sets(tmp_path: Path) -> None:
    sys.path.insert(0, str(SCRIPTS))
    import generate_required_operators as gen
    a = ("a.onnx", {"": {"Add", "Relu"}}, {"": {17}})
    b = ("b.onnx", {"": {"Add", "Mul"}, "com.microsoft": {"Foo"}},
         {"": {17}, "com.microsoft": {17}})
    merged_ops, merged_opsets = gen.merge_operator_sets([a, b])
    assert merged_ops == {"": {"Add", "Relu", "Mul"}, "com.microsoft": {"Foo"}}
    assert merged_opsets == {"": {17}, "com.microsoft": {17}}


def test_write_ort_config_format(tmp_path: Path) -> None:
    sys.path.insert(0, str(SCRIPTS))
    import generate_required_operators as gen
    cfg = tmp_path / "ops.config"
    gen.write_ort_config(
        {"": {"Add", "Mul"}, "com.microsoft": {"Foo", "Bar"}},
        {"": {17}, "com.microsoft": {17}},
        cfg,
    )
    lines = cfg.read_text().strip().split("\n")
    # Empty domain first (sorted), then com.microsoft.
    # Format: <domain>;<opset>;<ops>
    assert lines[0] == ";17;Add,Mul"
    assert lines[1] == "com.microsoft;17;Bar,Foo"


def test_generate_end_to_end(tmp_path: Path) -> None:
    sys.path.insert(0, str(SCRIPTS))
    import generate_required_operators as gen
    m1 = tmp_path / "m1.onnx"
    m2 = tmp_path / "m2.onnx"
    _make_model(m1, [("", "Add"), ("", "Relu")])
    _make_model(m2, [("", "Mul"), ("com.microsoft", "Foo")])
    stem = tmp_path / "reqops"
    sidecar = gen.generate([m1, m2], stem)
    cfg = (tmp_path / "reqops.config").read_text().strip().split("\n")
    assert cfg[0] == ";17;Add,Mul,Relu"
    assert cfg[1] == "com.microsoft;17;Foo"
    assert sidecar["schema_version"] == "openkara.required-operators/v1"
    assert len(sidecar["contributors"]) == 2
    assert set(sidecar["union_operator_set"][""]) == {"Add", "Mul", "Relu"}
    assert sidecar["union_operator_set"]["com.microsoft"] == ["Foo"]
    assert "sha256" in sidecar["config_file"]


def test_generate_cli(tmp_path: Path) -> None:
    m = tmp_path / "m.onnx"
    _make_model(m, [("", "Add"), ("", "Mul")])
    out = tmp_path / "reqops"
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "generate_required_operators.py"),
         "--models", str(m), "--output", out],
        capture_output=True, text=True, check=True,
    )
    assert (tmp_path / "reqops.config").is_file()
    assert (tmp_path / "reqops.json").is_file()
    sidecar = json.loads((tmp_path / "reqops.json").read_text())
    assert sidecar["union_operator_set"][""] == ["Add", "Mul"]


def test_source_lock_reduced_build_section_valid() -> None:
    """The committed source lock has a well-formed reduced_build section."""
    sys.path.insert(0, str(SCRIPTS))
    import validate_source_lock as v
    lock = json.loads((ROOT / "ort" / "source-lock.json").read_text())
    assert "reduced_build" in lock
    errors = v.validate_lock(lock)
    assert errors == [], errors


def test_validate_source_lock_rejects_malformed_reduced_build(tmp_path: Path) -> None:
    sys.path.insert(0, str(SCRIPTS))
    import validate_source_lock as v
    lock = json.loads((ROOT / "ort" / "source-lock.json").read_text())
    lock["reduced_build"] = {"config_path": "x"}  # missing required fields
    errors = v.validate_lock(lock)
    assert any("reduced_build" in e and "missing" in e for e in errors)


def test_build_runtime_reduced_flag_help() -> None:
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "build_runtime.py"), "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert "--reduced" in r.stdout


def test_verify_runtime_package_parses_reduced_name() -> None:
    sys.path.insert(0, str(SCRIPTS))
    import verify_runtime_package as v
    target, is_reduced = v._target_from_archive_name(
        "onnxruntime-1.27.1-openkara-x86_64-unknown-linux-gnu-reduced.tar.gz"
    )
    assert target == "x86_64-unknown-linux-gnu"
    assert is_reduced is True
    target2, is_reduced2 = v._target_from_archive_name(
        "onnxruntime-1.27.1-openkara-x86_64-unknown-linux-gnu.tar.gz"
    )
    assert target2 == "x86_64-unknown-linux-gnu"
    assert is_reduced2 is False
