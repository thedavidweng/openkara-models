"""Tests for the model derivation tools (issue #22): initializer dedup and
the karaoke_2stem projection. Uses tiny synthetic graphs so CI never needs
the real model artifacts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import dedupe_onnx_initializers as dd  # noqa: E402
import derive_karaoke_2stem as k2s  # noqa: E402

FRAMES = 16


def _tiny_four_stem_model() -> onnx.ModelProto:
    """[1,2,FRAMES] input -> Conv-free graph producing [1,4,2,FRAMES].

    stems[:, i] = audio * w_i with two of the four weights byte-identical,
    exercising both derivation tools on one graph.
    """
    audio = helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, 2, FRAMES])
    stems = helper.make_tensor_value_info("stems", TensorProto.FLOAT, [1, 4, 2, FRAMES])

    inits = [
        onnx.numpy_helper.from_array(np.full((1, 1, 2, FRAMES), 0.5, np.float32), "w_drums"),
        onnx.numpy_helper.from_array(np.full((1, 1, 2, FRAMES), 0.25, np.float32), "w_bass"),
        # w_other duplicates w_drums content exactly (dedup target).
        onnx.numpy_helper.from_array(np.full((1, 1, 2, FRAMES), 0.5, np.float32), "w_other"),
        onnx.numpy_helper.from_array(np.full((1, 1, 2, FRAMES), 2.0, np.float32), "w_vocals"),
        onnx.numpy_helper.from_array(np.array([1, 1, 2, FRAMES], np.int64), "shape4d"),
    ]
    nodes = [
        helper.make_node("Reshape", ["audio", "shape4d"], ["audio4d"], name="reshape"),
        helper.make_node("Mul", ["audio4d", "w_drums"], ["drums"], name="mul_drums"),
        helper.make_node("Mul", ["audio4d", "w_bass"], ["bass"], name="mul_bass"),
        helper.make_node("Mul", ["audio4d", "w_other"], ["other"], name="mul_other"),
        helper.make_node("Mul", ["audio4d", "w_vocals"], ["vocals"], name="mul_vocals"),
        helper.make_node("Concat", ["drums", "bass", "other", "vocals"], ["stems"],
                         axis=1, name="concat"),
    ]
    graph = helper.make_graph(nodes, "tiny4stem", [audio], [stems], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    model.metadata_props.add(key="openkara.model_cache_key", value="tiny-test")
    onnx.checker.check_model(model)
    return model


def _run(model: onnx.ModelProto, audio: np.ndarray) -> np.ndarray:
    import onnxruntime as ort
    sess = ort.InferenceSession(model.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    return sess.run(None, {"audio": audio})[0]


AUDIO = np.random.default_rng(7).standard_normal((1, 2, FRAMES)).astype(np.float32)


# ---------------------------------------------------------------------------
# dedupe_onnx_initializers
# ---------------------------------------------------------------------------

def test_dedupe_merges_identical_and_preserves_outputs() -> None:
    model = _tiny_four_stem_model()
    before = _run(model, AUDIO)
    stats = dd.dedupe_initializers(model)
    assert stats["initializers_removed"] == 1  # w_other -> w_drums
    assert stats["bytes_saved"] == 2 * FRAMES * 4
    names = {t.name for t in model.graph.initializer}
    assert "w_drums" in names and "w_other" not in names
    onnx.checker.check_model(model)
    after = _run(model, AUDIO)
    np.testing.assert_array_equal(before, after)


def test_dedupe_keeps_lexicographically_smallest_name() -> None:
    model = _tiny_four_stem_model()
    dd.dedupe_initializers(model)
    mul_other = next(n for n in model.graph.node if n.name == "mul_other")
    assert mul_other.input[1] == "w_drums"  # 'w_drums' < 'w_other'


def test_dedupe_distinct_values_untouched() -> None:
    model = _tiny_four_stem_model()
    dd.dedupe_initializers(model)
    names = {t.name for t in model.graph.initializer}
    assert {"w_bass", "w_vocals"} <= names


def test_dedupe_is_deterministic() -> None:
    a, b = _tiny_four_stem_model(), _tiny_four_stem_model()
    dd.dedupe_initializers(a)
    dd.dedupe_initializers(b)
    assert a.SerializeToString() == b.SerializeToString()


# ---------------------------------------------------------------------------
# derive_karaoke_2stem
# ---------------------------------------------------------------------------

def test_projection_math_matches_source() -> None:
    src = _tiny_four_stem_model()
    src_out = _run(src, AUDIO)
    derived = k2s.derive_karaoke_2stem(_tiny_four_stem_model(), "f" * 64)
    onnx.checker.check_model(derived)
    der_out = _run(derived, AUDIO)
    assert der_out.shape == (1, 2, 2, FRAMES)
    np.testing.assert_array_equal(der_out[:, 0], src_out[:, 3])  # vocals bit-exact
    acc = src_out[:, 0] + src_out[:, 1] + src_out[:, 2]
    np.testing.assert_allclose(der_out[:, 1], acc, atol=1e-6)


def test_projection_uses_only_reduced_build_ops() -> None:
    """The projection must load on the published reduced-operator runtimes,
    so it may only add ops already present in the stable models."""
    allowed = {"Slice", "Sum", "Concat"}
    derived = k2s.derive_karaoke_2stem(_tiny_four_stem_model(), "f" * 64)
    added = {n.op_type for n in derived.graph.node if n.name.startswith(k2s.PREFIX)}
    assert added <= allowed, f"projection added ops outside the reduced set: {added - allowed}"


def test_projection_metadata_and_cache_key() -> None:
    derived = k2s.derive_karaoke_2stem(_tiny_four_stem_model(), "a" * 64)
    meta = {p.key: p.value for p in derived.metadata_props}
    assert meta["openkara.model_cache_key"] == "tiny-test-karaoke2stem"
    assert meta["openkara.stem_profile"] == "karaoke-2stem"
    assert meta["openkara.derived_from_sha256"] == "a" * 64
    assert "vocals/accompaniment" in meta["openkara.output_semantics"]


def test_projection_is_deterministic() -> None:
    a = k2s.derive_karaoke_2stem(_tiny_four_stem_model(), "a" * 64)
    b = k2s.derive_karaoke_2stem(_tiny_four_stem_model(), "a" * 64)
    assert a.SerializeToString() == b.SerializeToString()


def test_projection_rejects_non_four_source_graph() -> None:
    model = _tiny_four_stem_model()
    model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 2
    with pytest.raises(ValueError):
        k2s.derive_karaoke_2stem(model, "a" * 64)


def test_projection_rejects_name_collision() -> None:
    model = _tiny_four_stem_model()
    model.graph.node[0].output[0] = f"{k2s.PREFIX}stems"
    model.graph.node[1].input[0] = f"{k2s.PREFIX}stems"
    model.graph.node[2].input[0] = f"{k2s.PREFIX}stems"
    model.graph.node[3].input[0] = f"{k2s.PREFIX}stems"
    model.graph.node[4].input[0] = f"{k2s.PREFIX}stems"
    with pytest.raises(ValueError):
        k2s.derive_karaoke_2stem(model, "a" * 64)


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------

def test_cli_end_to_end(tmp_path: Path) -> None:
    src_path = tmp_path / "tiny.onnx"
    onnx.save(_tiny_four_stem_model(), str(src_path))

    dedup_path = tmp_path / "tiny.dedup.onnx"
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "dedupe_onnx_initializers.py"),
         "--input", str(src_path), "--output", str(dedup_path)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert dedup_path.stat().st_size < src_path.stat().st_size

    k2s_path = tmp_path / "tiny.k2s.onnx"
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "derive_karaoke_2stem.py"),
         "--input", str(dedup_path), "--output", str(k2s_path)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    out = onnx.load(str(k2s_path))
    dims = [d.dim_value for d in out.graph.output[0].type.tensor_type.shape.dim]
    assert dims == [1, 2, 2, FRAMES]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
