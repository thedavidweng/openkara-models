#!/usr/bin/env python3
"""Tests for scripts/spectral_core_graph.py (issue #23 PR 2) — onnx only,
no torch, no demucs, no model downloads. Runs in runtime-contract.yml
alongside test_conversion_pipeline.py."""

import sys
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from spectral_core_graph import (  # noqa: E402
    CONTRACT_INPUTS,
    CONTRACT_OUTPUTS,
    SPECTRAL_CONTRACT_METADATA,
    SPECTRAL_CONTRACT_VERSION,
    SpectralCoreContractError,
    assert_spectral_core_graph,
    assert_spectral_core_metadata,
    build_spectral_core_ensemble,
    strip_time_zero_add,
)


def _value_info(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def _contract_io():
    inputs = [_value_info(n, s) for n, s in CONTRACT_INPUTS]
    outputs = [_value_info(n, s) for n, s in CONTRACT_OUTPUTS]
    return inputs, outputs


def make_contract_stub(scale=1.0):
    """A minimal graph with the exact contract interface.

    spectral_out = Concat_4(Reshape(spectral)) * scale
    time_out     = Concat_4(Unsqueeze(mix))    * scale
    """
    inputs, outputs = _contract_io()
    spec_shape = numpy_helper.from_array(
        np.array([1, 1, 2, 2, 2048, 336], dtype=np.int64), name="spec_shape"
    )
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")
    scale_init = numpy_helper.from_array(
        np.array([scale], dtype=np.float32), name="scale"
    )
    nodes = [
        helper.make_node("Reshape", ["spectral", "spec_shape"], ["spec_r"]),
        helper.make_node(
            "Concat", ["spec_r"] * 4, ["spec_c"], axis=1
        ),
        helper.make_node("Mul", ["spec_c", "scale"], ["spectral_out"]),
        helper.make_node("Unsqueeze", ["mix", "axes"], ["mix_u"]),
        helper.make_node("Concat", ["mix_u"] * 4, ["mix_c"], axis=1),
        helper.make_node("Mul", ["mix_c", "scale"], ["time_out"]),
    ]
    graph = helper.make_graph(
        nodes, "contract_stub", inputs, outputs,
        initializer=[spec_shape, axes, scale_init],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)]
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def add_metadata(model, key=SPECTRAL_CONTRACT_METADATA,
                 value=SPECTRAL_CONTRACT_VERSION):
    prop = model.metadata_props.add()
    prop.key = key
    prop.value = value
    return model


class StripTimeZeroAddTests(unittest.TestCase):
    def _model_with_zero_add(self, zero_kind="initializer",
                             zero_value=0.0, extra_consumer=False):
        """time branch: mix --Mul(w)--> tb --Add(zero)--> time_out."""
        inputs, _ = _contract_io()
        outputs = [_value_info("time_out", CONTRACT_OUTPUTS[1][1])]
        w = numpy_helper.from_array(
            np.array([2.0], dtype=np.float32), name="w"
        )
        initializers = [w]
        nodes = [helper.make_node("Mul", ["mix", "w"], ["tb"])]
        if zero_kind == "constant_node":
            nodes.append(helper.make_node(
                "Constant", [], ["zero"],
                value=numpy_helper.from_array(
                    np.array(zero_value, dtype=np.float32), name=""
                ),
            ))
        elif zero_kind == "constant_of_shape":
            # What torch.Tensor.new_zeros(()) actually traces to: a shape
            # Constant feeding ConstantOfShape (default zero fill).
            nodes.append(helper.make_node(
                "Constant", [], ["zero_shape"],
                value=numpy_helper.from_array(
                    np.array([], dtype=np.int64), name=""
                ),
            ))
            nodes.append(
                helper.make_node("ConstantOfShape", ["zero_shape"], ["zero"])
            )
        else:
            initializers.append(numpy_helper.from_array(
                np.array(zero_value, dtype=np.float32), name="zero"
            ))
        nodes.append(helper.make_node("Add", ["tb", "zero"], ["time_out"]))
        if extra_consumer:
            nodes.append(helper.make_node("Add", ["time_out", "zero"], ["unused"]))
        graph = helper.make_graph(
            nodes, "g", [inputs[1]], outputs, initializer=initializers
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17)]
        )
        model.ir_version = 8
        return model

    def test_strips_initializer_zero(self):
        model = self._model_with_zero_add()
        strip_time_zero_add(model)
        ops = [n.op_type for n in model.graph.node]
        self.assertEqual(ops, ["Mul"])
        self.assertEqual(model.graph.node[0].output[0], "time_out")
        self.assertEqual([i.name for i in model.graph.initializer], ["w"])

    def test_strips_constant_node_zero(self):
        model = self._model_with_zero_add(zero_kind="constant_node")
        strip_time_zero_add(model)
        ops = [n.op_type for n in model.graph.node]
        self.assertEqual(ops, ["Mul"])
        self.assertEqual(model.graph.node[0].output[0], "time_out")

    def test_strips_constant_of_shape_zero(self):
        model = self._model_with_zero_add(zero_kind="constant_of_shape")
        strip_time_zero_add(model)
        # The whole dead chain goes: Add, ConstantOfShape, and its shape
        # Constant.
        ops = [n.op_type for n in model.graph.node]
        self.assertEqual(ops, ["Mul"])
        self.assertEqual(model.graph.node[0].output[0], "time_out")

    def test_keeps_shared_zero_constant(self):
        model = self._model_with_zero_add(extra_consumer=True)
        strip_time_zero_add(model)
        # The zero Add feeding time_out is gone, but the shared constant and
        # its other consumer survive.
        self.assertIn("zero", [i.name for i in model.graph.initializer])
        self.assertEqual(
            [n.op_type for n in model.graph.node], ["Mul", "Add"]
        )
        self.assertEqual(model.graph.node[0].output[0], "time_out")

    def test_rejects_nonzero_add(self):
        model = self._model_with_zero_add(zero_value=1.0)
        with self.assertRaises(SpectralCoreContractError):
            strip_time_zero_add(model)

    def test_rejects_non_add_producer(self):
        model = self._model_with_zero_add()
        # Rename the Add away so time_out is produced by the Mul directly.
        del model.graph.node[1]
        model.graph.node[0].output[0] = "time_out"
        with self.assertRaises(SpectralCoreContractError):
            strip_time_zero_add(model)

    def test_stripped_graph_is_numerically_identity(self):
        import onnxruntime as ort

        model = self._model_with_zero_add()
        strip_time_zero_add(model)
        onnx.checker.check_model(model)
        sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        mix = np.random.default_rng(0).standard_normal(
            CONTRACT_INPUTS[1][1]
        ).astype(np.float32)
        (out,) = sess.run(None, {"mix": mix})
        np.testing.assert_allclose(out, mix * 2.0, rtol=0, atol=0)


class AssertSpectralCoreGraphTests(unittest.TestCase):
    def test_conforming_graph_passes(self):
        assert_spectral_core_graph(make_contract_stub())

    def test_rejects_wrong_input_shape(self):
        model = make_contract_stub()
        model.graph.input[0].type.tensor_type.shape.dim[3].dim_value = 2049
        with self.assertRaises(SpectralCoreContractError):
            assert_spectral_core_graph(model)

    def test_rejects_wrong_output_name(self):
        model = make_contract_stub()
        model.graph.output[1].name = "stems"
        with self.assertRaises(SpectralCoreContractError):
            assert_spectral_core_graph(model)

    def test_rejects_symbolic_dim(self):
        model = make_contract_stub()
        model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch"
        with self.assertRaises(SpectralCoreContractError):
            assert_spectral_core_graph(model)

    def test_rejects_non_float_io(self):
        model = make_contract_stub()
        model.graph.input[1].type.tensor_type.elem_type = TensorProto.DOUBLE
        with self.assertRaises(SpectralCoreContractError):
            assert_spectral_core_graph(model)

    def test_rejects_transform_ops(self):
        for op in ("STFT", "DFT", "Col2Im", "HannWindow"):
            model = make_contract_stub()
            model.graph.node.append(
                helper.make_node(op, ["spectral"], ["dangling_" + op])
            )
            with self.assertRaises(SpectralCoreContractError):
                assert_spectral_core_graph(model)

    def test_rejects_dense_dft_initializer_signatures(self):
        for dims in ((2049, 1, 4096), (1, 2049, 4096), (4096, 2048)):
            model = make_contract_stub()
            # Zero-filled placeholder with the DFT filter-bank shape; the
            # gate keys on dimensions, not content.
            bank = onnx.TensorProto()
            bank.name = "dft_bank"
            bank.data_type = TensorProto.FLOAT
            bank.dims.extend(dims)
            model.graph.initializer.append(bank)
            with self.assertRaises(SpectralCoreContractError):
                assert_spectral_core_graph(model)

    def test_allows_benign_large_dims(self):
        model = make_contract_stub()
        benign = numpy_helper.from_array(
            np.zeros((512, 2048), dtype=np.float32), name="ff_weight"
        )
        model.graph.initializer.append(benign)
        assert_spectral_core_graph(model)

    def test_rejects_dft_shaped_constant_node(self):
        model = make_contract_stub()
        bank = onnx.TensorProto()
        bank.name = ""
        bank.data_type = TensorProto.FLOAT
        bank.dims.extend((2049, 1, 4096))
        model.graph.node.append(
            helper.make_node("Constant", [], ["bank"], value=bank)
        )
        with self.assertRaises(SpectralCoreContractError):
            assert_spectral_core_graph(model)


class AssertSpectralCoreMetadataTests(unittest.TestCase):
    def test_missing_metadata_rejected(self):
        with self.assertRaises(SpectralCoreContractError):
            assert_spectral_core_metadata(make_contract_stub())

    def test_wrong_version_rejected(self):
        model = add_metadata(make_contract_stub(),
                             value="openkara.spectral-contract/v2")
        with self.assertRaises(SpectralCoreContractError):
            assert_spectral_core_metadata(model)

    def test_correct_version_passes(self):
        assert_spectral_core_metadata(add_metadata(make_contract_stub()))


class BuildSpectralCoreEnsembleTests(unittest.TestCase):
    def test_merged_graph_structure(self):
        ensemble = build_spectral_core_ensemble(
            [make_contract_stub(1.0), make_contract_stub(3.0)], 2
        )
        onnx.checker.check_model(ensemble)
        assert_spectral_core_graph(ensemble)

        tensor_names = {
            name for n in ensemble.graph.node for name in n.output
        }
        self.assertTrue(any(n.startswith("sub0_") for n in tensor_names))
        self.assertTrue(any(n.startswith("sub1_") for n in tensor_names))

        ops = [n.op_type for n in ensemble.graph.node]
        self.assertEqual(ops.count("Sum"), 2)  # one per contract output
        self.assertNotIn("ReduceMean", ops)

        scale = next(
            i for i in ensemble.graph.initializer if i.name == "ensemble_scale"
        )
        np.testing.assert_allclose(numpy_helper.to_array(scale), [0.5])

        # Identical initializers across sub-models are deduplicated (the
        # stubs share spec_shape/axes; scale differs between subs).
        init_names = [i.name for i in ensemble.graph.initializer]
        self.assertIn("sub0_spec_shape", init_names)
        self.assertNotIn("sub1_spec_shape", init_names)
        self.assertIn("sub0_scale", init_names)
        self.assertIn("sub1_scale", init_names)

    def test_merged_graph_averages_numerically(self):
        import onnxruntime as ort

        ensemble = build_spectral_core_ensemble(
            [make_contract_stub(1.0), make_contract_stub(3.0)], 2
        )
        sess = ort.InferenceSession(
            ensemble.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        rng = np.random.default_rng(1)
        spectral = rng.standard_normal(CONTRACT_INPUTS[0][1]).astype(np.float32)
        mix = rng.standard_normal(CONTRACT_INPUTS[1][1]).astype(np.float32)
        spectral_out, time_out = sess.run(
            None, {"spectral": spectral, "mix": mix}
        )
        # average of 1x and 3x = 2x of the stub's per-sub output
        expected_time = np.concatenate([mix[:, None]] * 4, axis=1) * 2.0
        np.testing.assert_allclose(time_out, expected_time, atol=1e-5)
        expected_spec = np.concatenate(
            [spectral.reshape(1, 1, 2, 2, 2048, 336)] * 4, axis=1
        ) * 2.0
        np.testing.assert_allclose(spectral_out, expected_spec, atol=1e-5)

    def test_rejects_wrong_sub_model_interface(self):
        bad = make_contract_stub()
        bad.graph.input[0].name = "audio"
        for node in bad.graph.node:
            for i, name in enumerate(node.input):
                if name == "spectral":
                    node.input[i] = "audio"
        with self.assertRaises(SpectralCoreContractError):
            build_spectral_core_ensemble([bad, make_contract_stub()], 2)

    def test_rejects_wrong_count(self):
        with self.assertRaises(ValueError):
            build_spectral_core_ensemble([make_contract_stub()], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
