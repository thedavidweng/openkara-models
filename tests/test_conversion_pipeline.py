import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class RuntimeContractTests(unittest.TestCase):
    def test_forbidden_domains_gate_catches_nchwc(self):
        from onnx_runtime_contract import forbidden_domain_violations
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
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0][0], "com.microsoft.nchwc")

    def test_forbidden_domains_gate_passes_clean_model(self):
        from onnx_runtime_contract import forbidden_domain_violations
        from onnx import helper, TensorProto

        inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
        out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
        node = helper.make_node("Identity", ["x"], ["y"])
        graph = helper.make_graph([node], "g", [inp], [out])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        self.assertEqual(forbidden_domain_violations(model), [])

    def test_self_test_passes(self):
        from onnx_runtime_contract import run_self_test
        run_self_test()


class BuildEnsembleGraphTests(unittest.TestCase):
    def _make_tiny_model(self, name="tiny"):
        from onnx import helper, TensorProto
        import numpy as np
        from onnx import numpy_helper

        inp = helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, 2, 4])
        out = helper.make_tensor_value_info("stems", TensorProto.FLOAT, [1, 4, 2, 4])
        weights = numpy_helper.from_array(
            np.ones((4, 2, 1), dtype=np.float32), name=f"{name}_w"
        )
        node = helper.make_node("Conv", ["audio", f"{name}_w"], ["stems"])
        graph = helper.make_graph([node], name, [inp], [out], initializer=[weights])
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    def test_ensemble_graph_averages_two_sub_models(self):
        from ensemble_merge import build_ensemble_graph

        sub_models = [self._make_tiny_model("a"), self._make_tiny_model("b")]
        ensemble = build_ensemble_graph(sub_models, 2)

        self.assertEqual(len(ensemble.graph.input), 1)
        self.assertEqual(ensemble.graph.input[0].name, "audio")
        self.assertEqual(len(ensemble.graph.output), 1)
        self.assertEqual(ensemble.graph.output[0].name, "stems")

        op_types = {n.op_type for n in ensemble.graph.node}
        self.assertIn("Sum", op_types)
        self.assertIn("Mul", op_types)
        # Old Unsqueeze/Concat/ReduceMean approach should be gone.
        self.assertNotIn("ReduceMean", op_types)
        self.assertNotIn("Concat", op_types)

    def test_ensemble_graph_namespaces_sub_model_nodes(self):
        from ensemble_merge import build_ensemble_graph
        import numpy as np
        from onnx import numpy_helper

        # Use different weight content so dedup keeps both initializers.
        m0 = self._make_tiny_model("a")
        m1 = self._make_tiny_model("b")
        for init in m1.graph.initializer:
            if init.name == "b_w":
                init.CopyFrom(
                    numpy_helper.from_array(
                        np.full((4, 2, 1), 2.0, dtype=np.float32), name="b_w"
                    )
                )
        sub_models = [m0, m1]
        ensemble = build_ensemble_graph(sub_models, 2)

        init_names = {i.name for i in ensemble.graph.initializer}
        self.assertIn("sub0_a_w", init_names)
        self.assertIn("sub1_b_w", init_names)
        self.assertIn("ensemble_scale", init_names)

        sub0_inputs = {n for node in ensemble.graph.node for n in node.input}
        self.assertIn("sub0_a_w", sub0_inputs)
        self.assertIn("sub1_b_w", sub0_inputs)
        self.assertIn("audio", sub0_inputs)

    def test_ensemble_graph_deduplicates_identical_initializers(self):
        """Sub-models with identical weight content should share one initializer."""
        from ensemble_merge import build_ensemble_graph

        # Two sub-models with identical weights (same name "shared_w").
        sub_models = [self._make_tiny_model("shared"), self._make_tiny_model("shared")]
        ensemble = build_ensemble_graph(sub_models, 2)

        init_names = [i.name for i in ensemble.graph.initializer]
        # Both sub-models have identical weight content, so only one
        # canonical copy should remain (plus ensemble_scale).
        weight_inits = [n for n in init_names if "shared_w" in n]
        self.assertEqual(
            len(weight_inits), 1,
            f"Expected 1 shared weight initializer, got {weight_inits}",
        )

    def test_ensemble_graph_preserves_distinct_initializers(self):
        """Sub-models with different weights should keep both initializers."""
        from ensemble_merge import build_ensemble_graph
        import numpy as np
        from onnx import numpy_helper

        m0 = self._make_tiny_model("a")
        m1 = self._make_tiny_model("b")
        # Overwrite m1's weight with different content.
        for init in m1.graph.initializer:
            if init.name == "b_w":
                arr = np.zeros((4, 2, 1), dtype=np.float32)
                init.CopyFrom(numpy_helper.from_array(arr, name="b_w"))
        sub_models = [m0, m1]
        ensemble = build_ensemble_graph(sub_models, 2)

        init_names = {i.name for i in ensemble.graph.initializer}
        self.assertIn("sub0_a_w", init_names)
        self.assertIn("sub1_b_w", init_names)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class RealValuedSpectrogramPatchTests(unittest.TestCase):
    def _make_fake_model(self):
        class FakeModel:
            cac = True
            nfft = 16
            hop_length = 4

            def _spec(self, x):
                return "original_spec"

            def _ispec(self, z, length=None, scale=0):
                return "original_ispec"

            def _magnitude(self, z):
                return "original_magnitude"

            def _mask(self, z, m):
                return "original_mask"

        return FakeModel()

    def test_apply_swaps_four_spectrogram_methods(self):
        from onnx_stft import RealValuedSpectrogramPatch

        model = self._make_fake_model()
        patch = RealValuedSpectrogramPatch.from_model(model)
        patch.apply()

        self.assertNotEqual(model._spec("x"), "original_spec")
        self.assertNotEqual(model._magnitude("z"), "original_magnitude")
        self.assertNotEqual(model._mask("z", "m"), "original_mask")

        patch.restore()

    def test_restore_returns_original_methods(self):
        from onnx_stft import RealValuedSpectrogramPatch

        model = self._make_fake_model()
        original_spec = model._spec
        original_ispec = model._ispec
        original_magnitude = model._magnitude
        original_mask = model._mask

        with RealValuedSpectrogramPatch.from_model(model):
            pass

        self.assertIs(model._spec, original_spec)
        self.assertIs(model._ispec, original_ispec)
        self.assertIs(model._magnitude, original_magnitude)
        self.assertIs(model._mask, original_mask)

    def test_apply_restore_is_idempotent(self):
        from onnx_stft import RealValuedSpectrogramPatch

        model = self._make_fake_model()
        patch = RealValuedSpectrogramPatch.from_model(model)

        patch.apply()
        patch.apply()
        patch.restore()
        patch.restore()

    def test_rejects_non_cac_model(self):
        from onnx_stft import RealValuedSpectrogramPatch

        model = self._make_fake_model()
        model.cac = False
        with self.assertRaises(RuntimeError):
            RealValuedSpectrogramPatch.from_model(model)


class DemucsLoaderConstantsTests(unittest.TestCase):
    def test_supported_models_match_pipeline(self):
        from demucs_loader import SUPPORTED_MODELS
        self.assertEqual(SUPPORTED_MODELS, ("htdemucs", "htdemucs_ft"))


class MetadataConstantsTests(unittest.TestCase):
    def test_metadata_keys_are_stable(self):
        from onnx_runtime_contract import (
            MODEL_CACHE_KEY_METADATA,
            MODEL_OPTIMIZED_BY_METADATA,
        )
        self.assertEqual(MODEL_CACHE_KEY_METADATA, "openkara.model_cache_key")
        self.assertEqual(MODEL_OPTIMIZED_BY_METADATA, "openkara.optimized_by")


if __name__ == "__main__":
    unittest.main()
