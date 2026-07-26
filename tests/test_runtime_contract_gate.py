import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


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
