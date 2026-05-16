import ast
from pathlib import Path
import unittest


ROOT_DIR = Path(__file__).resolve().parents[1]
CONVERT_SCRIPT = ROOT_DIR / "scripts" / "convert_htdemucs_to_onnx.py"


def parse_convert_script():
    return ast.parse(CONVERT_SCRIPT.read_text())


class ConversionPipelineTests(unittest.TestCase):
    def test_conversion_has_one_patched_export_path(self):
        tree = parse_convert_script()
        function_names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }

        self.assertNotIn("try_direct_export", function_names)
        self.assertNotIn("export_single_model", function_names)
        self.assertIn("export_model_with_real_stft", function_names)

    def test_ort_optimization_stays_within_runtime_contract(self):
        tree = parse_convert_script()
        optimization_levels = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "GraphOptimizationLevel"
        }

        self.assertIn("ORT_ENABLE_EXTENDED", optimization_levels)
        self.assertNotIn("ORT_ENABLE_ALL", optimization_levels)

    def test_no_known_dead_conversion_symbols_remain(self):
        tree = parse_convert_script()
        assigned_names = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        imported_aliases = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }

        self.assertFalse({"N_FFT", "HOP_LENGTH", "SAMPLE_RATE"} & assigned_names)
        self.assertNotIn("nn", imported_aliases)


if __name__ == "__main__":
    unittest.main()
