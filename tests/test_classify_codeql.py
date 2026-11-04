import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from classify_codeql import classify


class ClassificationTests(unittest.TestCase):
    def test_paths_and_schedule_select_languages(self):
        self.assertEqual(classify(["docs/guide.md"], "pull_request"), [])
        self.assertEqual([item["language"] for item in classify(["scripts/export_spectral_core.py"], "pull_request")], ["python"])
        self.assertEqual([item["language"] for item in classify(["ort/smoke/main.cpp"], "pull_request")], ["c-cpp"])
        self.assertEqual([item["language"] for item in classify(["ort/source-lock.json"], "pull_request")], ["c-cpp"])
        self.assertEqual(len(classify([".github/workflows/ci.yml"], "pull_request")), 3)
        self.assertEqual(len(classify([], "schedule")), 3)


if __name__ == "__main__":
    unittest.main()
