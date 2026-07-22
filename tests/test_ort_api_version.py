"""Tests for scripts/ort_api_version.py — parsing ORT_API_VERSION from the
pinned source header.

The source lock no longer carries a manually maintained ort_api_version; it
is derived from include/onnxruntime/core/session/onnxruntime_c_api.h of the
pinned ORT source. For v1.27.1 the parsed value must be 27.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import ort_api_version  # noqa: E402

HEADER_REL = ort_api_version.HEADER_REL


def _write_header(source_dir: Path, content: str) -> None:
    header = source_dir / HEADER_REL
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_text(content, encoding="utf-8")


class ParseApiVersionTests(unittest.TestCase):
    def test_parses_simple_define(self):
        with tempfile.TemporaryDirectory() as td:
            sd = Path(td)
            _write_header(sd, "#define ORT_API_VERSION 27\n")
            self.assertEqual(ort_api_version.parse_ort_api_version(sd), 27)

    def test_parses_with_surrounding_code(self):
        with tempfile.TemporaryDirectory() as td:
            sd = Path(td)
            _write_header(sd, """
// Some preamble
#define ORT_API_VERSION 27
#define ORT_ML_API_VERSION 12
typedef struct OrtApi OrtApi;
""")
            self.assertEqual(ort_api_version.parse_ort_api_version(sd), 27)

    def test_missing_header_raises(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                ort_api_version.parse_ort_api_version(Path(td))

    def test_missing_macro_raises(self):
        with tempfile.TemporaryDirectory() as td:
            sd = Path(td)
            _write_header(sd, "// no macro here\n")
            with self.assertRaises(ValueError):
                ort_api_version.parse_ort_api_version(sd)

    def test_conflicting_definitions_raise(self):
        with tempfile.TemporaryDirectory() as td:
            sd = Path(td)
            _write_header(sd, "#define ORT_API_VERSION 27\n#define ORT_API_VERSION 28\n")
            with self.assertRaises(ValueError):
                ort_api_version.parse_ort_api_version(sd)


class RequiredApiVersionTests(unittest.TestCase):
    def test_v1_27_1_requires_27(self):
        self.assertEqual(ort_api_version.required_api_version_for_tag("v1.27.1"), 27)

    def test_unknown_tag_raises(self):
        with self.assertRaises(ValueError):
            ort_api_version.required_api_version_for_tag("v9.99.9")


class AssertApiVersionTests(unittest.TestCase):
    def test_assert_passes_on_match(self):
        with tempfile.TemporaryDirectory() as td:
            sd = Path(td)
            _write_header(sd, "#define ORT_API_VERSION 27\n")
            v = ort_api_version.assert_api_version(sd, "v1.27.1")
            self.assertEqual(v, 27)

    def test_assert_exits_on_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            sd = Path(td)
            _write_header(sd, "#define ORT_API_VERSION 21\n")
            with self.assertRaises(SystemExit):
                ort_api_version.assert_api_version(sd, "v1.27.1")


if __name__ == "__main__":
    unittest.main()
