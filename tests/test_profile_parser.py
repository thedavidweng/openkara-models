"""Compile and run the C++ unit tests for the ORT profile JSON parser.

The parser lives in ort/smoke/profile_parser.hpp and is shared between
ort_smoke.cpp (the native smoke harness) and this test program. The test
program (tests/test_profile_parser.cpp) does not link against ORT — it only
exercises the parser with synthetic profile JSON strings.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PARSER_HEADER = ROOT / "ort" / "smoke" / "profile_parser.hpp"
TEST_CPP = ROOT / "tests" / "test_profile_parser.cpp"


def _has_cpp_compiler() -> bool:
    return shutil.which("g++") is not None or shutil.which("clang++") is not None


@pytest.mark.skipif(not _has_cpp_compiler(), reason="no C++ compiler available")
def test_profile_parser() -> None:
    """Compile and run the C++ profile parser tests."""
    # Compile the test program. The parser is header-only so no extra sources
    # are needed. We do not need the ORT C API header for this test.
    compiler = "g++" if shutil.which("g++") else "clang++"
    out_bin = "/tmp/test_profile_parser_bin"
    cmd = [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
           "-I", str(ROOT / "ort" / "smoke"),
           str(TEST_CPP), "-o", out_bin]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, (
        f"compilation failed:\nstdout: {r.stdout}\nstderr: {r.stderr}"
    )

    r = subprocess.run([out_bin], capture_output=True, text=True)
    print(r.stdout)
    assert r.returncode == 0, (
        f"profile parser tests failed:\nstdout: {r.stdout}\nstderr: {r.stderr}"
    )
    assert "ALL PASS" in r.stdout
    # Sanity: the realistic-profile test must count 3 CPU nodes.
    assert "realistic_profile" in r.stdout
