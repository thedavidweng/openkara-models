#!/usr/bin/env python3
"""Parse the ORT C API version from the pinned source header.

The ORT C API version is defined as ``ORT_API_VERSION`` in
``include/onnxruntime/core/session/onnxruntime_c_api.h`` of the ORT source
tree. It is no longer manually maintained in ``ort/source-lock.json``; instead
it is derived from the pinned source at build time and recorded in
``build-manifest.json`` and ``provenance.json``.

For the pinned ORT v1.27.1 source the parsed value must be ``27``.

This module is shared by ``build_runtime.py`` (which records the value in the
build manifest) and ``verify_runtime_package.py`` (which parses the pinned
source header and compares it to the package build manifest).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# The header path relative to the ORT source root.
HEADER_REL = "include/onnxruntime/core/session/onnxruntime_c_api.h"

# Regex matching:  #define ORT_API_VERSION 27
_API_VERSION_RE = re.compile(
    r"^\s*#\s*define\s+ORT_API_VERSION\s+(\d+)\s*$",
    re.MULTILINE,
)


def parse_ort_api_version(source_dir: Path) -> int:
    """Parse and return the ORT_API_VERSION integer from the pinned header.

    Raises FileNotFoundError if the header is missing and ValueError if the
    macro is not found or is defined more than once with conflicting values.
    """
    header = source_dir / HEADER_REL
    if not header.is_file():
        raise FileNotFoundError(
            f"ORT C API header not found: {header}. "
            "Ensure the ORT source is checked out at the pinned commit."
        )
    text = header.read_text(encoding="utf-8", errors="replace")
    matches = _API_VERSION_RE.findall(text)
    if not matches:
        raise ValueError(
            f"ORT_API_VERSION macro not found in {header}"
        )
    values = {int(m) for m in matches}
    if len(values) != 1:
        raise ValueError(
            f"ORT_API_VERSION defined multiple times with conflicting values "
            f"in {header}: {sorted(values)}"
        )
    return values.pop()


def required_api_version_for_tag(tag: str) -> int:
    """Return the required ORT_API_VERSION for a pinned upstream tag.

    The source lock pins a specific ORT release; the API version derived from
    that release's header must match the expected value. For v1.27.1 the C API
    version is 27.
    """
    expected = {
        "v1.27.1": 27,
    }
    if tag not in expected:
        raise ValueError(
            f"no required ORT_API_VERSION recorded for upstream tag {tag}; "
            f"add it to ort_api_version.required_api_version_for_tag"
        )
    return expected[tag]


def assert_api_version(source_dir: Path, tag: str, *, label: str = "build") -> int:
    """Parse the header, assert it matches the required value for ``tag``.

    Returns the parsed version on success. Exits the process with a non-zero
    status on mismatch (for CLI use) — callers that prefer an exception should
    call ``parse_ort_api_version`` and ``required_api_version_for_tag``
    directly.
    """
    parsed = parse_ort_api_version(source_dir)
    required = required_api_version_for_tag(tag)
    if parsed != required:
        print(
            f"ERROR [{label}]: ORT_API_VERSION mismatch for {tag}: "
            f"header={parsed} required={required}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return parsed


if __name__ == "__main__":
    # CLI: python scripts/ort_api_version.py <source_dir> <tag>
    if len(sys.argv) != 3:
        print("usage: ort_api_version.py <source_dir> <tag>", file=sys.stderr)
        raise SystemExit(2)
    v = assert_api_version(Path(sys.argv[1]), sys.argv[2])
    print(f"ORT_API_VERSION={v}")
