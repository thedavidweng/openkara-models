#!/usr/bin/env python3
"""Parse the ORT C API version from the pinned source header.

The ORT C API version is defined as ``ORT_API_VERSION`` in
``include/onnxruntime/core/session/onnxruntime_c_api.h`` of the ORT source
tree. It is no longer manually maintained in ``ort/source-lock.json``; instead
it is derived from the pinned source at build time and recorded in
``build-manifest.json`` and ``provenance.json``.

There is no tag-to-API table: the immutable upstream header is the authority.

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


def required_headers(header: bytes) -> list[str]:
    """The main API header includes the EP API header on newer ORT versions."""
    names = ["onnxruntime_c_api.h"]
    if re.search(rb'^\s*#\s*include\s*"onnxruntime_ep_c_api.h"', header, re.MULTILINE):
        names.append("onnxruntime_ep_c_api.h")
    return names

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
    return parse_header(header.read_text(encoding="utf-8"))


def parse_header(text: str) -> int:
    matches = _API_VERSION_RE.findall(text)
    if not matches:
        raise ValueError(
            f"ORT_API_VERSION macro not found in header"
        )
    values = {int(m) for m in matches}
    if len(values) != 1:
        raise ValueError(
            f"ORT_API_VERSION defined multiple times with conflicting values "
            f"in header: {sorted(values)}"
        )
    return values.pop()



def pinned_header(upstream: dict, name: str = "onnxruntime_c_api.h") -> bytes:
    """Fetch only the small C header at the exact locked upstream commit."""
    from urllib.request import urlopen

    commit = upstream["commit_sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("upstream commit_sha must be a full immutable commit")
    if upstream["repo"] != "microsoft/onnxruntime":
        raise ValueError("unexpected ORT upstream repository")
    if name not in ("onnxruntime_c_api.h", "onnxruntime_ep_c_api.h"):
        raise ValueError("unexpected ORT API header")
    rel = str(Path(HEADER_REL).with_name(name)).replace("\\", "/")
    url = f"https://raw.githubusercontent.com/microsoft/onnxruntime/{commit}/{rel}"
    with urlopen(url, timeout=30) as response:
        return response.read()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: ort_api_version.py <source_dir>", file=sys.stderr)
        raise SystemExit(2)
    print(f"ORT_API_VERSION={parse_ort_api_version(Path(sys.argv[1]))}")
