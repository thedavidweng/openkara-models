#!/usr/bin/env python3
"""Select CodeQL languages from pull request paths."""

import json
import sys
from pathlib import Path

ALL = [
    {"language": "actions", "queries": ""},
    {"language": "c-cpp", "queries": ""},
    {"language": "python", "queries": "security-extended"},
]


def classify(files: list[str], event: str) -> list[dict[str, str]]:
    if event != "pull_request" or any(f.startswith(".github/workflows/") for f in files):
        return ALL
    code = [f for f in files if not f.endswith(".md")]
    selected = set()
    for file in code:
        path = Path(file)
        if path.suffix == ".py" or path.name.startswith("requirements"):
            selected.add("python")
        if file.startswith("ort/") or path.suffix in {".c", ".cc", ".cpp", ".h", ".hpp", ".cmake"}:
            selected.add("c-cpp")
        if file.startswith(".github/actions/"):
            selected.add("actions")
    return [entry for entry in ALL if entry["language"] in selected]


if __name__ == "__main__":
    print(json.dumps({"include": classify(Path(sys.argv[1]).read_text().splitlines(), sys.argv[2])}, separators=(",", ":")))
