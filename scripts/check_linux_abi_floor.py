#!/usr/bin/env python3
"""Verify a packaged Linux runtime's ABI floor.

The shared library's glibc symbol requirements are set by the OS the build
ran on. OpenKara's Linux release baseline is ubuntu-22.04 (glibc 2.35,
GLIBCXX 3.4.30); a runtime referencing newer versioned symbols fails to
dlopen there. This gate parses the actual versioned-symbol references out of
the packaged library so a runner-image bump can never silently raise the
floor again.

Usage:
  python scripts/check_linux_abi_floor.py \
      --archive ort/packages/onnxruntime-...-x86_64-unknown-linux-gnu.tar.gz \
      --library libonnxruntime.so \
      --max-glibc 2.35 --max-glibcxx 3.4.30 --max-cxxabi 1.3.13
"""

from __future__ import annotations

import argparse
import re
import sys
import tarfile


def parse_version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--library", default="libonnxruntime.so")
    parser.add_argument("--max-glibc", required=True)
    parser.add_argument("--max-glibcxx", required=True)
    parser.add_argument("--max-cxxabi", required=True)
    args = parser.parse_args()

    with tarfile.open(args.archive, "r:gz") as archive:
        member = None
        for candidate in archive.getmembers():
            name = candidate.name.split("/")[-1]
            if name == args.library:
                member = candidate
                break
        if member is None:
            print(f"error: {args.library} not found in {args.archive}")
            return 1
        extracted = archive.extractfile(member)
        assert extracted is not None
        payload = extracted.read()

    floors = {
        "GLIBC": parse_version(args.max_glibc),
        "GLIBCXX": parse_version(args.max_glibcxx),
        "CXXABI": parse_version(args.max_cxxabi),
    }
    patterns = {
        "GLIBC": rb"GLIBC_(\d+\.\d+)",
        "GLIBCXX": rb"GLIBCXX_(\d+\.\d+\.\d+)",
        "CXXABI": rb"CXXABI_(\d+\.\d+(?:\.\d+)?)",
    }

    failures = []
    for family, pattern in patterns.items():
        referenced = sorted(
            {parse_version(m.decode()) for m in re.findall(pattern, payload)}
        )
        if not referenced:
            continue
        highest = referenced[-1]
        limit = floors[family]
        rendered = ".".join(str(part) for part in highest)
        allowed = ".".join(str(part) for part in limit)
        if highest > limit:
            failures.append(
                f"{family}_{rendered} referenced, but the release baseline only provides {family}_{allowed}"
            )
        else:
            print(f"ok: highest {family} reference {rendered} <= floor {allowed}")

    if failures:
        print("ABI floor check FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("ABI floor check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
