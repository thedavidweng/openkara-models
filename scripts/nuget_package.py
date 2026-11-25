#!/usr/bin/env python3
"""Download, pin, and signature-verify Microsoft ONNX Runtime NuGet packages.

The upstream GitHub repository is not a dependency: nuget.org serves the same
Microsoft-signed native binaries, and every package carries an author
signature (Microsoft Corporation) plus a nuget.org repository countersignature.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

FLAT_CONTAINER = "https://api.nuget.org/v3-flatcontainer"
AUTHOR_SUBJECT = "CN=Microsoft Corporation, O=Microsoft Corporation, L=Redmond, S=Washington, C=US"
REPOSITORY_SUBJECT = "CN=NuGet.org Repository by Microsoft, O=NuGet.org Repository by Microsoft, L=Redmond, S=Washington, C=US"


def package_url(package_id: str, version: str) -> str:
    lower = package_id.lower()
    return f"{FLAT_CONTAINER}/{lower}/{version}/{lower}.{version}.nupkg"


def sha256_file(path: Path) -> tuple[int, str]:
    with path.open("rb") as stream:
        return path.stat().st_size, hashlib.file_digest(stream, "sha256").hexdigest()


def download(url: str, dest: Path, size: int, sha256: str) -> Path:
    """Download ``url`` to ``dest`` unless a verified copy is already there."""
    if dest.is_file() and sha256_file(dest) == (size, sha256):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=300) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out)
    actual = sha256_file(partial)
    if actual != (size, sha256):
        partial.unlink()
        raise ValueError(f"{url}: expected {size} bytes sha256 {sha256}, got {actual[0]} bytes sha256 {actual[1]}")
    partial.replace(dest)
    return dest


def _signature_blocks(output: str) -> dict[str, str]:
    blocks: dict[str, str] = {}
    parts = re.split(r"^Signature type: (\w+)\s*$", output, flags=re.MULTILINE)
    for kind, body in zip(parts[1::2], parts[2::2]):
        blocks[kind] = body
    return blocks


def _first_subject(body: str) -> str:
    match = re.search(r"Subject Name: (.+)", body)
    return match.group(1).strip() if match else ""


def check_verify_output(output: str, package_id: str, version: str) -> list[str]:
    """Return problems in ``dotnet nuget verify --all -v n`` output."""
    problems = []
    if f"Successfully verified package '{package_id}.{version}'." not in output:
        problems.append("dotnet nuget verify did not report success")
    blocks = _signature_blocks(output)
    author = blocks.get("Author")
    if author is None or _first_subject(author) != AUTHOR_SUBJECT:
        problems.append(f"author signature is not from {AUTHOR_SUBJECT}")
    repository = blocks.get("Repository")
    if repository is None or _first_subject(repository) != REPOSITORY_SUBJECT:
        problems.append(f"repository countersignature is not from {REPOSITORY_SUBJECT}")
    elif "Service index: https://api.nuget.org/v3/index.json" not in repository:
        problems.append("repository countersignature is not from the nuget.org service index")
    return problems


def verify_signatures(nupkg: Path, package_id: str, version: str) -> str:
    """Run ``dotnet nuget verify --all`` and require Microsoft + nuget.org signers."""
    dotnet = shutil.which("dotnet")
    if dotnet is None:
        raise RuntimeError("dotnet SDK is required for `dotnet nuget verify`")
    result = subprocess.run(
        [dotnet, "nuget", "verify", "--all", str(nupkg), "-v", "n"],
        capture_output=True, text=True, check=False,
        env={**os.environ, "DOTNET_CLI_TELEMETRY_OPTOUT": "1", "DOTNET_NOLOGO": "1"},
    )
    output = result.stdout + result.stderr
    problems = check_verify_output(output, package_id, version)
    if result.returncode != 0 or problems:
        raise RuntimeError("NuGet signature verification failed:\n  " + "\n  ".join(problems or ["non-zero exit"]) + f"\n{output}")
    return output

