#!/usr/bin/env python3
"""Compare the Linux x64 candidate runtime against the frozen stable catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import archive_utils
import nuget_package
import ort_api_version
import repackage_nuget_runtime
from compare_runtimes import verify_archive

ROOT = Path(__file__).resolve().parents[1]
# The stable runtime has no headers; its C API headers come from Microsoft's
# signed NuGet package for the same release.
STABLE_HEADERS = ("Microsoft.ML.OnnxRuntime", "1.27.1", 135152698, "9359e46eba4482ded00e678c98f22b68f51bb411d7934f5516d64050edfa3383")


def download(url: str, dest: Path, size: int, sha256: str) -> None:
    with urllib.request.urlopen(url, timeout=120) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out)
    verify_archive(dest, size, sha256)


def stable_headers(cache: Path, source: Path) -> None:
    package_id, version, size, sha256 = STABLE_HEADERS
    nupkg = download_package(package_id, version, size, sha256, cache)
    target = source / Path(ort_api_version.HEADER_REL).parent
    target.mkdir(parents=True)
    with zipfile.ZipFile(nupkg) as package:
        def read(name: str) -> bytes:
            return package.read(f"build/native/include/{name}")

        for name in ort_api_version.required_headers(read("onnxruntime_c_api.h"), read):
            (target / name).write_bytes(read(name))


def download_package(package_id: str, version: str, size: int, sha256: str, cache: Path) -> Path:
    nupkg = nuget_package.download(nuget_package.package_url(package_id, version), cache / f"{package_id}.{version}.nupkg", size, sha256)
    nuget_package.verify_signatures(nupkg, package_id, version)
    return nupkg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--candidate", type=Path, help="Candidate archive (default: repackage the locked NuGet runtime)")
    parser.add_argument("--cache", type=Path, default=ROOT / "ort" / "nuget-cache", help="NuGet download cache")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    pointer = json.loads((ROOT / "catalog/channels/stable.json").read_text())
    manifest = json.loads((ROOT / "catalog/releases" / f"{pointer['release_id']}.json").read_text())
    runtimes = [
        item for item in manifest["artifacts"]["runtimes"]
        if item["target_triple"] == "x86_64-unknown-linux-gnu"
        and not item["deprecation"]["deprecated"]
        and "cpu" in item["runtime"]["execution_providers"]
    ]
    if len(runtimes) != 1 or runtimes[0]["runtime"]["version"] != "v1.27.1":
        raise ValueError("expected one stable Linux x64 CPU runtime at v1.27.1")
    runtime = runtimes[0]
    model = next(
        item for item in manifest["artifacts"]["models"]
        if item["artifact_id"] == "htdemucs.spectral.fp32.onnx"
    )
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        stable = tmp / "stable.tar.gz"
        download(runtime["download_url"], stable, runtime["byte_size"], runtime["archive_digest"])
        if args.model is None:
            model_path = tmp / "model.onnx"
            download(model["download_url"], model_path, model["byte_size"], model["archive_digest"])
        else:
            model_path = args.model.resolve()
        stable_headers(args.cache, tmp / "source-stable")
        if args.candidate:
            candidate = args.candidate.resolve()
        else:
            lock = repackage_nuget_runtime.load_lock()
            nupkg = repackage_nuget_runtime.fetch_verified_package(lock, args.cache)
            target = "x86_64-unknown-linux-gnu"
            candidate = repackage_nuget_runtime.repackage(nupkg, lock, [target], tmp / "candidate")[target]
        archive_utils.unpack_runtime_headers(candidate, tmp / "source-candidate")
        with candidate.open("rb") as file:
            candidate_sha256 = hashlib.file_digest(file, "sha256").hexdigest()
        return subprocess.run([
            sys.executable, str(ROOT / "scripts/compare_runtimes.py"),
            "--stable", str(stable), "--stable-size", str(runtime["byte_size"]),
            "--stable-sha256", runtime["archive_digest"],
            "--stable-source", str(tmp / "source-stable"),
            "--candidate", str(candidate),
            "--candidate-size", str(candidate.stat().st_size),
            "--candidate-sha256", candidate_sha256,
            "--candidate-source", str(tmp / "source-candidate"),
            "--model", str(model_path), "--report", str(args.report.resolve()),
        ], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
