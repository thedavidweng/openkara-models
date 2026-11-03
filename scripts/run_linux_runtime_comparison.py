#!/usr/bin/env python3
"""Compare the Linux x64 candidate runtime against the frozen stable catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from compare_runtimes import verify_archive

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = "https://github.com/microsoft/onnxruntime/releases/download"
HEADERS = {
    "1.27.1": (8828892, "25b1ef1fea1acd210d63f8f24dc870ad6e077795ce1f54876252c6d3803c15af"),
    "1.31.1": (11413655, "c849e36a1ef3e22d4ff217b9d1ce4a3cf5b9467b975610913ae959ad5b16bdc9"),
}


def download(url: str, dest: Path, size: int, sha256: str) -> None:
    with urllib.request.urlopen(url, timeout=120) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out)
    verify_archive(dest, size, sha256)


def unpack_headers(archive: Path, source: Path) -> None:
    target = source / "include/onnxruntime/core/session"
    target.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if len(parts) == 3 and parts[1] == "include" and member.isfile():
                with tar.extractfile(member) as inp, (target / parts[2]).open("wb") as out:
                    shutil.copyfileobj(inp, out)
    if not (target / "onnxruntime_c_api.h").is_file():
        raise ValueError("official archive does not include the C API header")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--candidate", type=Path, help="Native candidate archive from the build")
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
        archives = {}
        for version, (size, digest) in HEADERS.items():
            name = f"onnxruntime-linux-x64-{version}.tgz"
            archive = tmp / name
            download(f"{OFFICIAL}/v{version}/{name}", archive, size, digest)
            unpack_headers(archive, tmp / f"source-{version}")
            archives[version] = archive
        candidate = args.candidate.resolve() if args.candidate else archives["1.31.1"]
        with candidate.open("rb") as file:
            candidate_sha256 = hashlib.file_digest(file, "sha256").hexdigest()
        return subprocess.run([
            sys.executable, str(ROOT / "scripts/compare_runtimes.py"),
            "--stable", str(stable), "--stable-size", str(runtime["byte_size"]),
            "--stable-sha256", runtime["archive_digest"],
            "--stable-source", str(tmp / "source-1.27.1"),
            "--candidate", str(candidate),
            "--candidate-size", str(candidate.stat().st_size),
            "--candidate-sha256", candidate_sha256,
            "--candidate-source", str(tmp / ("source-1.27.1" if args.candidate else "source-1.31.1")),
            "--model", str(model_path), "--report", str(args.report.resolve()),
        ], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
