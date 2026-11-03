#!/usr/bin/env python3
"""Generate the v2 catalog from verified runtime and plugin archives."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tarfile
import zipfile
from pathlib import Path

import archive_utils
from validate_catalog import validate_document

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.31.1"
COMMIT = "6af1a932e2286d5c58166085d24c68783033edee"
OFFICIAL = "https://github.com/microsoft/onnxruntime/releases/download/v1.31.1/"
PACKAGES = {
    "aarch64-apple-darwin": ("onnxruntime-osx-arm64-1.31.1.tgz", 36936435, "2e1344a1926ee50eb25b38153cb6992c3567515aef0e677a1a520a1798896d6f"),
    "aarch64-unknown-linux-gnu": ("onnxruntime-linux-aarch64-1.31.1.tgz", 10332381, "5317505e2da99be65f8e2775ba3c2afd6d2961fd78aba1bd8f56159f48820c78"),
    "x86_64-unknown-linux-gnu": ("onnxruntime-linux-x64-1.31.1.tgz", 11413655, "c849e36a1ef3e22d4ff217b9d1ce4a3cf5b9467b975610913ae959ad5b16bdc9"),
    "x86_64-pc-windows-msvc": ("onnxruntime-win-x64-1.31.1.zip", 82929944, "5869e26bb25314375f2d5ef1877e3ea06b1543d6c710e6e4a1ec2f7177ab4b2c"),
}
PLUGINS = {
    "aarch64-apple-darwin": ("https://files.pythonhosted.org/packages/2c/c3/e047d38981e7d6f198f92b488e2c9cd12e45c73017ec851996ea69778922/onnxruntime_ep_webgpu-0.5.0-py3-none-macosx_14_0_universal2.whl", 4864395, "c9a757c7afaf114fde86993420810ff616e2d3046d42ae419bb0c8b9ebcf57fb"),
    "x86_64-pc-windows-msvc": ("https://files.pythonhosted.org/packages/7e/36/9935a5b7851099da3da81d9e04f3021bcc61a5a1039e6c098cf08e5c6607/onnxruntime_ep_webgpu-0.5.0-py3-none-win_amd64.whl", 13608886, "713b94e7b4ce5979a356e19b926ecba2c0d4ba24f857ef4bbb11c75c37cc5b61"),
}


def digest_file(path: Path) -> tuple[int, str]:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return path.stat().st_size, digest


def extracted_digests(path: Path) -> dict[str, dict]:
    result = {}
    total_size = 0

    def add(name, stream):
        nonlocal total_size
        name = archive_utils._check_traversal(name)
        if name in result:
            raise ValueError(f"unsafe or duplicate archive member: {name}")
        digest = hashlib.sha256()
        size = 0
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            size += len(chunk)
            total_size += len(chunk)
            if total_size > archive_utils.MAX_EXTRACTED_SIZE:
                raise ValueError("archive exceeds extracted size limit")
            digest.update(chunk)
        result[name] = {"size": size, "sha256": digest.hexdigest()}

    if path.suffix in (".zip", ".whl"):
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) > archive_utils.MAX_MEMBER_COUNT:
                raise ValueError("archive exceeds member count limit")
            for info in members:
                if not info.is_dir():
                    with archive.open(info) as stream:
                        add(info.filename, stream)
    else:
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
            if len(members) > archive_utils.MAX_MEMBER_COUNT:
                raise ValueError("archive exceeds member count limit")
            for member in members:
                archive_utils._check_traversal(member.name)
                if member.issym() or member.islnk():
                    archive_utils._check_link_escape(member.name, member.linkname, Path("."))
                if member.isfile():
                    with archive.extractfile(member) as stream:
                        add(member.name, stream)
    return result


def generate(packages: Path, intel: Path, release_id: str, created_at: str, producer_commit: str) -> dict:
    old = json.loads((ROOT / "catalog/releases/2026-08-12-001.json").read_text())
    models = [copy.deepcopy(m) for m in old["artifacts"]["models"] if not m["deprecation"]["deprecated"]]
    model_ids = [model["artifact_id"] for model in models]
    runtimes = []
    package_items = list(PACKAGES.items()) + [("x86_64-apple-darwin", (intel.name, *digest_file(intel)))]
    for target, (filename, size, sha256) in package_items:
        path = intel if target == "x86_64-apple-darwin" else packages / filename
        if digest_file(path) != (size, sha256):
            raise ValueError(f"archive integrity mismatch: {path}")
        template = next(r for r in old["artifacts"]["runtimes"] if r["target_triple"] == target and not r["deprecation"]["deprecated"] and "cpu" in r["runtime"]["execution_providers"])
        runtime = copy.deepcopy(template)
        runtime.update({
            "artifact_id": f"onnxruntime-{VERSION}-{target}",
            "filename": filename,
            "download_url": (f"https://github.com/thedavidweng/openkara-models/releases/download/infra-{release_id}/{filename}" if path == intel else OFFICIAL + filename),
            "byte_size": size,
            "archive_digest": sha256,
            "extracted_file_digests": extracted_digests(path),
            "upstream": {"project": "microsoft/onnxruntime", "tag": f"v{VERSION}", "commit_sha": COMMIT},
        })
        runtime.pop("supply_chain", None)
        runtime["runtime"].update({"version": f"v{VERSION}", "ort_c_api_level": "31", "execution_providers": ["cpu", "coreml"] if "apple" in target else ["cpu"], "companion_files": [], "supported_model_artifact_ids": model_ids})
        runtime["toolchain"] = {"image": "macos-latest" if path == intel else "Microsoft official release", "compiler": "Apple clang" if path == intel else "Microsoft official build", "cmake_flags": []}
        runtimes.append(runtime)

    plugins = []
    for target, (url, size, sha256) in PLUGINS.items():
        filename = url.rsplit("/", 1)[-1]
        path = packages / filename
        if digest_file(path) != (size, sha256):
            raise ValueError(f"archive integrity mismatch: {path}")
        plugins.append({"artifact_id": f"onnxruntime-webgpu-0.5.0-{target}", "target_triple": target, "provider": "webgpu", "version": "0.5.0", "filename": filename, "download_url": url, "byte_size": size, "archive_digest": sha256, "extracted_file_digests": extracted_digests(path)})

    runtime_ids = [runtime["artifact_id"] for runtime in runtimes]
    for model in models:
        model["model"]["compatible_runtime_ids"] = runtime_ids
    compatibility = [
        {"model_artifact_id": mid, "runtime_artifact_id": runtime["artifact_id"], "target_triple": runtime["target_triple"], "execution_provider": provider, "status": "supported"}
        for mid in model_ids for runtime in runtimes for provider in runtime["runtime"]["execution_providers"]
    ]
    manifest = {"schema_version": "openkara.catalog/release-v2", "release_id": release_id, "generation": old["generation"] + 1, "created_at": created_at, "producer": {"repo": "thedavidweng/openkara-models", "commit_sha": producer_commit, "workflow": "Intel macOS runtime", "run_id": "manual"}, "artifacts": {"models": models, "runtimes": runtimes, "bundles": [], "plugins": plugins}, "compatibility": compatibility}
    validation = validate_document(manifest)
    if not validation.ok:
        raise ValueError(f"invalid catalog: {validation}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--intel", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = generate(args.packages, args.intel, args.release_id, args.created_at, args.producer_commit)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
