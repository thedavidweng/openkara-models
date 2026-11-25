#!/usr/bin/env python3
"""Generate the v2 catalog from verified runtime and plugin archives.

Runtimes for Linux, Windows, and Apple Silicon macOS are the Microsoft-signed
binaries from the Microsoft.ML.OnnxRuntime NuGet package, repackaged by
``scripts/repackage_nuget_runtime.py`` and hosted on this repository's
``infra-<release-id>`` release. The Intel macOS runtime built from source by
this repository is carried forward unchanged from the current stable-v2
release.
"""

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
NUGET_LOCK = ROOT / "ort/nuget-lock.json"
RELEASE_BASE = "https://github.com/thedavidweng/openkara-models/releases/download"
INTEL = "x86_64-apple-darwin"
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


def _current_release() -> tuple[dict, dict]:
    pointer = json.loads((ROOT / "catalog/channels/stable-v2.json").read_text())
    return pointer, json.loads((ROOT / "catalog/releases" / f'{pointer["release_id"]}.json').read_text())


def intel_download_url() -> str:
    """URL of the Intel macOS runtime the next release carries forward."""
    return _intel_runtime(_current_release()[1], None)["download_url"]


def _intel_runtime(current: dict, intel: Path | None) -> dict:
    """Carry the current Intel macOS runtime forward, byte for byte."""
    runtime = copy.deepcopy(next(r for r in current["artifacts"]["runtimes"] if r["target_triple"] == INTEL and not r["deprecation"]["deprecated"]))
    if intel is not None and digest_file(intel) != (runtime["byte_size"], runtime["archive_digest"]):
        raise ValueError(f"archive integrity mismatch: {intel}")
    return runtime


def _nuget_runtime(template: dict, lock: dict, target: str, path: Path, release_id: str) -> dict:
    size, sha256 = digest_file(path)
    version = lock["package"]["version"]
    runtime = copy.deepcopy(template)
    runtime.update({
        "artifact_id": f"onnxruntime-{version}-{target}-{sha256[:16]}",
        "filename": path.name,
        "download_url": f"{RELEASE_BASE}/infra-{release_id}/{path.name}",
        "byte_size": size,
        "archive_digest": sha256,
        "extracted_file_digests": extracted_digests(path),
        "upstream": {"project": lock["upstream"]["project"], "tag": lock["upstream"]["tag"], "commit_sha": lock["upstream"]["commit_sha"]},
    })
    runtime.pop("supply_chain", None)
    runtime["toolchain"] = {"image": f"nuget.org {lock['package']['id']} {version}", "compiler": "Microsoft official build", "cmake_flags": []}
    return runtime


def generate(packages: Path, intel: Path | None, release_id: str, created_at: str, producer_commit: str, run_id: str = "manual") -> dict:
    import repackage_nuget_runtime

    lock = json.loads(NUGET_LOCK.read_text())
    pointer, current = _current_release()
    models = [copy.deepcopy(m) for m in current["artifacts"]["models"] if not m["deprecation"]["deprecated"]]
    model_ids = [model["artifact_id"] for model in models]
    templates = {r["target_triple"]: r for r in current["artifacts"]["runtimes"] if not r["deprecation"]["deprecated"]}
    runtimes = []
    for target in sorted(lock["targets"]):
        path = packages / repackage_nuget_runtime.archive_name(lock, target)
        runtimes.append(_nuget_runtime(templates[target], lock, target, path, release_id))
    runtimes.append(_intel_runtime(current, intel))
    for runtime in runtimes:
        runtime["runtime"]["supported_model_artifact_ids"] = model_ids

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
    manifest = {"schema_version": "openkara.catalog/release-v2", "release_id": release_id, "generation": pointer["generation"] + 1, "created_at": created_at, "producer": {"repo": "thedavidweng/openkara-models", "commit_sha": producer_commit, "workflow": "Publish v2 catalog", "run_id": run_id}, "artifacts": {"models": models, "runtimes": runtimes, "bundles": [], "plugins": plugins}, "compatibility": compatibility}
    validation = validate_document(manifest)
    if not validation.ok:
        raise ValueError(f"invalid catalog: {validation}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--intel", type=Path, help="Carried-forward Intel macOS archive to re-verify")
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--run-id", default="manual")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = generate(args.packages, args.intel, args.release_id, args.created_at, args.producer_commit, args.run_id)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
