"""Tests for the runtime catalog entry generator."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _make_archive(path: Path, files: dict[str, bytes]) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    path.write_bytes(buf.getvalue())


def _make_build_manifest(target: str = "x86_64-unknown-linux-gnu",
                         reduced: bool = False) -> dict:
    lock = json.loads((ROOT / "ort" / "source-lock.json").read_text())
    bm = {
        "schema_version": "openkara.ort-build-manifest/v1",
        "target": target,
        "upstream": lock["upstream"],
        "build": {
            "target": target,
            "commit_sha": lock["upstream"]["commit_sha"],
            "tag": lock["upstream"]["tag"],
            "submodule_state": {},
            "cmake_args": lock["targets"][target]["cmake_args"],
            "build_os": "Linux 6.0",
            "build_arch": "x86_64",
            "ort_api_version": 27,
        },
        "toolchain": lock["toolchain"],
        "c_api_level": {
            "ort_api_version": 27,
            "ort_api_version_note": "Parsed at build time from ORT_API_VERSION.",
        },
        "files": {
            "libonnxruntime.so": {"size": 100, "sha256": "a" * 64},
            "LICENSE.onnxruntime": {"size": 10, "sha256": "b" * 64},
        },
        "supply_chain": {
            "sbom": {"path": "sbom.spdx.json", "size": 500, "sha256": "c" * 64},
            "provenance": {"path": "provenance.json", "size": 300, "sha256": "d" * 64},
        },
    }
    if reduced:
        bm["build"]["reduced_build"] = True
        bm["build"]["ops_config_sha256"] = "e" * 64
    return bm


def test_parse_target_linux() -> None:
    sys.path.insert(0, str(SCRIPTS))
    import generate_runtime_catalog_entries as gen
    target, arch, os_name, is_reduced = gen._parse_target(
        "onnxruntime-1.27.1-openkara-x86_64-unknown-linux-gnu.tar.gz"
    )
    assert target == "x86_64-unknown-linux-gnu"
    assert arch == "x86_64"
    assert os_name == "linux"
    assert is_reduced is False


def test_parse_target_macos_arm() -> None:
    sys.path.insert(0, str(SCRIPTS))
    import generate_runtime_catalog_entries as gen
    target, arch, os_name, is_reduced = gen._parse_target(
        "onnxruntime-1.27.1-openkara-aarch64-apple-darwin.tar.gz"
    )
    assert target == "aarch64-apple-darwin"
    assert arch == "aarch64"
    assert os_name == "macos"
    assert is_reduced is False


def test_parse_target_windows() -> None:
    sys.path.insert(0, str(SCRIPTS))
    import generate_runtime_catalog_entries as gen
    target, arch, os_name, is_reduced = gen._parse_target(
        "onnxruntime-1.27.1-openkara-x86_64-pc-windows-msvc.zip"
    )
    assert target == "x86_64-pc-windows-msvc"
    assert arch == "x86_64"
    assert os_name == "windows"
    assert is_reduced is False


def test_parse_target_reduced() -> None:
    sys.path.insert(0, str(SCRIPTS))
    import generate_runtime_catalog_entries as gen
    target, arch, os_name, is_reduced = gen._parse_target(
        "onnxruntime-1.27.1-openkara-x86_64-unknown-linux-gnu-reduced.tar.gz"
    )
    assert target == "x86_64-unknown-linux-gnu"
    assert is_reduced is True


def test_build_entry_full(tmp_path: Path) -> None:
    sys.path.insert(0, str(SCRIPTS))
    import generate_runtime_catalog_entries as gen
    archive = tmp_path / "onnxruntime-1.27.1-openkara-x86_64-unknown-linux-gnu.tar.gz"
    _make_archive(archive, {
        "libonnxruntime.so": b"x" * 100,
        "build-manifest.json": json.dumps(_make_build_manifest(), sort_keys=True).encode(),
    })
    lock = json.loads((ROOT / "ort" / "source-lock.json").read_text())
    entry = gen._build_entry(archive, lock, "2026-07-20-001", ["htdemucs.balanced.fp32.onnx"])
    assert entry["kind"] == "runtime"
    assert entry["target_triple"] == "x86_64-unknown-linux-gnu"
    assert entry["arch"] == "x86_64"
    assert entry["os"] == "linux"
    assert entry["runtime"]["version"] == lock["upstream"]["tag"]
    assert entry["runtime"]["ort_c_api_level"] == "27"
    assert "cpu" in entry["runtime"]["execution_providers"]
    assert entry["runtime"]["supported_model_artifact_ids"] == ["htdemucs.balanced.fp32.onnx"]
    assert entry["runtime"]["reduced_build"] is False
    assert entry["_digest"]["archive_digest"] != ""
    assert entry["_digest"]["byte_size"] > 0
    assert entry["supply_chain"]["sbom"]["sha256"] == "c" * 64
    assert entry["toolchain"]["cmake_flags"] == lock["targets"]["x86_64-unknown-linux-gnu"]["cmake_args"]


def test_build_entry_reduced(tmp_path: Path) -> None:
    sys.path.insert(0, str(SCRIPTS))
    import generate_runtime_catalog_entries as gen
    archive = tmp_path / "onnxruntime-1.27.1-openkara-x86_64-unknown-linux-gnu-reduced.tar.gz"
    _make_archive(archive, {
        "libonnxruntime.so": b"x" * 80,
        "build-manifest.json": json.dumps(_make_build_manifest(reduced=True), sort_keys=True).encode(),
    })
    lock = json.loads((ROOT / "ort" / "source-lock.json").read_text())
    entry = gen._build_entry(archive, lock, None, None)
    assert entry["runtime"]["reduced_build"] is True
    assert entry["runtime"]["ops_config_sha256"] == "e" * 64
    assert "-reduced" in entry["artifact_id"]


def test_build_entry_download_url(tmp_path: Path) -> None:
    sys.path.insert(0, str(SCRIPTS))
    import generate_runtime_catalog_entries as gen
    archive = tmp_path / "onnxruntime-1.27.1-openkara-x86_64-unknown-linux-gnu.tar.gz"
    _make_archive(archive, {
        "libonnxruntime.so": b"x",
        "build-manifest.json": json.dumps(_make_build_manifest(), sort_keys=True).encode(),
    })
    lock = json.loads((ROOT / "ort" / "source-lock.json").read_text())
    entry = gen._build_entry(archive, lock, "2026-07-20-001", None)
    assert "infra-2026-07-20-001" in entry["download_url"]
    assert entry["download_url"].endswith(archive.name)


def test_cli_help() -> None:
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "generate_runtime_catalog_entries.py"), "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert "--release" in r.stdout
    assert "--compatible-models" in r.stdout
