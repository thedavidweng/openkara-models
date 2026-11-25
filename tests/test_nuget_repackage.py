"""Tests for the NuGet runtime repackager."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import nuget_package  # noqa: E402
import repackage_nuget_runtime as repackage  # noqa: E402

VERIFY_OK = """Verifying Microsoft.ML.OnnxRuntime.1.32.0
Signature type: Author
Verifying the author primary signature with certificate:
  Subject Name: CN=Microsoft Corporation, O=Microsoft Corporation, L=Redmond, S=Washington, C=US
Timestamp: 11/13/2026 00:36:55
  Subject Name: CN=DigiCert SHA256 RSA4096 Timestamp Responder 2026 1, O="DigiCert, Inc.", C=US

Signature type: Repository
Service index: https://api.nuget.org/v3/index.json
Verifying the repository countersignature with certificate:
  Subject Name: CN=NuGet.org Repository by Microsoft, O=NuGet.org Repository by Microsoft, L=Redmond, S=Washington, C=US

Successfully verified package 'Microsoft.ML.OnnxRuntime.1.32.0'.
"""


def test_verify_output_requires_microsoft_author_and_nuget_repository():
    assert nuget_package.check_verify_output(VERIFY_OK, "Microsoft.ML.OnnxRuntime", "1.32.0") == []
    forged = VERIFY_OK.replace("CN=Microsoft Corporation, O=Microsoft Corporation", "CN=Someone Else, O=Someone Else")
    assert nuget_package.check_verify_output(forged, "Microsoft.ML.OnnxRuntime", "1.32.0")
    unsigned = VERIFY_OK.split("Signature type: Repository")[0] + "Successfully verified package 'Microsoft.ML.OnnxRuntime.1.32.0'.\n"
    assert nuget_package.check_verify_output(unsigned, "Microsoft.ML.OnnxRuntime", "1.32.0")
    failed = VERIFY_OK.replace("Successfully verified", "Failed to verify")
    assert nuget_package.check_verify_output(failed, "Microsoft.ML.OnnxRuntime", "1.32.0")


def test_lock_binaries_match_the_pre_outage_catalog():
    """Linux and Windows binaries are byte-identical to the GitHub release archives."""
    lock = repackage.load_lock()
    current = json.loads((ROOT / "catalog/releases/2026-11-17-001.json").read_text())
    declared = {
        name: digest
        for runtime in current["artifacts"]["runtimes"]
        for name, digest in runtime["extracted_file_digests"].items()
    }
    for target, config in lock["targets"].items():
        for source in config["files"].values():
            member = lock["members"][source]
            if target == "aarch64-apple-darwin":
                assert member["apple_team_id"] == "UBF8T346G9"
                continue
            assert declared[member["pre_outage_member"]] == {"size": member["size"], "sha256": member["sha256"]}


def _fake_package(tmp_path: Path) -> tuple[Path, dict]:
    files = {
        "LICENSE": b"license",
        "ThirdPartyNotices.txt": b"notices",
        "build/native/include/onnxruntime_c_api.h": b'#define ORT_API_VERSION 32\n#include "onnxruntime_ep_c_api.h"\n',
        "build/native/include/onnxruntime_ep_c_api.h": b"// ep\n",
        "runtimes/linux-x64/native/libonnxruntime.so": b"\x7fELF runtime",
        "runtimes/win-x64/native/onnxruntime.dll": b"MZ runtime",
    }
    nupkg = tmp_path / "pkg.nupkg"
    with zipfile.ZipFile(nupkg, "w") as package:
        for name, data in files.items():
            package.writestr(name, data)
    lock = {
        "package": {"id": "Microsoft.ML.OnnxRuntime", "version": "1.32.0", "url": "u", "size": 1, "sha256": "0", "signers": {}},
        "upstream": {"project": "microsoft/onnxruntime", "tag": "v1.32.0", "commit_sha": "f" * 40},
        "members": {name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in files.items()},
        "common": {"LICENSE": "LICENSE", "ThirdPartyNotices.txt": "ThirdPartyNotices.txt", "include": "build/native/include"},
        "targets": {
            "x86_64-unknown-linux-gnu": {
                "root": "onnxruntime-linux-x64-1.32.0", "archive_format": "tar.gz",
                "files": {"lib/libonnxruntime.so.1.32.0": "runtimes/linux-x64/native/libonnxruntime.so"},
                "symlinks": {"lib/libonnxruntime.so.1": "libonnxruntime.so.1.32.0", "lib/libonnxruntime.so": "libonnxruntime.so.1"},
            },
            "x86_64-pc-windows-msvc": {
                "root": "onnxruntime-win-x64-1.32.0", "archive_format": "zip",
                "files": {"lib/onnxruntime.dll": "runtimes/win-x64/native/onnxruntime.dll"},
                "symlinks": {},
            },
        },
    }
    return nupkg, lock


def test_repackage_writes_canonical_deterministic_archives(tmp_path):
    nupkg, lock = _fake_package(tmp_path)
    first = repackage.repackage(nupkg, lock, sorted(lock["targets"]), tmp_path / "a")
    second = repackage.repackage(nupkg, lock, sorted(lock["targets"]), tmp_path / "b")
    for target in lock["targets"]:
        assert first[target].read_bytes() == second[target].read_bytes()
    with tarfile.open(first["x86_64-unknown-linux-gnu"]) as archive:
        members = {m.name: m for m in archive.getmembers()}
        root = "onnxruntime-linux-x64-1.32.0"
        assert set(members) == {f"{root}/{name}" for name in (
            "LICENSE", "ThirdPartyNotices.txt", "build-manifest.json",
            "include/onnxruntime_c_api.h", "include/onnxruntime_ep_c_api.h",
            "lib/libonnxruntime.so.1.32.0", "lib/libonnxruntime.so.1", "lib/libonnxruntime.so",
        )}
        assert members[f"{root}/lib/libonnxruntime.so"].linkname == "libonnxruntime.so.1"
        manifest = json.load(archive.extractfile(f"{root}/build-manifest.json"))
    assert manifest["files"]["lib/libonnxruntime.so.1.32.0"]["source"] == "runtimes/linux-x64/native/libonnxruntime.so"
    with zipfile.ZipFile(first["x86_64-pc-windows-msvc"]) as archive:
        assert "onnxruntime-win-x64-1.32.0/lib/onnxruntime.dll" in archive.namelist()


def test_repackage_rejects_a_member_that_differs_from_the_lock(tmp_path):
    nupkg, lock = _fake_package(tmp_path)
    lock["members"]["runtimes/linux-x64/native/libonnxruntime.so"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="libonnxruntime.so"):
        repackage.repackage(nupkg, lock, ["x86_64-unknown-linux-gnu"], tmp_path / "out")


def test_download_rejects_wrong_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout: io.BytesIO(b"tampered"))
    with pytest.raises(ValueError, match="expected"):
        nuget_package.download("https://api.nuget.org/x.nupkg", tmp_path / "x.nupkg", 3, "0" * 64)
    assert not (tmp_path / "x.nupkg").exists()
