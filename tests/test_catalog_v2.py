import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import publish_catalog_release as publisher
from publish_catalog_release import _verify_plugin_asset
from generate_catalog_v2 import extracted_digests
from validate_catalog import validate_document


def test_v2_channel_has_its_own_version():
    pointer = json.loads((ROOT / "catalog/channels/stable.json").read_text())
    pointer["schema_version"] = "openkara.catalog/channel-v2"
    assert validate_document(pointer, schema="channel").ok
    pointer["schema_version"] = "openkara.catalog/channel-v3"
    assert not validate_document(pointer, schema="channel").ok


def test_plugin_download_checks_bytes_and_size():
    data = b"official wheel bytes"
    plugin = {
        "artifact_id": "webgpu-test",
        "download_url": "https://files.pythonhosted.org/packages/test.whl",
        "byte_size": len(data),
        "archive_digest": hashlib.sha256(data).hexdigest(),
    }
    with patch("urllib.request.urlopen", return_value=io.BytesIO(data)):
        assert _verify_plugin_asset(plugin) == []
    plugin["archive_digest"] = "0" * 64
    with patch("urllib.request.urlopen", return_value=io.BytesIO(data)):
        assert _verify_plugin_asset(plugin)


def test_v2_release_uses_v1_artifact_shapes_and_validates_plugins():
    release = json.loads((ROOT / "catalog/releases/2026-08-12-001.json").read_text())
    release["schema_version"] = "openkara.catalog/release-v2"
    release["artifacts"] = {"models": [], "runtimes": [], "bundles": [], "plugins": []}
    release["compatibility"] = []
    release.pop("gates", None)
    assert validate_document(release, schema="release").ok
    release["artifacts"]["plugins"].append({"artifact_id": "incomplete"})
    assert not validate_document(release, schema="release").ok


def test_catalog_generation_rejects_escaping_archive_link(tmp_path):
    archive_path = tmp_path / "runtime.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        entry = tarfile.TarInfo("lib/libonnxruntime.dylib")
        entry.type = tarfile.SYMTYPE
        entry.linkname = "../../outside"
        archive.addfile(entry)
    import pytest
    with pytest.raises(ValueError, match="traversal target"):
        extracted_digests(archive_path)


def test_v2_pointer_advances_without_modifying_legacy_pointer(tmp_path, monkeypatch):
    channels = tmp_path / "channels"
    channels.mkdir()
    original = (ROOT / "catalog/channels/stable.json").read_bytes()
    (channels / "stable.json").write_bytes(original)
    manifest = {
        "release_id": "2026-11-02-001",
        "generation": 14,
        "created_at": "2026-11-02T00:00:00Z",
        "producer": {"repo": "thedavidweng/openkara-models"},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(publisher, "CHANNELS_DIR", channels)
    assert publisher._advance_channel_pointer(manifest, manifest_path, "stable-v2") == []
    assert (channels / "stable.json").read_bytes() == original
    pointer = json.loads((channels / "stable-v2.json").read_text())
    assert pointer["schema_version"] == "openkara.catalog/channel-v2"
    assert pointer["previous_release_id"] == "2026-08-12-001"


def test_v2_release_is_rejected_on_legacy_candidate_channel(tmp_path, monkeypatch, capsys):
    release_id = "2026-11-02-001"
    (tmp_path / f"{release_id}.json").write_text('{"schema_version":"openkara.catalog/release-v2"}')
    monkeypatch.setattr(publisher, "RELEASES_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["publish_catalog_release.py", "--release", release_id, "--channel", "candidate"])
    assert publisher.main() == 1
    assert "v2 releases require stable-v2" in capsys.readouterr().err


def _fake_nuget_archives(packages, generator):
    import repackage_nuget_runtime

    lock = json.loads(generator.NUGET_LOCK.read_text())
    for target in lock["targets"]:
        path = packages / repackage_nuget_runtime.archive_name(lock, target)
        data = f"runtime for {target}".encode()
        if path.suffix == ".zip":
            import zipfile
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(f"{lock['targets'][target]['root']}/lib/onnxruntime.dll", data)
        else:
            with tarfile.open(path, "w:gz") as archive:
                entry = tarfile.TarInfo(f"{lock['targets'][target]['root']}/lib/libonnxruntime.bin")
                entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))
    return lock


def test_generated_catalog_hosts_nuget_runtimes_and_carries_intel_forward(tmp_path, monkeypatch):
    import generate_catalog_v2 as generator

    lock = _fake_nuget_archives(tmp_path, generator)
    monkeypatch.setattr(generator, "PLUGINS", {})
    result = generator.generate(tmp_path, None, "2026-11-25-001", "2026-11-25T00:00:00Z", "a" * 40, "123")
    pointer = json.loads((ROOT / "catalog/channels/stable-v2.json").read_text())
    current = json.loads((ROOT / "catalog/releases" / f"{pointer['release_id']}.json").read_text())
    assert result["generation"] == pointer["generation"] + 1
    assert result["producer"]["run_id"] == "123"
    runtimes = {r["target_triple"]: r for r in result["artifacts"]["runtimes"]}
    assert set(runtimes) == set(lock["targets"]) | {"x86_64-apple-darwin"}
    for target in lock["targets"]:
        runtime = runtimes[target]
        assert runtime["download_url"].startswith("https://github.com/thedavidweng/openkara-models/releases/download/infra-2026-11-25-001/")
        assert runtime["artifact_id"].endswith(runtime["archive_digest"][:16])
        assert runtime["upstream"]["commit_sha"] == lock["upstream"]["commit_sha"]
    intel = next(r for r in current["artifacts"]["runtimes"] if r["target_triple"] == "x86_64-apple-darwin")
    for key in ("artifact_id", "download_url", "byte_size", "archive_digest", "extracted_file_digests"):
        assert runtimes["x86_64-apple-darwin"][key] == intel[key]
    assert runtimes["x86_64-apple-darwin"]["runtime"]["execution_providers"] == ["cpu"]
    assert "microsoft/onnxruntime/releases" not in json.dumps(result)


def test_generated_catalog_rejects_a_different_intel_archive(tmp_path, monkeypatch):
    import pytest
    import generate_catalog_v2 as generator

    _fake_nuget_archives(tmp_path, generator)
    monkeypatch.setattr(generator, "PLUGINS", {})
    intel = tmp_path / "intel.tar.gz"
    intel.write_bytes(b"not the published Intel runtime")
    with pytest.raises(ValueError, match="integrity mismatch"):
        generator.generate(tmp_path, intel, "2026-11-25-001", "2026-11-25T00:00:00Z", "a" * 40)
