import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

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
    with pytest.raises(ValueError):
        extracted_digests(archive_path)
