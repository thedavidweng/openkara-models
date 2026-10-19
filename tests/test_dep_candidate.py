"""Tests for the dependency candidate generator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def test_generate_candidate_summary_ort() -> None:
    import generate_dep_candidate as g
    changes = {
        "type": "ort",
        "old_tag": "v1.27.1",
        "new_tag": "v1.28.0",
        "old_commit": "abc123",
        "new_commit": "def456",
        "lock_path": "ort/source-lock.json",
    }
    summary = g.generate_candidate_summary(changes)
    assert "ONNX Runtime source update" in summary
    assert "v1.27.1" in summary
    assert "v1.28.0" in summary
    assert "Rebuild all 6 runtime targets" in summary
    assert "Quality gates pass" in summary


def test_generate_candidate_summary_model() -> None:
    import generate_dep_candidate as g
    changes = {
        "type": "model",
        "model": "htdemucs",
        "old_commit": "abc123",
        "new_commit": "def456",
    }
    summary = g.generate_candidate_summary(changes)
    assert "Model-weight revision update" in summary
    assert "htdemucs" in summary
    assert "Re-convert affected ONNX models" in summary


def test_cli_ort_requires_commit() -> None:
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "generate_dep_candidate.py"), "--ort-tag", "v1.28.0"],
        capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "--ort-commit" in r.stderr


def test_cli_model_requires_commit() -> None:
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "generate_dep_candidate.py"), "--model", "htdemucs"],
        capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "--model-commit" in r.stderr


def test_cli_help() -> None:
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "generate_dep_candidate.py"), "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert "--ort-tag" in r.stdout
    assert "--model" in r.stdout


def test_update_model_lock_roundtrip(tmp_path) -> None:
    """update_model_lock records old/new commits and rewrites the lock."""
    import generate_dep_candidate as g
    src = ROOT / "models" / "source-lock.json"
    work = tmp_path / "source-lock.json"
    work.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    original = g.MODEL_LOCK_PATH
    g.MODEL_LOCK_PATH = work
    try:
        changes = g.update_model_lock("htdemucs", "a" * 40)
        assert changes["type"] == "model"
        assert changes["new_commit"] == "a" * 40
        assert changes["old_commit"] and changes["old_commit"] != changes["new_commit"]
        updated = json.loads(work.read_text(encoding="utf-8"))
        assert updated["models"]["htdemucs"]["weights"]["commit_sha"] == "a" * 40
        # The other model entry is untouched.
        orig = json.loads(src.read_text(encoding="utf-8"))
        assert (updated["models"]["htdemucs_ft"]["weights"]["commit_sha"]
                == orig["models"]["htdemucs_ft"]["weights"]["commit_sha"])
    finally:
        g.MODEL_LOCK_PATH = original


def test_update_model_lock_unknown_model(tmp_path) -> None:
    import generate_dep_candidate as g
    src = ROOT / "models" / "source-lock.json"
    work = tmp_path / "source-lock.json"
    work.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    original = g.MODEL_LOCK_PATH
    g.MODEL_LOCK_PATH = work
    try:
        import pytest
        with pytest.raises(SystemExit):
            g.update_model_lock("nonexistent", "a" * 40)
    finally:
        g.MODEL_LOCK_PATH = original


def test_update_ort_lock_refreshes_coupled_pins(tmp_path, monkeypatch) -> None:
    import base64
    import generate_dep_candidate as g
    import detect_ort_release as d

    work = tmp_path / "source-lock.json"
    work.write_bytes(g.LOCK_PATH.read_bytes())
    before = json.loads(work.read_text())
    monkeypatch.setattr(g, "LOCK_PATH", work)
    commit = "a" * 40
    texts = {
        ".gitmodules": '[submodule "new"]\npath = cmake/external/new\nurl = https://github.com/owner/new.git\n',
        "cmake/deps.txt": "#Name;Url;SHA1\nnew;https://example.com/new.zip;" + "b" * 40,
    }
    tree = {"truncated": False, "tree": [
        {"path": "cmake/external/new", "mode": "160000", "sha": "c" * 40},
    ]}

    def github_get(url, token):
        assert commit in url
        if "/git/trees/" in url:
            return tree
        path = url.split("/contents/")[1].split("?")[0]
        return {"content": base64.b64encode(texts[path].encode()).decode()}

    with mock.patch.object(d, "_tag_commit_sha", return_value=commit), \
         mock.patch.object(d, "_github_get", side_effect=github_get):
        changes = g.update_ort_lock("v1.31.0", commit)
    updated = json.loads(work.read_text())
    assert updated["submodules"] == {
        "cmake/external/new": {"repo": "owner/new", "expected_sha": "c" * 40},
    }
    assert updated["deps"]["entries"] == {
        "new": {"url": "https://example.com/new.zip", "sha1": "b" * 40},
    }
    assert updated["targets"] == before["targets"]
    assert changes["old_tag"] == before["upstream"]["tag"]
    assert changes["new_commit"] == commit

    saved = work.read_bytes()
    for malformed in ("new;url", "new;url;sha;extra", "new;url;bad", "", "new;url;" + "b" * 40 + "\nnew;url;" + "b" * 40):
        texts["cmake/deps.txt"] = malformed
        with mock.patch.object(d, "_tag_commit_sha", return_value=commit), \
             mock.patch.object(d, "_github_get", side_effect=github_get):
            with pytest.raises(ValueError):
                g.update_ort_lock("v1.31.0", commit)
        assert work.read_bytes() == saved
    texts[".gitmodules"] = texts[".gitmodules"].replace("cmake/external/new", "cmake/external/missing")
    with mock.patch.object(d, "_tag_commit_sha", return_value=commit), \
         mock.patch.object(d, "_github_get", side_effect=github_get):
        with pytest.raises(ValueError, match="gitlinks disagree"):
            g.update_ort_lock("v1.31.0", commit)
    assert work.read_bytes() == saved
    with mock.patch.object(d, "_tag_commit_sha", return_value="d" * 40):
        with pytest.raises(ValueError, match="tag"):
            g.update_ort_lock("v1.31.0", commit)
    assert work.read_bytes() == saved


def test_dispatch_candidate_is_lightweight_and_deduplicated():
    from dispatch_dep_candidate import dispatch
    inputs = {"ort_tag": "v1.31.0", "ort_commit": "a" * 40}
    with mock.patch("dispatch_dep_candidate.subprocess.run") as run:
        run.return_value.stdout = "[]"
        assert dispatch(inputs, "owner/repo") is True
        command = run.call_args.args[0]
        assert "build_native=false" in command
        assert "ort_commit=" + "a" * 40 in command
        run.reset_mock()
        run.return_value.stdout = '[{"number": 1}]'
        assert dispatch(inputs, "owner/repo") is False
        assert run.call_count == 1
    with pytest.raises(ValueError, match="SHA"):
        dispatch({"model": "htdemucs", "model_commit": "main"}, "owner/repo")


def test_candidate_reuses_exact_native_pipeline():
    candidate = (ROOT / ".github/workflows/dep-candidate.yml").read_text()
    native = (ROOT / ".github/workflows/ort-build.yml").read_text()
    lock = json.loads((ROOT / "ort/source-lock.json").read_text())
    assert "uses: ./.github/workflows/ort-build.yml" in candidate
    assert "python scripts/build_runtime.py" not in candidate
    assert "if: inputs.build_native && inputs.ort_tag != ''" in candidate
    assert 'default: false' in candidate
    for target in lock["targets"]:
        assert f"  build-{target}:" in native
        assert f"python scripts/build_runtime.py --locked --target {target}" in native
    # Native work is scheduled in three pairs; no six-way runner burst.
    assert "needs: [validate-source-lock, build-aarch64-apple-darwin, build-x86_64-apple-darwin]" in native
    assert "inputs.candidate_lock_artifact == ''" in native
    assert "git push --force" not in candidate
