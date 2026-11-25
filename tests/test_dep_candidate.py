"""Tests for the dependency candidate generator."""

from __future__ import annotations

import json
import re
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
        "old_tag": "v1.32.0",
        "new_tag": "v1.33.0",
        "old_commit": "a" * 40,
        "new_commit": "b" * 40,
        "lock_path": "ort/source-lock.json",
        "nuget_lock_path": "ort/nuget-lock.json",
        "ort_api_version": 33,
        "source_lock_submodules_refreshed": False,
        "nuget_package": {
            "id": "Microsoft.ML.OnnxRuntime",
            "version": "1.33.0",
            "url": "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/1.33.0/microsoft.ml.onnxruntime.1.33.0.nupkg",
            "size": 12345,
            "sha256": "c" * 64,
            "signers": {
                "author": {"sha256_fingerprint": "A" * 64, "subject": "CN=Microsoft Corporation"},
                "repository": {"sha256_fingerprint": "B" * 64, "subject": "CN=NuGet.org Repository by Microsoft"},
            },
        },
        "intel_macos": {
            "status": "carried-forward-on-upstream-unavailable",
            "has_nuget_osx_x64": False,
            "previous": {
                "tag": "v1.32.0",
                "artifact_id": "onnxruntime-1.32.0-x86_64-apple-darwin-a3b2691f53534069",
                "archive_digest": "a3b2691f" + "0" * 56,
            },
        },
    }
    summary = g.generate_candidate_summary(changes)
    assert "ONNX Runtime NuGet update" in summary
    assert "v1.32.0" in summary
    assert "v1.33.0" in summary
    assert "does not ship an `osx-x64` binary" in summary
    assert "Degraded (carried forward if upstream source is unavailable)" in summary
    assert "official-runtime-smoke.yml" in summary
    assert "Runtime comparison passes" in summary


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


def test_update_ort_lock_refreshes_nuget_and_source_locks_without_github(tmp_path, monkeypatch) -> None:
    import hashlib
    import zipfile
    import generate_dep_candidate as g
    import nuget_package

    source_work = tmp_path / "source-lock.json"
    nuget_work = tmp_path / "nuget-lock.json"
    source_work.write_bytes(g.LOCK_PATH.read_bytes())
    nuget_work.write_bytes(g.NUGET_LOCK_PATH.read_bytes())
    before_source = json.loads(source_work.read_text())
    monkeypatch.setattr(g, "LOCK_PATH", source_work)
    monkeypatch.setattr(g, "NUGET_LOCK_PATH", nuget_work)

    commit = "a" * 40
    nupkg = tmp_path / "Microsoft.ML.OnnxRuntime.1.33.0.nupkg"
    members = {
        "LICENSE": b"new license",
        "ThirdPartyNotices.txt": b"new notices",
        "build/native/include/onnxruntime_c_api.h": b'#define ORT_API_VERSION 33\n#include "onnxruntime_ep_c_api.h"\n',
        "build/native/include/onnxruntime_ep_c_api.h": b"// ep 33\n",
        "runtimes/linux-arm64/native/libonnxruntime.so": b"arm64-so-33",
        "runtimes/linux-arm64/native/libonnxruntime_providers_shared.so": b"arm64-shared-33",
        "runtimes/linux-x64/native/libonnxruntime.so": b"x64-so-33",
        "runtimes/linux-x64/native/libonnxruntime_providers_shared.so": b"x64-shared-33",
        "runtimes/osx-arm64/native/libonnxruntime.dylib": b"dylib Developer ID Application: Microsoft Corporation (UBF8T346G9)",
        "runtimes/win-x64/native/onnxruntime.dll": b"win-dll-33",
        "runtimes/win-x64/native/onnxruntime_providers_shared.dll": b"win-shared-33",
    }
    with zipfile.ZipFile(nupkg, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)

    author_fp = "A" * 64
    repo_fp = "B" * 64
    verify_out = (
        f"Verifying Microsoft.ML.OnnxRuntime.1.33.0\n"
        f"Signature type: Author\n"
        f"  Subject Name: {nuget_package.AUTHOR_SUBJECT}\n"
        f"  SHA256 hash: {author_fp}\n"
        f"Signature type: Repository\n"
        f"Service index: https://api.nuget.org/v3/index.json\n"
        f"  Subject Name: {nuget_package.REPOSITORY_SUBJECT}\n"
        f"  SHA256 hash: {repo_fp}\n"
        f"Successfully verified package 'Microsoft.ML.OnnxRuntime.1.33.0'.\n"
    )

    with mock.patch.object(nuget_package, "nuspec_commit", return_value=commit), \
         mock.patch.object(nuget_package, "download_unpinned", return_value=nupkg), \
         mock.patch.object(nuget_package, "verify_signatures", return_value=verify_out):
        changes = g.update_ort_lock("v1.33.0", commit, cache_dir=tmp_path)

    updated_source = json.loads(source_work.read_text())
    updated_nuget = json.loads(nuget_work.read_text())
    assert updated_source["upstream"]["tag"] == "v1.33.0"
    assert updated_source["upstream"]["commit_sha"] == commit
    assert updated_source["upstream"]["release_url"] == "https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime/1.33.0"
    assert updated_source["submodules"] == before_source["submodules"]
    assert updated_source["deps"] == before_source["deps"]
    assert updated_source["targets"] == before_source["targets"]

    assert updated_nuget["package"]["version"] == "1.33.0"
    assert updated_nuget["package"]["size"] == nupkg.stat().st_size
    assert updated_nuget["package"]["sha256"] == hashlib.sha256(nupkg.read_bytes()).hexdigest()
    assert updated_nuget["package"]["signers"]["author"]["sha256_fingerprint"] == author_fp
    assert updated_nuget["package"]["signers"]["repository"]["sha256_fingerprint"] == repo_fp
    assert updated_nuget["targets"]["aarch64-apple-darwin"]["root"] == "onnxruntime-osx-arm64-1.33.0"
    assert "lib/libonnxruntime.1.33.0.dylib" in updated_nuget["targets"]["aarch64-apple-darwin"]["files"]
    assert updated_nuget["targets"]["x86_64-unknown-linux-gnu"]["symlinks"]["lib/libonnxruntime.so.1"] == "libonnxruntime.so.1.33.0"
    assert updated_nuget["members"]["runtimes/linux-x64/native/libonnxruntime.so"] == {
        "size": len(members["runtimes/linux-x64/native/libonnxruntime.so"]),
        "sha256": hashlib.sha256(members["runtimes/linux-x64/native/libonnxruntime.so"]).hexdigest(),
    }
    assert updated_nuget["members"]["runtimes/osx-arm64/native/libonnxruntime.dylib"]["apple_team_id"] == "UBF8T346G9"
    assert changes["ort_api_version"] == 33
    assert changes["intel_macos"]["has_nuget_osx_x64"] is False
    assert changes["intel_macos"]["status"] == "carried-forward-on-upstream-unavailable"

    saved_source = source_work.read_bytes()
    saved_nuget = nuget_work.read_bytes()
    with mock.patch.object(nuget_package, "nuspec_commit", return_value="d" * 40):
        with pytest.raises(ValueError, match="tag"):
            g.update_ort_lock("v1.33.0", commit, cache_dir=tmp_path)
    assert source_work.read_bytes() == saved_source
    assert nuget_work.read_bytes() == saved_nuget

    with mock.patch.object(nuget_package, "nuspec_commit", return_value=commit), \
         mock.patch.object(nuget_package, "download_unpinned", return_value=nupkg), \
         mock.patch.object(nuget_package, "verify_signatures", side_effect=RuntimeError("bad sig")):
        with pytest.raises(RuntimeError, match="bad sig"):
            g.update_ort_lock("v1.33.0", commit, cache_dir=tmp_path)
    assert source_work.read_bytes() == saved_source
    assert nuget_work.read_bytes() == saved_nuget


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


def _numpy_floor() -> tuple[int, ...]:
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line.lower().startswith("numpy"):
            for part in line.split("numpy", 1)[1].split(","):
                if part.strip().startswith(">="):
                    return tuple(int(x) for x in part.strip()[2:].split("."))
    raise AssertionError("requirements.txt has no numpy lower bound")


def test_candidate_builds_only_intel():
    candidate = (ROOT / ".github/workflows/dep-candidate.yml").read_text()
    native = (ROOT / ".github/workflows/ort-build.yml").read_text()
    assert "uses: ./.github/workflows/ort-build.yml" in candidate
    assert "--target x86_64-apple-darwin" in native
    assert "runs-on: macos-latest" in native
    assert "CMAKE_OSX_ARCHITECTURES=x86_64" in native
    assert "--reduced" not in native


def test_requirements_numpy_floor_keeps_python_311_resolvable():
    """``numpy>=2.5`` cannot resolve on Python 3.11.

    ``requirements.txt`` is proven to resolve on Python 3.11 (NumPy 2.4.6), which
    the runtime toolchain still uses for its own separate dependency install.
    NumPy 2.5.0 onwards declares ``Requires-Python >=3.12``, so raising this
    floor would make the file unresolvable there. The floor must stay at 2.4 or
    above and below 2.5.
    """
    floor = _numpy_floor()
    assert floor >= (2, 4), "numpy floor drifted below a currently supported series"
    assert floor < (2, 5), "numpy>=2.5 cannot resolve on Python 3.11"


def test_converter_python_is_the_highest_still_supported_by_the_dependency_graph():
    """``convert.yml`` must run on a Python the full CPU dependency graph supports.

    Torch 2.14 ships ``cp310``-``cp314`` wheels and NumPy 2.5+ needs >=3.12, so
    3.14 is the newest CPython that can install the whole graph; 3.15 has no
    torch wheel at all. A resolver failure here would stop conversion before any
    export work starts, so every convert job pins the same supported version.
    """
    import re

    convert = (ROOT / ".github/workflows/convert.yml").read_text()
    versions = re.findall(r'python-version:\s*"([0-9.]+)"', convert)
    assert versions, "convert.yml must pin python-version explicitly"
    assert len(set(versions)) == 1, f"convert jobs disagree on Python: {sorted(set(versions))}"
    converter = tuple(int(x) for x in versions[0].split("."))

    # Torch/torchaudio publish no cp315 wheels, so 3.15 cannot resolve.
    assert converter <= (3, 14), f"Python {versions[0]} has no torch wheels"
    # NumPy 2.5+ requires >=3.12; anything older cannot take a current NumPy.
    assert converter >= (3, 12), f"Python {versions[0]} cannot resolve numpy>=2.4"


def test_dispatch_defaults_to_artifact_only():
    """Manual dispatch must not publish unless the caller asks for it.

    The default dispatch path previously ran the release step, which needs a
    write token. A dry run of the converter should be safe by default, so
    dispatch defaults ``publish`` to false and only tag pushes publish.
    """
    convert = (ROOT / ".github/workflows/convert.yml").read_text()

    dispatch = convert.split("workflow_dispatch:", 1)[1].split("permissions:", 1)[0]
    assert "publish:" in dispatch, "dispatch must expose a publish input"
    assert "type: boolean" in dispatch
    assert "default: false" in dispatch, "dispatch must default to artifact-only"


def _convert_jobs() -> dict[str, str]:
    """Split ``convert.yml`` into top-level job blocks, without needing a YAML parser."""
    import re

    convert = (ROOT / ".github/workflows/convert.yml").read_text()
    body = convert.split("\njobs:\n", 1)[1]
    blocks = re.split(r"\n(?=  [A-Za-z0-9_-]+:\n)", body)
    jobs: dict[str, str] = {}
    for block in blocks:
        name = block.split(":", 1)[0].strip()
        if name:
            jobs[name] = block
    return jobs


def _steps(job_block: str) -> dict[str, str]:
    """Split a job block into step blocks keyed by step name."""
    import re

    steps: dict[str, str] = {}
    current_name: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        if current_name is not None:
            steps[current_name] = "\n".join(current_lines)

    for line in job_block.splitlines():
        starts_step = line.startswith("      - ")
        if starts_step:
            flush()
            current_lines = [line]
            m = re.match(r"^      - name:\s*(.+)$", line)
            current_name = m.group(1).strip() if m else None
            continue
        if current_lines:
            current_lines.append(line)
    flush()
    return steps


def _step_with(step: str) -> dict[str, str]:
    """Parse a step's ``with:`` block into key -> value (``|`` blocks joined)."""
    import re

    lines = step.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "with:")
    inputs: dict[str, str] = {}
    i = start + 1
    while i < len(lines):
        m = re.match(r"^          ([A-Za-z0-9_-]+):\s*(.*)$", lines[i])
        if not m:
            break
        key, value = m.group(1), m.group(2).strip()
        if value == "|":
            block = []
            i += 1
            while i < len(lines) and lines[i].startswith("            "):
                block.append(lines[i].strip())
                i += 1
            inputs[key] = "\n".join(block)
            continue
        inputs[key] = value
        i += 1
    return inputs


def _lca_root(paths: list[str]) -> str:
    """Least common ancestor directory of upload paths, as upload-artifact computes it."""
    split = [p.split("/") for p in paths]
    common: list[str] = []
    for parts in zip(*split):
        if len(set(parts)) != 1:
            break
        common.append(parts[0])
    return "/".join(common) or "."


def test_artifact_producer_and_consumer_names_match():
    """Each release job must download the artifact its producer actually uploads.

    ``actions/download-artifact`` matches on the artifact name, so a stale or
    hard-coded name in the consumer silently breaks the release while every
    other check still passes.
    """
    jobs = _convert_jobs()
    pairs = (
        ("convert-single", "Upload artifacts", "release-single", "Download converted model"),
        ("finalize-ensemble", "Upload artifacts", "release-ensemble", "Download converted model"),
    )
    for producer_job, producer_step, consumer_job, consumer_step in pairs:
        produced = _step_with(_steps(jobs[producer_job])[producer_step])["name"]
        consumed = _step_with(_steps(jobs[consumer_job])[consumer_step])["name"]
        assert produced == consumed, (
            f"{consumer_job} downloads {consumed!r} but {producer_job} uploads {produced!r}"
        )

    # The single-model artifact name follows the dispatched model.
    single = _step_with(_steps(jobs["convert-single"])["Upload artifacts"])["name"]
    assert "needs.detect-model.outputs.model" in single


def test_artifact_restore_path_matches_the_upload_layout():
    """Downloads must extract at the repo root the upload was rooted at.

    ``upload-artifact`` roots a multi-path artifact at the least common ancestor
    of its paths. The producers upload ``models/<artifact>``,
    ``models/<artifact>.sha256`` and ``release-body.md``, whose LCA is the repo
    root, so the consumer has to extract into ``.``. Extracting into a narrower
    directory would place ``release-body.md`` where ``body_path`` cannot find it
    and shift the model away from the ``files:`` paths.
    """
    jobs = _convert_jobs()
    pairs = (
        ("convert-single", "release-single"),
        ("finalize-ensemble", "release-ensemble"),
    )
    for producer_job, consumer_job in pairs:
        upload = _step_with(_steps(jobs[producer_job])["Upload artifacts"])
        upload_paths = upload["path"].splitlines()
        assert _lca_root(upload_paths) == ".", (
            f"{producer_job} upload paths are not rooted at the repo root: {upload_paths}"
        )

        download = _step_with(_steps(jobs[consumer_job])["Download converted model"])
        assert download["path"] == ".", (
            f"{consumer_job} must restore at the repo root, got {download['path']!r}"
        )

        # Every restored file must be referenced where the release step expects it.
        release = _step_with(_steps(jobs[consumer_job])["Create Release"])
        release_text = "\n".join(release.values())
        for path in upload_paths:
            assert path in release_text, f"{consumer_job} does not reference restored {path!r}"



def test_only_the_release_jobs_hold_a_write_token():
    """Export/validation jobs stay read-only; release jobs are gated on publish.

    Keeping ``contents: write`` on the conversion jobs would let any dispatch
    write a release even when publishing is off. Only the release jobs may hold
    the token, and each one also requires the tag/publish condition.
    """
    convert = (ROOT / ".github/workflows/convert.yml").read_text()
    assert "\npermissions:\n  contents: read\n" in convert, "workflow default must be read-only"

    jobs = _convert_jobs()
    writers = [
        name for name, block in jobs.items()
        if "\n    permissions:\n      contents: write\n" in block
    ]
    assert sorted(writers) == ["release-ensemble", "release-single"], (
        f"unexpected jobs with a write token: {sorted(writers)}"
    )
    for name in writers:
        guard = jobs[name].split("if:", 1)[1].split("\n", 1)[0]
        assert "publish == 'true'" in guard, f"{name} must require publish"
        assert "startsWith(github.ref, 'refs/tags/')" in guard, f"{name} must require a tag"


def test_artifact_only_dispatch_does_not_reach_a_release_step():
    """With ``publish: false`` on a dispatch, no release job can be scheduled.

    GitHub evaluates ``if:`` per job; a false guard means the release job never
    runs, so the run leaves only ``upload-artifact`` output.
    """
    jobs = _convert_jobs()
    for name in ("release-single", "release-ensemble"):
        guard = jobs[name].split("if:", 1)[1].split("\n", 1)[0]
        # Dispatch with publish=false fails the publish half of the guard.
        assert "publish == 'true'" in guard
        # A dispatch ref is not a tag, so the tag half fails too.
        assert "refs/tags/" in guard

    # The conversion jobs must not create releases themselves.
    for name in ("convert-single", "finalize-ensemble"):
        assert "action-gh-release" not in jobs[name], f"{name} must not create a release"
        assert "contents: write" not in jobs[name]



def test_tag_push_still_publishes_the_same_artifact():
    """Tag pushes keep their existing release semantics and asset set.

    The publish guard must not change what a tagged build produces: the same
    ONNX file plus checksum, with release notes generated from the same body.
    """
    convert = (ROOT / ".github/workflows/convert.yml").read_text()
    detect = convert.split("Determine model from tag or input", 1)[1]
    assert "PUBLISH=true" in detect, "tag pushes must set publish=true"

    jobs = _convert_jobs()
    for job in ("release-single", "release-ensemble"):
        block = jobs[job]
        assert "models/${{ needs.detect-model.outputs.artifact }}" in block
        assert "models/${{ needs.detect-model.outputs.artifact }}.sha256" in block
        assert "body_path: release-body.md" in block
        assert "generate_release_notes: true" in block





def test_candidate_and_smoke_workflows_support_nuget_version_updates():
    candidate = (ROOT / ".github/workflows/dep-candidate.yml").read_text()
    assert "ort/nuget-lock.json" in candidate
    assert "git add ort/source-lock.json ort/nuget-lock.json" in candidate
    assert "repackage_nuget_runtime.py --validate-lock" in candidate
    smoke = (ROOT / ".github/workflows/official-runtime-smoke.yml").read_text()
    assert "archive: onnxruntime-1.32.0-" not in smoke
    assert "repackage_nuget_runtime" in smoke


def test_candidate_and_detection_workflows_support_webgpu_plugin():
    candidate = (ROOT / ".github/workflows/dep-candidate.yml").read_text()
    assert "plugin_version" in candidate
    assert "git add ort/plugin-lock.json" in candidate
    assert "plugin_package.py --validate-lock" in candidate

    detection = (ROOT / ".github/workflows/dep-detection.yml").read_text()
    assert "detect-plugin:" in detection
    assert "detect_plugin_release.py" in detection
    assert "--plugin webgpu" in detection

    smoke = (ROOT / ".github/workflows/official-runtime-smoke.yml").read_text()
    assert "provider: webgpu" in smoke
    assert "plugin_package.py --download plugins" in smoke


def test_update_plugin_lock_verifies_sha_and_ort_compatibility(tmp_path, monkeypatch):
    import generate_dep_candidate as gdc
    import plugin_package
    from test_plugin_package import _build_wheel

    nuget_lock = json.loads((ROOT / "ort/nuget-lock.json").read_text())
    plugin_lock_path = tmp_path / "plugin-lock.json"
    plugin_lock_path.write_text((ROOT / "ort/plugin-lock.json").read_text())
    nuget_lock_path = tmp_path / "nuget-lock.json"
    nuget_lock_path.write_text(json.dumps(nuget_lock))

    monkeypatch.setattr(gdc, "PLUGIN_LOCK_PATH", plugin_lock_path)
    monkeypatch.setattr(gdc, "NUGET_LOCK_PATH", nuget_lock_path)

    win_whl = _build_wheel(
        tmp_path / "onnxruntime_ep_webgpu-0.6.0-py3-none-win_amd64.whl",
        version="0.6.0",
        target="x86_64-pc-windows-msvc",
    )
    mac_whl = _build_wheel(
        tmp_path / "onnxruntime_ep_webgpu-0.6.0-py3-none-macosx_14_0_universal2.whl",
        version="0.6.0",
        target="aarch64-apple-darwin",
    )
    pypi_payload = {
        "urls": [
            {
                "filename": win_whl.name,
                "url": f"https://files.pythonhosted.org/packages/ab/cd/{'0' * 60}/{win_whl.name}",
                "size": win_whl.stat().st_size,
                "digests": {"sha256": plugin_package.sha256_file(win_whl)[1]},
                "yanked": False,
            },
            {
                "filename": mac_whl.name,
                "url": f"https://files.pythonhosted.org/packages/ab/cd/{'0' * 60}/{mac_whl.name}",
                "size": mac_whl.stat().st_size,
                "digests": {"sha256": plugin_package.sha256_file(mac_whl)[1]},
                "yanked": False,
            },
        ]
    }
    monkeypatch.setattr(plugin_package, "fetch_pypi_json", lambda version=None: pypi_payload)
    summary = gdc.update_plugin_lock(
        "webgpu", "0.6.0", cache_dir=tmp_path, reuse_cache=True, ort_c_api_level=32
    )
    assert summary["new_version"] == "0.6.0"
    updated = json.loads(plugin_lock_path.read_text())
    assert updated["package"]["version"] == "0.6.0"
    assert plugin_package.validate_lock(updated, nuget_lock) == []
