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


def test_reduced_job_uses_the_locked_python_and_resolvable_deps():
    """The reduced job must not pick a Python the toolchain lock rejects.

    ``build_runtime.py`` calls ``_assert_toolchain`` before any CMake work and
    fails closed on a python_version mismatch, so a job that selects a different
    interpreter can never build. NumPy 2.5.x also requires Python >= 3.12, which
    is why pinning both at once is impossible on the locked 3.11.
    """
    native = (ROOT / ".github/workflows/ort-build.yml").read_text()
    lock = json.loads((ROOT / "ort/source-lock.json").read_text())
    locked_python = lock["toolchain"]["python_version"]

    reduced = native.split("  reduced-build-and-compare:", 1)[1].split(
        "  benchmark-x86_64-unknown-linux-gnu:", 1
    )[0]
    assert f'python-version: "{locked_python}"' in reduced
    # NumPy 2.5+ is unavailable on the locked interpreter; requiring it would
    # fail resolution before the build ever starts.
    assert "numpy>=2.5" not in reduced
    assert "numpy>=1.24" in reduced
    # The operator config needs the pinned flatbuffers and a current onnx.
    assert 'flatbuffers==25.12.19' in reduced
    assert "onnx>=1.23.2" in reduced


def _numpy_floor() -> tuple[int, ...]:
    """Lower bound of the numpy requirement in requirements.txt."""
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line.lower().startswith("numpy"):
            continue
        spec = line.split("numpy", 1)[1]
        for part in spec.split(","):
            part = part.strip()
            if part.startswith(">="):
                return tuple(int(x) for x in part[2:].split("."))
    raise AssertionError("requirements.txt has no numpy lower bound")


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



