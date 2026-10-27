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
    # The reduced job runs inside the manylinux container, so the interpreter
    # comes from the image's CPython tree instead of actions/setup-python.
    locked_interpreter = "/opt/python/cp" + locked_python.replace(".", "") + "-cp" + locked_python.replace(".", "")

    reduced = native.split("  reduced-build-and-compare:", 1)[1].split(
        "  benchmark-x86_64-unknown-linux-gnu:", 1
    )[0]
    assert locked_interpreter + "/bin/python" in reduced
    # NumPy 2.5+ is unavailable on the locked interpreter; requiring it would
    # fail resolution before the build ever starts.
    assert "numpy>=2.5" not in reduced
    assert "numpy>=1.24" in reduced
    # The operator config needs the pinned flatbuffers and a current onnx.
    assert 'flatbuffers==25.12.19' in reduced
    assert "onnx>=1.23.2" in reduced


def test_linux_build_jobs_pin_the_manylinux_glibc_floor():
    """Every Linux artifact must be built inside the controlled manylinux image.

    A shared library's glibc symbol floor comes from the environment it is
    built in, so a distro runner silently raises the floor above OpenKara's
    ubuntu-22.04 baseline (glibc 2.35). The x86_64 reduced and aarch64 jobs did
    exactly that (GLIBC_2.38); each Linux job must therefore run in the pinned
    manylinux_2_28 container and assert the floor on its own artifact.
    """
    native = (ROOT / ".github/workflows/ort-build.yml").read_text()
    publish = (ROOT / ".github/workflows/ort-publish.yml").read_text()
    for workflow, jobs in (
        (native, (
            "build-x86_64-unknown-linux-gnu",
            "build-aarch64-unknown-linux-gnu",
            "reduced-build-and-compare",
        )),
        (publish, ("build-x86_64-linux", "build-aarch64-linux")),
    ):
        for job in jobs:
            body = workflow.split(f"  {job}:", 1)[1]
            body = body.split("\n  build-", 1)[0]
            body = body.split("\n  reduced-build", 1)[0]
            body = body.split("\n  benchmark-", 1)[0]
            body = body.split("\n  publish", 1)[0]
            assert "container:" in body, f"{job} must run in a container"
            assert "quay.io/pypa/manylinux_2_28_" in body, job
            assert "@sha256:" in body, f"{job} must pin the image digest"
            assert "check_linux_abi_floor.py" in body, f"{job} must gate the ABI floor"


def test_linux_only_dispatch_skips_non_linux_jobs():
    """A linux_only dispatch must not rebuild the Apple or Windows targets.

    Rebuilding the two affected Linux archives is the only work a fix needs; a
    full six-target dispatch would burn native runner minutes re-verifying
    artifacts that did not change. The non-Linux jobs are gated on the input and
    the Linux jobs tolerate their `needs` being skipped.
    """
    native = (ROOT / ".github/workflows/ort-build.yml").read_text()
    inputs = native.split("  workflow_dispatch:\n", 1)[1].split("  pull_request:", 1)[0]
    assert "linux_only:" in inputs
    assert "type: boolean" in inputs

    def job_block(name: str) -> str:
        """Body of one top-level job: from its name to the next 2-space key."""
        return re.split(r"\n  (?=\S)", native.split(f"\n  {name}:\n", 1)[1], maxsplit=1)[0]

    for job in (
        "build-aarch64-apple-darwin",
        "build-x86_64-apple-darwin",
        "build-x86_64-pc-windows-msvc",
        "build-x86_64-pc-windows-msvc-cpu",
    ):
        body = job_block(job)
        assert "!inputs.linux_only" in body, f"{job} must be skipped by linux_only"
    for job in ("build-x86_64-unknown-linux-gnu", "build-aarch64-unknown-linux-gnu"):
        body = job_block(job)
        assert "inputs.linux_only" in body, f"{job} must run on linux_only"
        assert "!cancelled()" in body, f"{job} must tolerate skipped Apple needs"


def test_linux_only_dispatch_rebuilds_the_abi_affected_archives():
    """A linux_only dispatch must rebuild both ABI-affected archives.

    The two bad archives are the x86_64 Linux reduced archive and the aarch64
    Linux reduced archive. The x86_64 reduced archive is produced by
    reduced-build-and-compare, not by build-x86_64-unknown-linux-gnu (which is
    the full, already-qualified build). So the dispatch must run that reduced
    job, and the full build as its comparison baseline; only benchmark, which
    consumes the reduced archive and emits a report, may stay skipped.
    """
    native = (ROOT / ".github/workflows/ort-build.yml").read_text()

    def job_block(name: str) -> str:
        """Body of one top-level job: from its name to the next 2-space key."""
        return re.split(r"\n  (?=\S)", native.split(f"\n  {name}:\n", 1)[1], maxsplit=1)[0]

    reduced = job_block("reduced-build-and-compare")
    assert "!inputs.linux_only" not in reduced, "reduced job produces a bad archive"
    assert "inputs.candidate_lock_artifact == ''" in reduced, "dispatch gate stays"

    benchmark = job_block("benchmark-x86_64-unknown-linux-gnu")
    assert "!inputs.linux_only" in benchmark, "benchmark may stay skipped"


def test_reduced_job_is_the_x86_64_reduced_producer():
    """Pin which job produces which archive, so the selector cannot drift.

    build-x86_64-unknown-linux-gnu builds the full library (no --reduced);
    reduced-build-and-compare builds the reduced one and uploads it under
    ort-x86_64-unknown-linux-gnu-reduced, consuming the full artifact as its
    comparison baseline. A selector that skipped the reduced job would leave
    the bad x86_64 reduced archive unbuilt while rebuilding the good full one.
    """
    native = (ROOT / ".github/workflows/ort-build.yml").read_text()

    def job_block(name: str) -> str:
        """Body of one top-level job: from its name to the next 2-space key."""
        return re.split(r"\n  (?=\S)", native.split(f"\n  {name}:\n", 1)[1], maxsplit=1)[0]

    def build_lines(body: str) -> list[str]:
        return [line for line in body.splitlines() if "build_runtime.py --" in line]

    full = job_block("build-x86_64-unknown-linux-gnu")
    assert any("--target x86_64-unknown-linux-gnu" in line for line in build_lines(full))
    assert not any("--reduced" in line for line in build_lines(full)), (
        "the full build must not be the reduced producer"
    )

    reduced = job_block("reduced-build-and-compare")
    reduced_builds = build_lines(reduced)
    assert reduced_builds, "reduced job must build an archive"
    for line in reduced_builds:
        assert "--target x86_64-unknown-linux-gnu" in line, line.strip()
        assert "--reduced" in line, f"reduced job must pass --reduced: {line.strip()}"
        assert "--locked" in line, f"reduced build must pass --locked: {line.strip()}"
    assert "name: ort-x86_64-unknown-linux-gnu-reduced" in reduced, (
        "reduced job must upload the reduced archive under its own name"
    )
    # The full artifact is the reduced job's comparison baseline.
    assert "needs: [validate-source-lock, build-x86_64-unknown-linux-gnu]" in reduced
    assert "name: ort-x86_64-unknown-linux-gnu\n" in reduced, (
        "reduced job must download the full artifact as its baseline"
    )


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


def test_requirements_numpy_resolves_on_the_converter_python():
    """The shared requirements file must be installable on the converter's Python.

    ``convert.yml`` installs ``requirements.txt`` on Python 3.11, and NumPy
    2.5.0 onwards declares ``Requires-Python >=3.12``, so a 2.5+ floor makes
    ``pip install -r requirements.txt`` fail with "No matching distribution"
    before any conversion work starts (observed on run 40184776819). The floor
    therefore has to stay below 2.5 while the converter stays on 3.11, and must
    still allow a current NumPy series.
    """
    import re

    convert = (ROOT / ".github/workflows/convert.yml").read_text()
    versions = re.findall(r'python-version:\s*"([0-9.]+)"', convert)
    assert versions, "convert.yml must pin python-version explicitly"
    oldest = min(tuple(int(x) for x in v.split(".")) for v in versions)

    floor = _numpy_floor()
    # NumPy 2.5+ requires Python >=3.12; anything older cannot resolve it.
    if oldest < (3, 12):
        assert floor < (2, 5), (
            f"numpy>={'.'.join(map(str, floor))} cannot resolve on Python "
            f"{'.'.join(map(str, oldest))} used by convert.yml"
        )
    assert floor >= (2, 4), "numpy floor drifted below a currently supported series"


