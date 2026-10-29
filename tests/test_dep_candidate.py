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

    def job_block(workflow: str, name: str) -> str:
        match = re.search(r"(?:^|\n)  " + re.escape(name) + r":\n(.*?)(?=\n  \S|\Z)", workflow, re.DOTALL)
        assert match is not None, f"job {name} not found"
        return match.group(1)

    expected_abi_floor = {
        "max-glibc": "2.35",
        "max-glibcxx": "3.4.30",
        "max-cxxabi": "1.3.13",
    }

    for workflow, jobs in (
        (native, (
            "build-x86_64-unknown-linux-gnu",
            "build-aarch64-unknown-linux-gnu",
            "reduced-build-and-compare",
        )),
        (publish, ("build-x86_64-linux", "build-aarch64-linux")),
    ):
        for job in jobs:
            body = job_block(workflow, job)
            assert "container:" in body, f"{job} must run in a container"
            assert "quay.io/pypa/manylinux_2_28_" in body, job
            assert "@sha256:" in body, f"{job} must pin the image digest"

            steps = re.split(r"\n    - |\n      - ", body)[1:]
            abi_runs = []
            for step in steps:
                run_match = re.search(r"(?:^|\n)\s*run:\s*[>|]?(.*)", step, re.DOTALL)
                if not run_match:
                    continue
                code_lines = [l for l in run_match.group(1).splitlines() if not l.strip().startswith("#")]
                code = " ".join(code_lines)
                if "check_linux_abi_floor.py" in code:
                    abi_runs.append(code)
            assert len(abi_runs) == 1, f"{job} must have exactly one check_linux_abi_floor.py run step"
            code = abi_runs[0]
            m_glibc = re.search(r"--max-glibc\s+([^\s\\]+)", code)
            m_glibcxx = re.search(r"--max-glibcxx\s+([^\s\\]+)", code)
            m_cxxabi = re.search(r"--max-cxxabi\s+([^\s\\]+)", code)
            assert m_glibc is not None, f"{job} missing --max-glibc"
            assert m_glibcxx is not None, f"{job} missing --max-glibcxx"
            assert m_cxxabi is not None, f"{job} missing --max-cxxabi"
            parsed_args = {
                "max-glibc": m_glibc.group(1),
                "max-glibcxx": m_glibcxx.group(1),
                "max-cxxabi": m_cxxabi.group(1),
            }
            assert parsed_args == expected_abi_floor, (
                f"{job} ABI floor arguments mismatch: {parsed_args} != {expected_abi_floor}"
            )


def test_publish_job_depends_on_all_linux_producers_and_downloads_arm64():
    """Publish must depend on all runtime producers and its pattern must cover arm64.

    If publish drops build-aarch64-linux (or build-x86_64-linux), it runs in
    parallel with uncompleted builds and cuts an incomplete release. The
    download-artifact pattern must cover ort-aarch64-unknown-linux-gnu.
    """
    import fnmatch

    publish = (ROOT / ".github/workflows/ort-publish.yml").read_text()
    match = re.search(r"(?:^|\n)  publish:\n(.*?)(?=\n  \S|\Z)", publish, re.DOTALL)
    assert match is not None, "publish job not found"
    body = match.group(1)

    needs_match = re.search(r"^\s*needs:\s*\[(.*?)\]", body, re.MULTILINE)
    if needs_match:
        needs = [n.strip() for n in needs_match.group(1).split(",") if n.strip()]
    else:
        needs_list_match = re.search(r"^\s*needs:\s*\n((?:\s+-\s+\S+\n?)+)", body, re.MULTILINE)
        assert needs_list_match is not None, "publish job must specify needs list"
        needs = [re.sub(r"^\s+-\s+", "", line).strip() for line in needs_list_match.group(1).splitlines() if line.strip()]
    assert needs == ["build", "build-x86_64-linux", "build-aarch64-linux"]

    pattern_match = re.search(r"actions/download-artifact.*?pattern:\s*(\S+)", body, re.DOTALL)
    assert pattern_match is not None, "publish job must use actions/download-artifact with pattern"
    pattern = pattern_match.group(1).strip()
    assert fnmatch.fnmatch("ort-aarch64-unknown-linux-gnu", pattern), (
        f"download-artifact pattern '{pattern}' does not cover ort-aarch64-unknown-linux-gnu"
    )


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
    # The reduced job must not depend on the scheduler's implicit success():
    # a skipped Apple ancestor propagates through the DAG (runner #2205 shape),
    # so both DIRECT needs are pinned to explicit success, with the run-level
    # cancel guard. Static shape only; it cannot prove server scheduling.
    assert "!cancelled()" in reduced, "reduced job must keep the cancel guard"
    assert "needs.validate-source-lock.result == 'success'" in reduced, (
        "reduced job must require validate success explicitly"
    )
    assert "needs.build-x86_64-unknown-linux-gnu.result == 'success'" in reduced, (
        "reduced job must require the full baseline success explicitly"
    )
    assert "github.event_name == 'workflow_dispatch'" in reduced, "event gate stays"

    benchmark = job_block("benchmark-x86_64-unknown-linux-gnu")
    assert "!inputs.linux_only" in benchmark, "benchmark may stay skipped"


def test_reduced_job_gate_runs_only_on_healthy_direct_needs():
    """Static truth table for the reduced job gate (shape check, not a proof).

    Evaluates the reduced job's ``if:`` conjunction over dispatch/call,
    linux_only true/false, candidate gate set/empty, and success/failed/
    skipped/cancelled direct needs. This pins the intended scheduler contract
    in-repo; only a real dispatch can confirm GitHub's server evaluation.
    """
    native = (ROOT / ".github/workflows/ort-build.yml").read_text()

    def job_block(name: str) -> str:
        """Body of one top-level job: from its name to the next 2-space key."""
        return re.split(r"\n  (?=\S)", native.split(f"\n  {name}:\n", 1)[1], maxsplit=1)[0]

    reduced = job_block("reduced-build-and-compare")
    assert "inputs.linux_only" not in reduced, (
        "reduced job must not branch on linux_only at all"
    )
    gate = next(
        line.split("if:", 1)[1].strip()
        for line in reduced.splitlines()
        if line.strip().startswith("if:")
    )

    def holds(event, candidate_artifact, validate, full, cancelled):
        """Evaluate each && atom of the gate against one static context."""
        context = {
            "!cancelled()": not cancelled,
            "github.event_name == 'workflow_dispatch'": event == "workflow_dispatch",
            "inputs.candidate_lock_artifact == ''": candidate_artifact == "",
            "needs.validate-source-lock.result == 'success'": validate == "success",
            "needs.build-x86_64-unknown-linux-gnu.result == 'success'": full == "success",
        }
        atoms = [atom.strip().strip("${} ") for atom in gate.split("&&")]
        assert atoms, "reduced gate must be a non-empty conjunction"
        for atom in atoms:
            assert atom in context, f"gate gained an unmodelled atom: {atom}"
            if not context[atom]:
                return False
        return True

    healthy = dict(event="workflow_dispatch", candidate_artifact="",
                   validate="success", full="success", cancelled=False)
    # linux_only true AND false dispatch with healthy needs: both run.
    assert holds(**healthy) is True
    # workflow_call never enters the reduced job (unchanged contract).
    assert holds(**{**healthy, "event": "workflow_call"}) is False
    assert holds(**{**healthy, "event": "pull_request"}) is False
    # A candidate lock artifact present: candidate path takes over.
    assert holds(**{**healthy, "candidate_artifact": "ort-x86_64-unknown-linux-gnu"}) is False
    # Either direct need failed or skipped: must not run.
    for bad in ("failure", "skipped"):
        assert holds(**{**healthy, "validate": bad}) is False
        assert holds(**{**healthy, "full": bad}) is False
    # Cancelled run: the guard stops the job.
    assert holds(**{**healthy, "cancelled": True}) is False


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


def test_reduced_job_preserves_baseline_and_uploads_bound_smoke():
    native = (ROOT / ".github/workflows/ort-build.yml").read_text()
    job = native.split("\n  reduced-build-and-compare:\n", 1)[1].split(
        "\n  benchmark-x86_64-unknown-linux-gnu:", 1
    )[0]
    reduced_glob = "ort/packages/onnxruntime-*-x86_64-unknown-linux-gnu-reduced.tar.gz"

    def check(body):
        supply = body.split("- name: Generate supply-chain records for reduced runtime", 1)[1].split("\n      - ", 1)[0]
        assert f"REDUCED=$(ls {reduced_glob})" in supply
        assert 'generate_runtime_supply_chain.py --archive "$REDUCED"' in supply
        assert "--all" not in supply
        smoke = body.split("- name: Run reduced native smoke harness (CPU + XNNPACK)", 1)[1].split("\n      - ", 1)[0]
        assert f"RUNTIME=$(ls {reduced_glob})" in smoke
        assert "for PROV in cpu xnnpack" in smoke
        assert '--runtime "$RUNTIME"' in smoke
        assert '--report "smoke-x86_64-unknown-linux-gnu-reduced-${PROV}.json"' in smoke
        upload = body.split("name: smoke-x86_64-unknown-linux-gnu-reduced\n", 1)[1].split("\n      - ", 1)[0]
        assert "smoke-x86_64-unknown-linux-gnu-reduced-cpu.json" in upload
        assert "smoke-x86_64-unknown-linux-gnu-reduced-xnnpack.json" in upload
        assert "if-no-files-found: error" in upload
        comparison = body.split("- name: Compare full vs reduced (size + inference)", 1)[1].split("\n      - ", 1)[0]
        assert '--full "$FULL" --reduced "$REDUCED"' in comparison
        assert f"REDUCED=$(ls {reduced_glob})" in comparison

    check(job)
    import pytest
    with pytest.raises(AssertionError):
        check(job.replace(f"RUNTIME=$(ls {reduced_glob})", "RUNTIME=full.tar.gz"))
    with pytest.raises(AssertionError):
        check(job.replace("smoke-x86_64-unknown-linux-gnu-reduced-xnnpack.json", "missing.json"))


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



