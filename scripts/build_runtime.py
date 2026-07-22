#!/usr/bin/env python3
"""Build reproducible target-specific ONNX Runtime distributions from source lock.

Reads ``ort/source-lock.json``, clones ORT at the pinned commit, configures CMake
with target-specific flags, builds the shared library, and packages a normalized
archive with licenses, notices, and a build manifest.

Usage::

    python scripts/build_runtime.py --target aarch64-apple-darwin
    python scripts/build_runtime.py --target x86_64-pc-windows-msvc
    python scripts/build_runtime.py --target aarch64-apple-darwin --skip-clone

The build is reproducible: the same source lock + toolchain produces the same
archive (modulo timestamps, which are fixed to the ORT commit date).

This script is the entry point for both local and CI builds. The CI workflow
calls it on native runners for each target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "ort" / "source-lock.json"
BUILD_DIR = ROOT / "ort" / "build"
SOURCE_DIR = ROOT / "ort" / "source"
PACKAGES_DIR = ROOT / "ort" / "packages"

sys.path.insert(0, str(ROOT / "scripts"))
import ort_api_version  # noqa: E402


def _load_lock() -> dict[str, Any]:
    with LOCK_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None,
         check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=check,
                          capture_output=False)


def _sha256_file(path: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return size, h.hexdigest()


def _clone_source(lock: dict[str, Any], target_dir: Path) -> None:
    upstream = lock["upstream"]
    repo = upstream["repo"]
    commit = upstream["commit_sha"]

    if target_dir.exists() and (target_dir / ".git").is_dir():
        print(f"  source dir exists, fetching {commit}...")
        _run(["git", "fetch", "origin", commit], cwd=target_dir)
        _run(["git", "checkout", commit], cwd=target_dir)
    else:
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", f"https://github.com/{repo}.git", str(target_dir)])
        _run(["git", "checkout", commit], cwd=target_dir)

    # Initialize and update submodules.
    _run(["git", "submodule", "update", "--init", "--recursive", "--depth", "1"],
         cwd=target_dir)

    # Verify submodule SHAs against the source lock.
    _verify_submodules(lock, target_dir)


def _verify_submodules(lock: dict[str, Any], source_dir: Path) -> None:
    """Verify that every submodule SHA in the lock matches the checkout."""
    expected = lock.get("submodules", {})
    if not expected:
        return
    actual = _get_submodule_state(source_dir)
    mismatches: list[str] = []
    for path, info in expected.items():
        expected_sha = info["expected_sha"]
        actual_sha = actual.get(path)
        if actual_sha is None:
            mismatches.append(f"{path}: submodule not found in checkout")
        elif actual_sha != expected_sha:
            mismatches.append(
                f"{path}: SHA mismatch (expected {expected_sha}, got {actual_sha})"
            )
    if mismatches:
        for m in mismatches:
            print(f"  SUBMODULE MISMATCH: {m}", file=sys.stderr)
        raise RuntimeError(
            f"submodule verification failed ({len(mismatches)} mismatch(es)); "
            "the source lock does not match the ORT checkout"
        )


def _get_submodule_state(source_dir: Path) -> dict[str, str]:
    """Record the exact SHA of each submodule."""
    result = subprocess.run(
        ["git", "submodule", "status", "--recursive"],
        cwd=str(source_dir),
        capture_output=True,
        text=True,
        check=True,
    )
    submodules: dict[str, str] = {}
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split()
        sha = parts[0].lstrip("+-")
        path = parts[1]
        submodules[path] = sha
    return submodules


def _assert_toolchain(lock: dict[str, Any], target: str) -> None:
    """Assert the local toolchain matches the source lock before building.

    Checks Python version, CMake minimum version, the configured compiler
    version, and Xcode version (on Apple targets). Fails the build on
    mismatch. This runs before any CMake invocation so a wrong toolchain is
    caught early instead of producing a broken library.
    """
    tc = lock.get("toolchain", {})
    errors: list[str] = []

    # Python version.
    required_py = tc.get("python_version", "3.11")
    actual_py = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_py != required_py:
        errors.append(f"python version: expected {required_py}, got {actual_py}")

    # CMake minimum version.
    required_cmake = tc.get("cmake_minimum_version", "3.28")
    try:
        r = subprocess.run(["cmake", "--version"], capture_output=True, text=True)
        if r.returncode != 0:
            errors.append("cmake: not found on PATH")
        else:
            first = r.stdout.strip().splitlines()[0]
            # "cmake version 3.28.3"
            parts = first.split()
            ver = parts[-1] if parts else "0"
            if not _version_ge(ver, required_cmake):
                errors.append(
                    f"cmake version: expected >= {required_cmake}, got {ver}"
                )
    except FileNotFoundError:
        errors.append("cmake: not found on PATH")

    # Xcode version (Apple targets only).
    if "darwin" in target:
        required_xcode = tc.get("xcode_version", "16.0")
        try:
            r = subprocess.run(["xcodebuild", "-version"], capture_output=True,
                               text=True)
            if r.returncode != 0:
                errors.append("xcodebuild: not found (Xcode not installed?)")
            else:
                first = r.stdout.strip().splitlines()[0]
                # "Xcode 16.0"
                parts = first.split()
                ver = parts[-1] if parts else "0"
                if not _version_ge(ver, required_xcode):
                    errors.append(
                        f"xcode version: expected >= {required_xcode}, got {ver}"
                    )
        except FileNotFoundError:
            errors.append("xcodebuild: not found (Xcode not installed?)")

    # Compiler version.
    if "linux" in target:
        required_gcc = tc.get("gcc_version", "13")
        try:
            r = subprocess.run(["gcc", "--version"], capture_output=True, text=True)
            if r.returncode != 0:
                errors.append("gcc: not found on PATH")
            else:
                first = r.stdout.strip().splitlines()[0]
                # "gcc (Ubuntu 13.x.x) 13.x.x"
                m = re.search(r"(\d+)\.", first)
                major = m.group(1) if m else "0"
                if major != str(required_gcc):
                    errors.append(
                        f"gcc major version: expected {required_gcc}, got {major}"
                    )
        except FileNotFoundError:
            errors.append("gcc: not found on PATH")
    elif "windows" in target:
        required_vs = tc.get("visual_studio_version", "2022")
        # On Windows the MSVC compiler (cl.exe) is not on PATH in a plain
        # shell — it is only available after sourcing vcvarsall.bat. CMake's
        # Visual Studio generator finds MSVC via the registry/vswhere without
        # vcvars, so the build itself works, but a direct cl.exe probe fails.
        # Use vswhere (always present on GitHub Windows runners and any VS
        # install) to detect Visual Studio 2022 (version 17.x) instead.
        vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio"
                       r"\Installer\vswhere.exe")
        vs_year = {"2022": "[17.0,18.0)", "2019": "[16.0,17.0)"}
        version_range = vs_year.get(str(required_vs))
        if version_range is None:
            errors.append(
                f"visual_studio_version: unknown required {required_vs}"
            )
        elif not vswhere.is_file():
            errors.append("vswhere.exe: not found (Visual Studio Installer "
                          "absent)")
        else:
            r = subprocess.run(
                [str(vswhere), "-latest", "-version", version_range,
                 "-products", "*", "-property", "displayName"],
                capture_output=True, text=True,
            )
            if r.returncode != 0 or not r.stdout.strip():
                errors.append(
                    f"visual studio: expected {required_vs} "
                    f"(version range {version_range}) not found by vswhere "
                    f"(exit {r.returncode}, stdout={r.stdout.strip()!r})"
                )

    if errors:
        for e in errors:
            print(f"  TOOLCHAIN MISMATCH: {e}", file=sys.stderr)
        raise RuntimeError(
            f"toolchain verification failed ({len(errors)} error(s)); "
            "the local toolchain does not match ort/source-lock.json"
        )


def _version_ge(actual: str, required: str) -> bool:
    """Return True if ``actual`` >= ``required`` as a dotted version."""
    def _key(v: str) -> tuple[int, ...]:
        return tuple(int(p) for p in re.split(r"[.]", v) if p.isdigit())
    return _key(actual) >= _key(required)


def _build_target(lock: dict[str, Any], target: str, source_dir: Path,
                  build_dir: Path, *, reduced: bool = False) -> dict[str, Any]:
    target_config = lock["targets"][target]
    cmake_args = list(target_config["cmake_args"])
    reduced_cfg = None
    if reduced:
        reduced_cfg = lock.get("reduced_build")
        if not reduced_cfg:
            raise RuntimeError(
                "source lock has no reduced_build section; cannot build reduced runtime"
            )
        cmake_args.extend(reduced_cfg.get("extra_cmake_args", []))

    # Assert the toolchain matches the source lock before building.
    _assert_toolchain(lock, target)

    # Derive the ORT C API version from the pinned source header. This
    # replaces the manually maintained c_api_level.ort_api_version value that
    # used to live in source-lock.json. For v1.27.1 the parsed value must be
    # 27; ort_api_version.assert_api_version exits non-zero on mismatch.
    tag = lock["upstream"]["tag"]
    ort_api_version_value = ort_api_version.assert_api_version(
        source_dir, tag, label=f"build:{target}",
    )

    # Set SOURCE_DATE_EPOCH for reproducible builds. This environment
    # variable is respected by many build tools (cmake, gcc, clang, etc.)
    # to normalize embedded timestamps.
    env = os.environ.copy()
    env["SOURCE_DATE_EPOCH"] = str(_ort_commit_timestamp(lock))

    # Configure.
    build_dir.mkdir(parents=True, exist_ok=True)
    configure_cmd = ["cmake", "-S", str(source_dir / "cmake"), "-B", str(build_dir),
                     *cmake_args]
    _run(configure_cmd, env=env)

    # Reduced-operator build: run ORT's reduce_op_kernels.py after configure
    # to prune unused operator kernels from the generated source. This keeps
    # ONNX-format model support, runtime optimization, and EP partitioning;
    # it only removes kernels for operators not in the required-operators config.
    if reduced:
        config_path = ROOT / reduced_cfg["config_path"]
        if not config_path.is_file():
            raise FileNotFoundError(
                f"required-operators config not found: {config_path}. "
                "Run scripts/generate_required_operators.py first."
            )
        script = source_dir / reduced_cfg["reduce_op_kernels_script"]
        if not script.is_file():
            raise FileNotFoundError(
                f"reduce_op_kernels.py not found at {script}; "
                "source checkout may be incomplete"
            )
        _run([sys.executable, str(script),
              "--cmake_build_dir", str(build_dir),
              "--config", str(config_path)], env=env)

    # Build.
    nproc = os.cpu_count() or 4
    build_cmd = ["cmake", "--build", str(build_dir), "--config", "Release",
                 "--parallel", str(nproc)]
    _run(build_cmd, env=env)

    # Collect build metadata. NOTE: build_host (hostname) is intentionally
    # omitted from the manifest — it varies per build host and breaks
    # reproducibility. The OS and arch are kept because they are determined
    # by the target triple and runner, not the specific machine.
    submodules = _get_submodule_state(source_dir)
    meta: dict[str, Any] = {
        "target": target,
        "commit_sha": lock["upstream"]["commit_sha"],
        "tag": lock["upstream"]["tag"],
        "submodule_state": submodules,
        "cmake_args": cmake_args,
        "build_os": f"{platform.system()} {platform.release()}",
        "build_arch": platform.machine(),
        "ort_api_version": ort_api_version_value,
    }
    if reduced:
        meta["reduced_build"] = True
        meta["ops_config_sha256"] = _sha256_file(ROOT / reduced_cfg["config_path"])[1]
    return meta


def _package_target(lock: dict[str, Any], target: str, source_dir: Path,
                    build_dir: Path, build_meta: dict[str, Any]) -> Path:
    target_config = lock["targets"][target]
    artifact_name = target_config["artifact_name"]
    companion_libs = target_config.get("companion_libraries", [])
    archive_format = target_config["archive_format"]

    # Find the built library.
    lib_path = _find_library(build_dir, artifact_name)
    if lib_path is None:
        raise FileNotFoundError(f"built library {artifact_name} not found under {build_dir}")

    # Create staging directory.
    staging = BUILD_DIR / "staging" / target
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    # Copy runtime library.
    shutil.copy2(lib_path, staging / lib_path.name)

    # Copy companion libraries.
    for comp in companion_libs:
        comp_path = _find_library(build_dir, comp)
        if comp_path:
            shutil.copy2(comp_path, staging / comp_path.name)

    # Copy licenses and notices from source.
    license_file = source_dir / "LICENSE"
    if license_file.is_file():
        shutil.copy2(license_file, staging / "LICENSE.onnxruntime")
    # ORT uses ThirdPartyNotices.txt, not NOTICE
    notice_file = source_dir / "NOTICE"
    if not notice_file.is_file():
        notice_file = source_dir / "ThirdPartyNotices.txt"
    if notice_file.is_file():
        shutil.copy2(notice_file, staging / "NOTICE.onnxruntime")
    # Third-party licenses
    tp_licenses = source_dir / "cmake" / "external" / "LICENSES"
    if tp_licenses.is_dir():
        shutil.copytree(tp_licenses, staging / "third-party-licenses", dirs_exist_ok=True)

    # Write build manifest.
    manifest = {
        "schema_version": "openkara.ort-build-manifest/v1",
        "target": target,
        "upstream": lock["upstream"],
        "build": build_meta,
        "toolchain": lock["toolchain"],
        "c_api_level": {
            "ort_api_version": build_meta.get("ort_api_version"),
            "ort_api_version_note": (
                "Parsed at build time from ORT_API_VERSION in "
                "include/onnxruntime/core/session/onnxruntime_c_api.h of the "
                "pinned source."
            ),
        },
        "files": {},
    }

    # Compute digests for all staged files.
    for f in sorted(staging.rglob("*")):
        if f.is_file():
            rel = f.relative_to(staging).as_posix()
            size, sha = _sha256_file(f)
            manifest["files"][rel] = {"size": size, "sha256": sha}

    with (staging / "build-manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")

    # Create archive with deterministic timestamps. All entries use the ORT
    # commit date as their mtime so the same source lock + toolchain produces
    # byte-identical archives across different build hosts and times.
    PACKAGES_DIR.mkdir(parents=True, exist_ok=True)
    version = lock["upstream"]["tag"].lstrip("v")
    archive_base = f"onnxruntime-{version}-openkara-{target}"
    if build_meta.get("reduced_build"):
        archive_base += "-reduced"
    fixed_timestamp = _ort_commit_timestamp(lock)

    if archive_format == "tar.gz":
        archive_path = PACKAGES_DIR / f"{archive_base}.tar.gz"
        _make_tar(archive_path, staging, fixed_timestamp)
    elif archive_format == "zip":
        archive_path = PACKAGES_DIR / f"{archive_base}.zip"
        _make_zip(archive_path, staging, fixed_timestamp)
    else:
        raise ValueError(f"unknown archive format: {archive_format}")

    return archive_path


def _ort_commit_timestamp(lock: dict[str, Any]) -> int:
    """Return a deterministic Unix timestamp for archive entry mtimes.

    We use 315532800 (1980-01-01 00:00:00 UTC), the earliest timestamp
    supported by both the zip and tar formats. Using a fixed timestamp
    ensures the same source lock + toolchain produces byte-identical
    archives across different build hosts and times.
    """
    return 315532800  # 1980-01-01 00:00:00 UTC (zip epoch)


def _make_tar(archive_path: Path, staging: Path, fixed_timestamp: int) -> None:
    """Create a tar.gz archive with deterministic entry timestamps and
    ordering using Python's tarfile (no external tar binary).

    The gzip wrapper uses mtime=fixed_timestamp and an empty filename in
    its header so the archive is byte-identical across runs regardless of
    the output file path.
    """
    import gzip
    import io
    import tarfile
    # Use BytesIO as the gzip target so no filename is embedded in the
    # gzip header. This ensures byte-identical output regardless of the
    # output file path.
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=fixed_timestamp, filename="") as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for f in sorted(staging.rglob("*")):
                if f.is_file():
                    rel = f.relative_to(staging).as_posix()
                    info = tarfile.TarInfo(name=rel)
                    info.size = f.stat().st_size
                    info.mtime = fixed_timestamp
                    info.mode = 0o644
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    with f.open("rb") as fh:
                        tar.addfile(info, fh)
    archive_path.write_bytes(buf.getvalue())


def _make_zip(archive_path: Path, staging: Path, fixed_timestamp: int) -> None:
    """Create a zip archive with deterministic entry timestamps and ordering."""
    import zipfile
    from datetime import datetime, timezone
    fixed_dt = datetime.fromtimestamp(fixed_timestamp, tz=timezone.utc)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(staging.rglob("*")):
            if f.is_file():
                rel = f.relative_to(staging).as_posix()
                info = zipfile.ZipInfo(filename=rel, date_time=fixed_dt.timetuple()[:6])
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                with f.open("rb") as fh:
                    zf.writestr(info, fh.read())


def _find_library(build_dir: Path, name: str) -> Path | None:
    """Search for a library file in the build directory."""
    for pattern in [f"**/{name}", f"**/Release/{name}", f"**/{name}.dylib",
                    f"**/{name}.so", f"**/{name}.dll"]:
        matches = list(build_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ORT from source lock.")
    parser.add_argument("--target", required=True,
                        choices=["aarch64-apple-darwin", "x86_64-apple-darwin",
                                 "x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu",
                                 "x86_64-pc-windows-msvc"],
                        help="Target triple to build.")
    parser.add_argument("--skip-clone", action="store_true",
                        help="Skip cloning; use existing ort/source/ checkout.")
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR,
                        help="Source directory (default: ort/source/).")
    parser.add_argument("--build-dir", type=Path, default=None,
                        help="Build directory (default: ort/build/<target>/).")
    parser.add_argument("--locked", action="store_true",
                        help="Verify that the checked-out source tree and "
                             "submodules exactly match the source lock before "
                             "building. Required by the verification commands "
                             "(verify_runtime_package.py, CI).")
    parser.add_argument("--reduced", action="store_true",
                        help="Build a reduced-operator runtime. Requires "
                             "ort/required-operators.config (generate with "
                             "scripts/generate_required_operators.py). Prunes "
                             "unused operator kernels via reduce_op_kernels.py "
                             "while retaining ONNX-format, runtime optimization, "
                             "and EP partitioning support.")
    args = parser.parse_args()

    lock = _load_lock()
    source_dir = args.source_dir
    build_dir = args.build_dir or (BUILD_DIR / args.target)
    if args.reduced:
        build_dir = build_dir.parent / f"{args.target}-reduced" if build_dir.name == args.target else build_dir

    if not args.skip_clone:
        print(f"Cloning ORT {lock['upstream']['tag']} ({lock['upstream']['commit_sha'][:12]})...")
        _clone_source(lock, source_dir)
    elif not source_dir.exists():
        print(f"ERROR: source dir not found: {source_dir}", file=sys.stderr)
        return 1

    # --locked: verify the upstream commit and all submodules match the lock
    # exactly. This catches drift when --skip-clone is used with a stale or
    # manually-modified checkout.
    if args.locked:
        print("Verifying source lock...")
        if not _verify_upstream_commit(lock, source_dir):
            return 1
        _verify_submodules(lock, source_dir)
        print("  source lock verified.")

    print(f"Building for {args.target}{' (reduced)' if args.reduced else ''}...")
    build_meta = _build_target(lock, args.target, source_dir, build_dir, reduced=args.reduced)

    print(f"Packaging {args.target}...")
    archive_path = _package_target(lock, args.target, source_dir, build_dir, build_meta)

    archive_size, archive_sha = _sha256_file(archive_path)
    print(f"OK: {args.target}")
    print(f"  archive: {archive_path}")
    print(f"  size:    {archive_size} bytes")
    print(f"  sha256:  {archive_sha}")
    return 0


def _verify_upstream_commit(lock: dict[str, Any], source_dir: Path) -> bool:
    """Verify the checked-out HEAD matches the source lock's commit_sha."""
    expected = lock["upstream"]["commit_sha"]
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(source_dir),
        capture_output=True,
        text=True,
        check=True,
    )
    actual = r.stdout.strip()
    if actual != expected:
        print(
            f"ERROR: upstream commit mismatch (expected {expected}, got {actual})",
            file=sys.stderr,
        )
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
