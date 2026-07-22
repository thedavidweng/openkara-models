"""Tests for the full-vs-reduced runtime comparison script (size/file logic).

The inference comparison requires a loadable runtime + model and runs on CI;
these tests cover the archive size/file-delta logic with synthetic archives.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _make_tar(path: Path, files: dict[str, bytes]) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            import io as _io
            tar.addfile(info, _io.BytesIO(data))
    path.write_bytes(buf.getvalue())


def _make_zip(path: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in sorted(files.items()):
            zf.writestr(name, data)


def test_size_comparison_full_vs_reduced(tmp_path: Path) -> None:
    sys.path.insert(0, str(SCRIPTS))
    import compare_runtime_builds as cmp
    full = tmp_path / "full.tar.gz"
    reduced = tmp_path / "reduced.tar.gz"
    _make_tar(full, {
        "libonnxruntime.so": b"x" * 1000,
        "LICENSE.onnxruntime": b"mit",
        "build-manifest.json": b"{}",
    })
    _make_tar(reduced, {
        "libonnxruntime.so": b"x" * 600,
        "LICENSE.onnxruntime": b"mit",
        "build-manifest.json": b"{}",
    })
    sz = cmp.size_comparison(full, reduced)
    assert sz["full"]["installed_bytes"] == 1000 + 3 + 2
    assert sz["reduced"]["installed_bytes"] == 600 + 3 + 2
    assert sz["installed_size_delta_bytes"] == -400
    assert sz["library_size_delta_bytes"] == -400
    assert sz["files_only_in_full"] == []
    assert sz["files_only_in_reduced"] == []


def test_size_comparison_file_delta(tmp_path: Path) -> None:
    sys.path.insert(0, str(SCRIPTS))
    import compare_runtime_builds as cmp
    full = tmp_path / "full.tar.gz"
    reduced = tmp_path / "reduced.tar.gz"
    _make_tar(full, {
        "libonnxruntime.so": b"x" * 100,
        "extra-provider.so": b"y" * 50,
        "LICENSE.onnxruntime": b"m",
    })
    _make_tar(reduced, {
        "libonnxruntime.so": b"x" * 80,
        "LICENSE.onnxruntime": b"m",
        "new-file.txt": b"z",
    })
    sz = cmp.size_comparison(full, reduced)
    assert sz["files_only_in_full"] == ["extra-provider.so"]
    assert sz["files_only_in_reduced"] == ["new-file.txt"]


def test_find_lib(tmp_path: Path) -> None:
    sys.path.insert(0, str(SCRIPTS))
    import compare_runtime_builds as cmp
    assert cmp._find_lib({"a.txt": b"x", "libonnxruntime.so": b"y"}) == "libonnxruntime.so"
    assert cmp._find_lib({"libonnxruntime.dylib": b"y"}) == "libonnxruntime.dylib"
    assert cmp._find_lib({"onnxruntime.dll": b"y"}) == "onnxruntime.dll"
    assert cmp._find_lib({"readme.txt": b"y"}) is None


def test_compare_cli_size_only(tmp_path: Path) -> None:
    full = tmp_path / "full.tar.gz"
    reduced = tmp_path / "reduced.tar.gz"
    _make_tar(full, {"libonnxruntime.so": b"x" * 100, "LICENSE.onnxruntime": b"m"})
    _make_tar(reduced, {"libonnxruntime.so": b"x" * 60, "LICENSE.onnxruntime": b"m"})
    report = tmp_path / "report.json"
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "compare_runtime_builds.py"),
         "--full", str(full), "--reduced", str(reduced), "--report", str(report)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    data = json.loads(report.read_text())
    assert data["sizes"]["installed_size_delta_bytes"] == -40
    assert "inference" not in data
