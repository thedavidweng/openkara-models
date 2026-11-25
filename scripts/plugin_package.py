#!/usr/bin/env python3
"""Download, lock, and compatibility-verify ORT WebGPU plugin wheels from PyPI.

Manages ``ort/plugin-lock.json``, which pins the ``onnxruntime-ep-webgpu``
wheel URLs, byte sizes, SHA-256 digests, and per-member digests for the
catalog's supported WebGPU plugin targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import archive_utils  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "ort" / "plugin-lock.json"
NUGET_LOCK_PATH = ROOT / "ort" / "nuget-lock.json"

PACKAGE_NAME = "onnxruntime-ep-webgpu"
DIST_PREFIX = "onnxruntime_ep_webgpu"
PROVIDER = "webgpu"
EP_NAME = "WebGpuExecutionProvider"
REGISTRATION_NAME = "webgpu"
MIN_PLUGIN_EP_API_LEVEL = 24
PYPI_PROJECT_URL = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"

TARGET_SPECS: dict[str, dict[str, str]] = {
    "aarch64-apple-darwin": {
        "wheel_tag": "py3-none-macosx_14_0_universal2",
        "library_member": "onnxruntime_ep_webgpu/libonnxruntime_providers_webgpu.dylib",
    },
    "x86_64-pc-windows-msvc": {
        "wheel_tag": "py3-none-win_amd64",
        "library_member": "onnxruntime_ep_webgpu/onnxruntime_providers_webgpu.dll",
    },
}

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WHEELHOST_RE = re.compile(r"^https://files\.pythonhosted\.org/packages/[0-9a-f]{2}/[0-9a-f]{2}/[0-9a-f]{60}/[^/?#]+\.whl$")


def parse_semver(version: str) -> tuple[int, int, int]:
    match = _VERSION_RE.fullmatch(version)
    if match is None:
        raise ValueError(f"version must be X.Y.Z, got {version!r}")
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def pypi_version_url(version: str) -> str:
    parse_semver(version)
    return f"https://pypi.org/pypi/{PACKAGE_NAME}/{version}/json"


def fetch_pypi_json(version: str | None = None) -> dict[str, Any]:
    url = pypi_version_url(version) if version is not None else PYPI_PROJECT_URL
    req = urllib.request.Request(url, headers={"User-Agent": "openkara-models-plugin-package"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"PyPI error {e.code}: {e.reason} ({url})") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"network error fetching PyPI metadata: {e.reason}") from e
    if not isinstance(data, dict):
        raise RuntimeError(f"unexpected PyPI response from {url}")
    return data


def load_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> tuple[int, str]:
    with path.open("rb") as stream:
        return path.stat().st_size, hashlib.file_digest(stream, "sha256").hexdigest()


def download_verified_bytes(url: str, dest: Path, size: int, sha256: str) -> Path:
    """Download ``url`` to ``dest`` and enforce exact size and SHA-256."""
    if dest.is_file() and sha256_file(dest) == (size, sha256):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=300) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
        actual = sha256_file(partial)
        if actual != (size, sha256):
            raise ValueError(
                f"{url}: expected {size} bytes sha256 {sha256}, got {actual[0]} bytes sha256 {actual[1]}"
            )
        partial.replace(dest)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    return dest


def download_verified_wheel(
    entry: dict[str, Any], dest_dir: Path, *, cache_dir: Path | None = None
) -> Path:
    """Download a locked wheel into ``dest_dir``, verifying archive and member digests."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / entry["filename"]
    if cache_dir is not None:
        cached = cache_dir / entry["filename"]
        if cached.is_file() and sha256_file(cached) == (entry["size"], entry["sha256"]):
            if dest != cached:
                shutil.copy2(cached, dest)
            _verify_members(dest, entry["members"])
            return dest
    download_verified_bytes(entry["url"], dest, entry["size"], entry["sha256"])
    _verify_members(dest, entry["members"])
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / entry["filename"]
        if cached != dest:
            shutil.copy2(dest, cached)
    return dest


def _verify_members(wheel_path: Path, expected_members: dict[str, dict[str, Any]]) -> None:
    files = archive_utils.safe_read_archive(wheel_path)
    if set(files) != set(expected_members):
        missing = sorted(set(expected_members) - set(files))
        extra = sorted(set(files) - set(expected_members))
        raise ValueError(f"{wheel_path.name}: wheel member mismatch (missing={missing}, extra={extra})")
    for name, raw in files.items():
        digest = hashlib.sha256(raw).hexdigest()
        expected = expected_members[name]
        if (len(raw), digest) != (expected["size"], expected["sha256"]):
            raise ValueError(
                f"{wheel_path.name}: member {name} mismatch "
                f"(expected {expected['size']} {expected['sha256']}, got {len(raw)} {digest})"
            )


def _pe_exports(data: bytes) -> tuple[int, list[str]]:
    if len(data) < 0x40 or data[:2] != b"MZ":
        raise ValueError("plugin DLL is not a valid MZ executable")
    pe_off = struct.unpack_from("<I", data, 0x3C)[0]
    if pe_off + 24 > len(data) or data[pe_off : pe_off + 4] != b"PE\0\0":
        raise ValueError("plugin DLL is missing PE header")
    machine, num_sections, _, _, _, opt_size, _ = struct.unpack_from("<HHIIIHH", data, pe_off + 4)
    opt_off = pe_off + 24
    if opt_off + opt_size > len(data):
        raise ValueError("plugin DLL optional header truncated")
    magic = struct.unpack_from("<H", data, opt_off)[0]
    if magic != 0x20B:
        raise ValueError(f"plugin DLL is not PE32+ (magic={hex(magic)})")
    exp_rva, _ = struct.unpack_from("<II", data, opt_off + 112)
    if exp_rva == 0:
        raise ValueError("plugin DLL has no export directory")
    sec_off = opt_off + opt_size
    sections = []
    for i in range(num_sections):
        s_off = sec_off + i * 40
        vsize, va, rsize, rptr = struct.unpack_from("<IIII", data, s_off + 8)
        sections.append((va, max(vsize, rsize), rptr))

    def rva_to_off(rva: int) -> int:
        for va, span, rptr in sections:
            if va <= rva < va + span:
                return rptr + (rva - va)
        raise ValueError(f"unmapped PE export RVA {hex(rva)}")

    exp_off = rva_to_off(exp_rva)
    num_names = struct.unpack_from("<I", data, exp_off + 24)[0]
    names_rva = struct.unpack_from("<I", data, exp_off + 32)[0]
    names_off = rva_to_off(names_rva)
    names: list[str] = []
    for i in range(num_names):
        nrva = struct.unpack_from("<I", data, names_off + i * 4)[0]
        noff = rva_to_off(nrva)
        end = data.index(b"\0", noff)
        names.append(data[noff:end].decode("ascii", errors="replace"))
    return machine, names


def _macho_arm64_exports(data: bytes) -> set[str]:
    if len(data) < 32:
        raise ValueError("plugin dylib is truncated")
    magic_be = struct.unpack_from(">I", data, 0)[0]
    slices: list[tuple[int, bytes]] = []
    if magic_be == 0xCAFEBABE:
        nfat = struct.unpack_from(">I", data, 4)[0]
        for i in range(nfat):
            cputype, _, offset, size, _ = struct.unpack_from(">IIIII", data, 8 + i * 20)
            slices.append((cputype, data[offset : offset + size]))
    elif struct.unpack_from("<I", data, 0)[0] == 0xFEEDFACF:
        cputype = struct.unpack_from("<I", data, 4)[0]
        slices.append((cputype, data))
    else:
        raise ValueError("plugin dylib is not a valid Mach-O binary")

    arm64_slices = [sdata for cputype, sdata in slices if cputype == 0x0100000C]
    if not arm64_slices:
        raise ValueError("plugin dylib does not contain an arm64 slice")

    sdata = arm64_slices[0]
    if struct.unpack_from("<I", sdata, 0)[0] != 0xFEEDFACF:
        raise ValueError("arm64 slice is not 64-bit Mach-O")
    ncmds = struct.unpack_from("<I", sdata, 16)[0]
    off = 32
    exported: set[str] = set()
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from("<II", sdata, off)
        if cmd == 0x2:  # LC_SYMTAB
            symoff, nsyms, stroff, strsize = struct.unpack_from("<IIII", sdata, off + 8)
            strtab = sdata[stroff : stroff + strsize]
            for idx in range(nsyms):
                nlist_off = symoff + idx * 16
                n_strx, n_type, n_sect, _, _ = struct.unpack_from("<IBBHQ", sdata, nlist_off)
                if (n_type & 0x01) and n_sect != 0 and n_strx < len(strtab):
                    end = strtab.index(b"\0", n_strx)
                    exported.add(strtab[n_strx:end].decode("ascii", errors="replace"))
        off += cmdsize
    return exported


def _extract_min_ort_version(readme_text: str, lib_bytes: bytes) -> str:
    readme_match = re.search(
        r"onnxruntime[)`\s]+of\s+version\s+`?(\d+\.\d+\.\d+)`?\s+or\s+later",
        readme_text,
        re.IGNORECASE,
    )
    marker = b"is below the minimum required version \""
    binary_min: str | None = None
    idx = lib_bytes.find(marker)
    if idx != -1:
        window = lib_bytes[max(0, idx - 64) : idx + 128]
        ver_match = re.search(rb"\b(\d+\.\d+\.\d+)\b", window)
        if ver_match:
            binary_min = ver_match.group(1).decode("ascii")
    if readme_match and binary_min and readme_match.group(1) != binary_min:
        raise ValueError(
            f"README minimum ORT version ({readme_match.group(1)}) disagrees with binary ({binary_min})"
        )
    if binary_min:
        return binary_min
    if readme_match:
        return readme_match.group(1)
    raise ValueError("could not determine minimum required ORT version from plugin wheel")


def check_wheel_compatibility(
    wheel_path: Path,
    target: str,
    version: str,
    locked_ort_version: str,
    locked_ort_c_api_level: int,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify a plugin wheel's hash, structure, native EP exports, and ORT compatibility."""
    if target not in TARGET_SPECS:
        raise ValueError(f"unsupported plugin target: {target!r}")
    parse_semver(version)
    locked_ort_tuple = parse_semver(locked_ort_version)
    if locked_ort_c_api_level < MIN_PLUGIN_EP_API_LEVEL:
        raise ValueError(
            f"locked ORT C API level {locked_ort_c_api_level} is below minimum plugin EP level {MIN_PLUGIN_EP_API_LEVEL}"
        )

    actual_size, actual_sha256 = sha256_file(wheel_path)
    if expected_size is not None and actual_size != expected_size:
        raise ValueError(f"{wheel_path.name}: size mismatch (expected {expected_size}, got {actual_size})")
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(f"{wheel_path.name}: sha256 mismatch (expected {expected_sha256}, got {actual_sha256})")

    files = archive_utils.safe_read_archive(wheel_path)
    dist_info = f"{DIST_PREFIX}-{version}.dist-info"
    metadata_path = f"{dist_info}/METADATA"
    wheel_info_path = f"{dist_info}/WHEEL"
    init_path = f"{DIST_PREFIX}/__init__.py"
    readme_path = f"{DIST_PREFIX}/README.md"
    spec = TARGET_SPECS[target]
    library_member = spec["library_member"]

    for required in (metadata_path, wheel_info_path, init_path, readme_path, library_member):
        if required not in files:
            raise ValueError(f"{wheel_path.name}: missing required wheel member {required}")

    metadata = files[metadata_path].decode("utf-8")
    name_match = re.search(r"^Name:\s*(.+)$", metadata, re.MULTILINE)
    ver_match = re.search(r"^Version:\s*(.+)$", metadata, re.MULTILINE)
    req_py_match = re.search(r"^Requires-Python:\s*(.+)$", metadata, re.MULTILINE)
    if not name_match or name_match.group(1).strip().lower() != PACKAGE_NAME:
        raise ValueError(f"{wheel_path.name}: unexpected METADATA Name")
    if not ver_match or ver_match.group(1).strip() != version:
        raise ValueError(f"{wheel_path.name}: unexpected METADATA Version")
    requires_python = req_py_match.group(1).strip() if req_py_match else ""

    wheel_txt = files[wheel_info_path].decode("utf-8")
    tags = [m.group(1).strip() for m in re.finditer(r"^Tag:\s*(.+)$", wheel_txt, re.MULTILINE)]
    if spec["wheel_tag"] not in tags:
        raise ValueError(f"{wheel_path.name}: WHEEL tags {tags!r} missing {spec['wheel_tag']!r}")

    init_txt = files[init_path].decode("utf-8")
    if EP_NAME not in init_txt:
        raise ValueError(f"{wheel_path.name}: __init__.py does not declare {EP_NAME}")

    lib_bytes = files[library_member]
    if EP_NAME.encode("ascii") not in lib_bytes:
        raise ValueError(f"{wheel_path.name}: {library_member} missing {EP_NAME} symbol/string")

    if target == "x86_64-pc-windows-msvc":
        machine, exports = _pe_exports(lib_bytes)
        if machine != 0x8664:
            raise ValueError(f"{wheel_path.name}: PE machine {hex(machine)} != 0x8664 (AMD64)")
        missing_exports = {"CreateEpFactories", "ReleaseEpFactory"} - set(exports)
        if missing_exports:
            raise ValueError(f"{wheel_path.name}: missing DLL exports {sorted(missing_exports)}")
    elif target == "aarch64-apple-darwin":
        exports = _macho_arm64_exports(lib_bytes)
        missing_exports = {"_CreateEpFactories", "_ReleaseEpFactory"} - exports
        if missing_exports:
            raise ValueError(f"{wheel_path.name}: missing dylib exports {sorted(missing_exports)}")

    readme_txt = files[readme_path].decode("utf-8")
    min_ort_version = _extract_min_ort_version(readme_txt, lib_bytes)
    if locked_ort_tuple < parse_semver(min_ort_version):
        raise ValueError(
            f"{wheel_path.name}: requires ORT >= {min_ort_version}, but locked ORT is {locked_ort_version}"
        )

    members = {
        name: {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        for name, raw in sorted(files.items())
    }
    return {
        "size": actual_size,
        "sha256": actual_sha256,
        "requires_python": requires_python,
        "min_ort_version": min_ort_version,
        "library_member": library_member,
        "ep_name": EP_NAME,
        "members": members,
    }


def validate_lock(
    lock: dict[str, Any], nuget_lock: dict[str, Any] | None = None
) -> list[str]:
    """Validate ort/plugin-lock.json structure, digests, and ORT compatibility."""
    errors: list[str] = []
    if lock.get("schema_version") != "openkara.ort-plugin-lock/v1":
        errors.append(f"unexpected schema_version: {lock.get('schema_version')!r}")

    pkg = lock.get("package", {})
    if pkg.get("name") != PACKAGE_NAME:
        errors.append(f"package.name must be {PACKAGE_NAME!r}, got {pkg.get('name')!r}")
    if pkg.get("provider") != PROVIDER:
        errors.append(f"package.provider must be {PROVIDER!r}, got {pkg.get('provider')!r}")
    version = pkg.get("version", "")
    try:
        parse_semver(version)
    except ValueError as e:
        errors.append(f"package.version invalid: {e}")
    else:
        expected_pypi = pypi_version_url(version)
        if pkg.get("pypi_url") != expected_pypi:
            errors.append(f"package.pypi_url must be {expected_pypi!r}, got {pkg.get('pypi_url')!r}")
    min_ort = pkg.get("min_ort_version", "")
    try:
        min_ort_tuple = parse_semver(min_ort)
    except ValueError as e:
        errors.append(f"package.min_ort_version invalid: {e}")
        min_ort_tuple = None

    compat = lock.get("compatibility", {})
    if compat.get("ep_name") != EP_NAME:
        errors.append(f"compatibility.ep_name must be {EP_NAME!r}, got {compat.get('ep_name')!r}")
    if compat.get("registration_name") != REGISTRATION_NAME:
        errors.append(
            f"compatibility.registration_name must be {REGISTRATION_NAME!r}, got {compat.get('registration_name')!r}"
        )
    api_level = compat.get("ort_c_api_level")
    if type(api_level) is not int or api_level < MIN_PLUGIN_EP_API_LEVEL:
        errors.append(
            f"compatibility.ort_c_api_level must be an int >= {MIN_PLUGIN_EP_API_LEVEL}, got {api_level!r}"
        )
    ort_version = compat.get("ort_version", "")
    try:
        ort_tuple = parse_semver(ort_version)
    except ValueError as e:
        errors.append(f"compatibility.ort_version invalid: {e}")
        ort_tuple = None
    if ort_tuple is not None and min_ort_tuple is not None and ort_tuple < min_ort_tuple:
        errors.append(
            f"compatibility.ort_version {ort_version!r} is below package.min_ort_version {min_ort!r}"
        )

    if nuget_lock is not None:
        locked_ort = nuget_lock.get("package", {}).get("version", "")
        try:
            locked_tuple = parse_semver(locked_ort)
        except ValueError as e:
            errors.append(f"nuget lock package.version invalid: {e}")
        else:
            if min_ort_tuple is not None and locked_tuple < min_ort_tuple:
                errors.append(
                    f"locked ORT {locked_ort!r} is below plugin min_ort_version {min_ort!r}"
                )
            if ort_tuple is not None and ort_tuple > locked_tuple:
                errors.append(
                    f"plugin compatibility.ort_version {ort_version!r} exceeds locked ORT {locked_ort!r}"
                )

    targets = lock.get("targets")
    if not isinstance(targets, dict) or set(targets) != set(TARGET_SPECS):
        errors.append(f"targets must be {sorted(TARGET_SPECS)}, got {sorted(targets or {})}")
        return errors

    for target, spec in TARGET_SPECS.items():
        entry = targets[target]
        if entry.get("wheel_tag") != spec["wheel_tag"]:
            errors.append(f"targets.{target}.wheel_tag must be {spec['wheel_tag']!r}")
        expected_filename = f"{DIST_PREFIX}-{version}-{spec['wheel_tag']}.whl" if _VERSION_RE.fullmatch(version) else ""
        if expected_filename and entry.get("filename") != expected_filename:
            errors.append(f"targets.{target}.filename must be {expected_filename!r}, got {entry.get('filename')!r}")
        url = entry.get("url", "")
        if not _WHEELHOST_RE.fullmatch(url) or not url.endswith("/" + entry.get("filename", "")):
            errors.append(f"targets.{target}.url must be a pythonhosted URL ending in filename, got {url!r}")
        if type(entry.get("size")) is not int or entry["size"] <= 0:
            errors.append(f"targets.{target}.size must be a positive int")
        if not _SHA256_RE.fullmatch(entry.get("sha256", "")):
            errors.append(f"targets.{target}.sha256 must be 64 lowercase hex chars")
        lib_member = entry.get("library_member")
        if lib_member != spec["library_member"]:
            errors.append(f"targets.{target}.library_member must be {spec['library_member']!r}")
        members = entry.get("members")
        if not isinstance(members, dict) or lib_member not in members:
            errors.append(f"targets.{target}.members missing {lib_member!r}")
        else:
            for member_name, digest in members.items():
                if not isinstance(digest, dict) or type(digest.get("size")) is not int or digest["size"] < 0:
                    errors.append(f"targets.{target}.members.{member_name} invalid size")
                if not isinstance(digest, dict) or not _SHA256_RE.fullmatch(digest.get("sha256", "")):
                    errors.append(f"targets.{target}.members.{member_name} invalid sha256")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=LOCK_PATH, help="Path to ort/plugin-lock.json.")
    parser.add_argument("--validate-lock", action="store_true", help="Validate ort/plugin-lock.json and exit.")
    parser.add_argument("--download", type=Path, help="Download and verify all locked plugin wheels into DIR.")
    parser.add_argument("--target", action="append", help="Target triple to download (defaults to all locked targets).")
    parser.add_argument("--cache", type=Path, default=None, help="Optional wheel cache directory.")
    args = parser.parse_args()

    lock = load_lock(args.lock)
    nuget_lock = json.loads(NUGET_LOCK_PATH.read_text(encoding="utf-8")) if NUGET_LOCK_PATH.is_file() else None

    if args.validate_lock:
        errors = validate_lock(lock, nuget_lock)
        if errors:
            for e in errors:
                print(f"ERROR: {e}", file=sys.stderr)
            return 1
        print(
            f"OK: plugin lock valid ({len(lock['targets'])} targets, "
            f"{lock['package']['name']} {lock['package']['version']})"
        )
        return 0

    if not args.download:
        parser.error("provide --validate-lock or --download DIR")

    errors = validate_lock(lock, nuget_lock)
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    targets = args.target or sorted(lock["targets"])
    unknown = sorted(set(targets) - set(lock["targets"]))
    if unknown:
        parser.error(f"unknown target(s): {', '.join(unknown)}")

    for target in targets:
        wheel = download_verified_wheel(lock["targets"][target], args.download, cache_dir=args.cache)
        size, sha256 = sha256_file(wheel)
        print(f"{target}: {wheel.name} {size} {sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
