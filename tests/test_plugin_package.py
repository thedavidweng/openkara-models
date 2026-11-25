"""Tests for ort/plugin-lock.json and scripts/plugin_package.py."""

from __future__ import annotations

import hashlib
import io
import json
import struct
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import plugin_package  # noqa: E402


def _make_pe32plus_dll(exports: list[str], *, machine: int = 0x8664) -> bytes:
    """Construct a minimal PE32+ DLL with an export directory."""
    pe_off = 0x80
    opt_off = pe_off + 24
    opt_size = 240
    sec_off = opt_off + opt_size
    sec_size = 40
    edata_raw = 0x200
    edata_rva = 0x1000

    names_sorted = sorted(exports)
    names_bytes = b"".join(n.encode("ascii") + b"\0" for n in names_sorted)
    name_ptrs_off = 40
    names_str_off = name_ptrs_off + 4 * len(names_sorted)
    edata_buf = bytearray(names_str_off + len(names_bytes))
    struct.pack_into("<I", edata_buf, 24, len(names_sorted))
    struct.pack_into("<I", edata_buf, 32, edata_rva + name_ptrs_off)
    cur = names_str_off
    for i, name in enumerate(names_sorted):
        struct.pack_into("<I", edata_buf, name_ptrs_off + i * 4, edata_rva + cur)
        raw = name.encode("ascii") + b"\0"
        edata_buf[cur : cur + len(raw)] = raw
        cur += len(raw)

    buf = bytearray(edata_raw + len(edata_buf))
    buf[:2] = b"MZ"
    struct.pack_into("<I", buf, 0x3C, pe_off)
    buf[pe_off : pe_off + 4] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", buf, pe_off + 4, machine, 1, 0, 0, 0, opt_size, 0x2000)
    struct.pack_into("<H", buf, opt_off, 0x20B)
    struct.pack_into("<II", buf, opt_off + 112, edata_rva, len(edata_buf))
    buf[sec_off : sec_off + 8] = b".edata\0\0"
    struct.pack_into("<IIII", buf, sec_off + 8, len(edata_buf), edata_rva, len(edata_buf), edata_raw)
    buf[edata_raw : edata_raw + len(edata_buf)] = edata_buf
    return bytes(buf)


def _make_macho_arm64(exports: list[str], *, fat: bool = True) -> bytes:
    """Construct a minimal 64-bit arm64 Mach-O (optionally wrapped in a fat binary)."""
    strtab = bytearray(b"\0")
    nlists = bytearray()
    for sym in exports:
        strx = len(strtab)
        strtab.extend(sym.encode("ascii") + b"\0")
        nlists.extend(struct.pack("<IBBHQ", strx, 0x0F, 1, 0, 0))

    header_size = 32
    symtab_cmd_size = 24
    symoff = header_size + symtab_cmd_size
    stroff = symoff + len(nlists)
    macho = bytearray(stroff + len(strtab))
    struct.pack_into("<IIIIIIII", macho, 0, 0xFEEDFACF, 0x0100000C, 0, 6, 1, symtab_cmd_size, 0, 0)
    struct.pack_into("<IIIIII", macho, header_size, 0x2, symtab_cmd_size, symoff, len(exports), stroff, len(strtab))
    macho[symoff : symoff + len(nlists)] = nlists
    macho[stroff : stroff + len(strtab)] = strtab
    if not fat:
        return bytes(macho)

    fat_hdr_size = 8 + 20
    fat_buf = bytearray(fat_hdr_size + len(macho))
    struct.pack_into(">II", fat_buf, 0, 0xCAFEBABE, 1)
    struct.pack_into(">IIIII", fat_buf, 8, 0x0100000C, 0, fat_hdr_size, len(macho), 14)
    fat_buf[fat_hdr_size:] = macho
    return bytes(fat_buf)


def _build_wheel(
    path: Path,
    *,
    version: str,
    target: str,
    min_ort: str = "1.24.4",
    ep_name: str = "WebGpuExecutionProvider",
    lib_bytes: bytes | None = None,
) -> Path:
    spec = plugin_package.TARGET_SPECS[target]
    dist_info = f"{plugin_package.DIST_PREFIX}-{version}.dist-info"
    if lib_bytes is None:
        if target == "x86_64-pc-windows-msvc":
            lib_bytes = _make_pe32plus_dll(["CreateEpFactories", "ReleaseEpFactory"])
        else:
            lib_bytes = _make_macho_arm64(["_CreateEpFactories", "_ReleaseEpFactory"])
        lib_bytes += f'\0{ep_name}\0is below the minimum required version "{min_ort}"\0'.encode("ascii")

    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{dist_info}/METADATA", f"Metadata-Version: 2.1\nName: onnxruntime-ep-webgpu\nVersion: {version}\nRequires-Python: >=3.11\n")
        z.writestr(f"{dist_info}/WHEEL", f"Wheel-Version: 1.0\nTag: {spec['wheel_tag']}\n")
        z.writestr(f"{plugin_package.DIST_PREFIX}/__init__.py", f'def get_ep_name():\n    return "{ep_name}"\n')
        z.writestr(f"{plugin_package.DIST_PREFIX}/README.md", f"Requires `onnxruntime` of version `{min_ort}` or later.\n")
        z.writestr(spec["library_member"], lib_bytes)
    return path


def test_committed_plugin_lock_validates_against_nuget_lock():
    lock = plugin_package.load_lock()
    nuget_lock = json.loads((ROOT / "ort/nuget-lock.json").read_text(encoding="utf-8"))
    assert plugin_package.validate_lock(lock, nuget_lock) == []


def test_pe_and_macho_export_parsers():
    pe = _make_pe32plus_dll(["CreateEpFactories", "ReleaseEpFactory"])
    machine, exports = plugin_package._pe_exports(pe)
    assert machine == 0x8664
    assert set(exports) == {"CreateEpFactories", "ReleaseEpFactory"}

    macho_fat = _make_macho_arm64(["_CreateEpFactories", "_ReleaseEpFactory"], fat=True)
    assert {"_CreateEpFactories", "_ReleaseEpFactory"} <= plugin_package._macho_arm64_exports(macho_fat)

    macho_thin = _make_macho_arm64(["_CreateEpFactories"], fat=False)
    assert plugin_package._macho_arm64_exports(macho_thin) == {"_CreateEpFactories"}


def test_check_wheel_compatibility_accepts_valid_wheels(tmp_path):
    win_whl = _build_wheel(tmp_path / "win.whl", version="0.6.0", target="x86_64-pc-windows-msvc")
    mac_whl = _build_wheel(tmp_path / "mac.whl", version="0.6.0", target="aarch64-apple-darwin")
    win_info = plugin_package.check_wheel_compatibility(
        win_whl, "x86_64-pc-windows-msvc", "0.6.0", "1.32.0", 32
    )
    mac_info = plugin_package.check_wheel_compatibility(
        mac_whl, "aarch64-apple-darwin", "0.6.0", "1.32.0", 32
    )
    assert win_info["min_ort_version"] == "1.24.4"
    assert mac_info["min_ort_version"] == "1.24.4"


def test_check_wheel_compatibility_rejects_missing_exports_and_ort_mismatch(tmp_path):
    bad_pe = _make_pe32plus_dll(["CreateEpFactories"]) + b'\0WebGpuExecutionProvider\0is below the minimum required version "1.24.4"\0'
    bad_whl = _build_wheel(tmp_path / "bad.whl", version="0.6.0", target="x86_64-pc-windows-msvc", lib_bytes=bad_pe)
    with pytest.raises(ValueError, match="missing DLL exports"):
        plugin_package.check_wheel_compatibility(bad_whl, "x86_64-pc-windows-msvc", "0.6.0", "1.32.0", 32)

    newer_ort_whl = _build_wheel(tmp_path / "newer.whl", version="0.6.0", target="aarch64-apple-darwin", min_ort="1.99.0")
    with pytest.raises(ValueError, match="requires ORT >= 1.99.0"):
        plugin_package.check_wheel_compatibility(newer_ort_whl, "aarch64-apple-darwin", "0.6.0", "1.32.0", 32)

    with pytest.raises(ValueError, match="below minimum plugin EP level"):
        plugin_package.check_wheel_compatibility(newer_ort_whl, "aarch64-apple-darwin", "0.6.0", "1.99.0", 23)


def test_validate_lock_rejects_bad_digest_and_incompatible_ort():
    lock = plugin_package.load_lock()
    nuget_lock = json.loads((ROOT / "ort/nuget-lock.json").read_text(encoding="utf-8"))
    lock["targets"]["aarch64-apple-darwin"]["sha256"] = "bad"
    assert any("sha256" in e for e in plugin_package.validate_lock(lock, nuget_lock))

    lock = plugin_package.load_lock()
    lock["package"]["min_ort_version"] = "2.0.0"
    assert any("below plugin min_ort_version" in e for e in plugin_package.validate_lock(lock, nuget_lock))
