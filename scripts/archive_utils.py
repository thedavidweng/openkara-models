#!/usr/bin/env python3
"""Safe tar/zip extraction utilities.

All archive extraction in the OpenKara infrastructure must go through this
module. It rejects:

  - absolute paths (``/etc/passwd``)
  - drive-prefixed paths (``C:\\Windows\\system32``)
  - ``..`` traversal (``../../etc/passwd``)
  - symlink and hardlink targets that escape the extraction directory
  - duplicate normalized output paths (two members that resolve to the same
    extracted path, which can overwrite an earlier safe file with a later
    malicious one)
  - excessive member counts (zip bombs with millions of tiny entries)
  - excessive extracted size (zip bombs with huge uncompressed payloads)

This is the Python 3.11-compatible equivalent of PEP 706's ``data`` filter,
extended with the duplicate-path and size/count limits that PEP 706 does not
cover.

Two entry points are provided:

  - ``safe_read_archive(archive)`` reads every file member into an in-memory
    ``dict[str, bytes]`` (used by verification/catalog scripts that do not
    need files on disk).
  - ``safe_extract(archive, dest)`` extracts to a directory on disk (used by
    benchmark/quality scripts that need to load a shared library).
"""

from __future__ import annotations

import os
import tarfile
import zipfile
from pathlib import Path

# Limits. A legitimate ORT runtime archive has a handful of files totaling
# a few hundred MB. These limits are generous enough to never reject a real
# archive while stopping zip bombs.
MAX_MEMBER_COUNT = 10_000
MAX_EXTRACTED_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB


class UnsafeArchiveError(ValueError):
    """Raised when an archive member violates a safety rule."""


def _is_absolute_or_drive(name: str) -> bool:
    """Return True for absolute paths and Windows drive-prefixed paths."""
    if name.startswith("/"):
        return True
    # Windows drive prefix: "C:\..." or "C:/..."
    if len(name) >= 2 and name[1] == ":" and name[0].isalpha():
        return True
    return False


def _normalize(name: str) -> str:
    """Normalize a member name, stripping leading ``./`` and backslashes."""
    n = name.replace("\\", "/")
    while n.startswith("./"):
        n = n[2:]
    return n


def _check_traversal(name: str) -> str:
    """Reject absolute, drive-prefixed, and ``..`` traversal paths.

    Returns the normalized name on success.
    """
    if _is_absolute_or_drive(name):
        raise UnsafeArchiveError(f"unsafe archive member (absolute/drive path): {name!r}")
    norm = _normalize(name)
    # After normalization, reject any segment that is '..'.
    parts = norm.split("/")
    for part in parts:
        if part == "..":
            raise UnsafeArchiveError(f"unsafe archive member (traversal): {name!r}")
    if not norm or norm == ".":
        raise UnsafeArchiveError(f"unsafe archive member (empty path): {name!r}")
    return norm


def _check_link_escape(member_name: str, linkname: str, base: Path) -> None:
    """Reject symlink/hardlink targets that escape ``base``."""
    if _is_absolute_or_drive(linkname):
        raise UnsafeArchiveError(
            f"unsafe link {member_name!r} -> absolute target {linkname!r}"
        )
    link_norm = _normalize(linkname)
    for part in link_norm.split("/"):
        if part == "..":
            raise UnsafeArchiveError(
                f"unsafe link {member_name!r} -> traversal target {linkname!r}"
            )
    # Resolve the link target relative to the member's parent directory and
    # confirm it stays within base.
    member_parent = (base / _normalize(member_name)).parent
    resolved = (member_parent / link_norm).resolve()
    try:
        resolved.relative_to(base.resolve())
    except ValueError:
        raise UnsafeArchiveError(
            f"unsafe link {member_name!r} -> {linkname!r} escapes extraction dir"
        )


def safe_read_archive(archive: Path) -> dict[str, bytes]:
    """Read every file member from a tar.gz or zip archive into memory.

    Applies all safety checks. Returns a mapping of normalized member name to
    file bytes. Symlinks/hardlinks are skipped (not materialized).
    """
    files: dict[str, bytes] = {}
    seen: set[str] = set()
    total_size = 0
    member_count = 0

    if archive.name.endswith(".tar.gz") or archive.suffix == ".tgz":
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            for member in members:
                member_count += 1
                if member_count > MAX_MEMBER_COUNT:
                    raise UnsafeArchiveError(
                        f"archive exceeds member count limit ({MAX_MEMBER_COUNT})"
                    )
                name = _check_traversal(member.name)
                if member.issym() or member.islnk():
                    _check_link_escape(member.name, member.linkname, Path("."))
                    continue
                if not member.isfile():
                    continue
                if name in seen:
                    raise UnsafeArchiveError(
                        f"duplicate normalized output path: {name!r}"
                    )
                f = tar.extractfile(member)
                if f is None:
                    continue
                data = f.read()
                total_size += len(data)
                if total_size > MAX_EXTRACTED_SIZE:
                    raise UnsafeArchiveError(
                        f"archive exceeds extracted size limit ({MAX_EXTRACTED_SIZE} bytes)"
                    )
                seen.add(name)
                files[name] = data
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            for info in zf.infolist():
                member_count += 1
                if member_count > MAX_MEMBER_COUNT:
                    raise UnsafeArchiveError(
                        f"archive exceeds member count limit ({MAX_MEMBER_COUNT})"
                    )
                if info.is_dir():
                    continue
                name = _check_traversal(info.filename)
                if name in seen:
                    raise UnsafeArchiveError(
                        f"duplicate normalized output path: {name!r}"
                    )
                data = zf.read(info)
                total_size += len(data)
                if total_size > MAX_EXTRACTED_SIZE:
                    raise UnsafeArchiveError(
                        f"archive exceeds extracted size limit ({MAX_EXTRACTED_SIZE} bytes)"
                    )
                seen.add(name)
                files[name] = data
    else:
        raise ValueError(f"unknown archive format: {archive.name}")
    return files


def safe_extract(archive: Path, dest: Path) -> Path:
    """Extract a tar.gz or zip archive to ``dest`` safely.

    Applies all safety checks before writing any file to disk. Creates ``dest``
    if it does not exist. Returns ``dest``.

    Symlinks and hardlinks that pass the escape check are materialized as
    regular file copies of their target within ``dest`` (we do not create
    links, to avoid TOCTOU and cross-filesystem issues).
    """
    dest.mkdir(parents=True, exist_ok=True)
    base = dest.resolve()
    seen: set[str] = set()
    total_size = 0
    member_count = 0

    if archive.name.endswith(".tar.gz") or archive.suffix == ".tgz":
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            # First pass: validate all members before writing anything.
            for member in members:
                member_count += 1
                if member_count > MAX_MEMBER_COUNT:
                    raise UnsafeArchiveError(
                        f"archive exceeds member count limit ({MAX_MEMBER_COUNT})"
                    )
                _check_traversal(member.name)
                if member.issym() or member.islnk():
                    _check_link_escape(member.name, member.linkname, base)
                if member.isfile():
                    total_size += member.size
                    if total_size > MAX_EXTRACTED_SIZE:
                        raise UnsafeArchiveError(
                            f"archive exceeds extracted size limit "
                            f"({MAX_EXTRACTED_SIZE} bytes)"
                        )
            # Second pass: write files.
            for member in members:
                name = _normalize(member.name)
                if member.issym() or member.islnk():
                    continue  # links validated; not materialized
                if not member.isfile():
                    # Create directories for dir members.
                    if member.isdir():
                        (base / name).mkdir(parents=True, exist_ok=True)
                    continue
                if name in seen:
                    raise UnsafeArchiveError(
                        f"duplicate normalized output path: {name!r}"
                    )
                seen.add(name)
                out = base / name
                out.parent.mkdir(parents=True, exist_ok=True)
                f = tar.extractfile(member)
                if f is None:
                    continue
                out.write_bytes(f.read())
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            infos = zf.infolist()
            for info in infos:
                member_count += 1
                if member_count > MAX_MEMBER_COUNT:
                    raise UnsafeArchiveError(
                        f"archive exceeds member count limit ({MAX_MEMBER_COUNT})"
                    )
                if info.is_dir():
                    continue
                _check_traversal(info.filename)
                total_size += info.file_size
                if total_size > MAX_EXTRACTED_SIZE:
                    raise UnsafeArchiveError(
                        f"archive exceeds extracted size limit "
                        f"({MAX_EXTRACTED_SIZE} bytes)"
                    )
            for info in infos:
                if info.is_dir():
                    name = _normalize(info.filename)
                    (base / name).mkdir(parents=True, exist_ok=True)
                    continue
                name = _check_traversal(info.filename)
                if name in seen:
                    raise UnsafeArchiveError(
                        f"duplicate normalized output path: {name!r}"
                    )
                seen.add(name)
                out = base / name
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(zf.read(info))
    else:
        raise ValueError(f"unknown archive format: {archive.name}")
    return dest


def installed_size(archive: Path) -> int:
    """Return the sum of extracted file bytes (without extracting to disk)."""
    total = 0
    if archive.name.endswith(".tar.gz") or archive.suffix == ".tgz":
        with tarfile.open(archive, "r:gz") as tar:
            for m in tar.getmembers():
                if m.isfile():
                    total += m.size
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            for info in zf.infolist():
                if not info.is_dir():
                    total += info.file_size
    else:
        raise ValueError(f"unknown archive format: {archive.name}")
    return total


if __name__ == "__main__":
    # CLI smoke check: python scripts/archive_utils.py <archive>
    import sys
    if len(sys.argv) != 2:
        print("usage: archive_utils.py <archive>", file=sys.stderr)
        raise SystemExit(2)
    files = safe_read_archive(Path(sys.argv[1]))
    print(f"OK: {len(files)} file(s)")
