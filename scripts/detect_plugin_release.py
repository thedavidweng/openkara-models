#!/usr/bin/env python3
"""Detect new stable onnxruntime-ep-webgpu releases on PyPI and compare against ort/plugin-lock.json.

Polls https://pypi.org/pypi/onnxruntime-ep-webgpu/json, filters prereleases and
yanked or incomplete releases (requiring non-yanked wheels for every target in
``ort/plugin-lock.json``), picks the highest ``X.Y.Z`` semantic version, and
compares it against ``ort/plugin-lock.json``.

Exit codes:
  0: update available
  1: lock is up to date
  2: API or validation failure
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import plugin_package  # noqa: E402

LOCK_PATH = plugin_package.LOCK_PATH


def _stable_version(version: str) -> tuple[int, int, int] | None:
    try:
        return plugin_package.parse_semver(version)
    except ValueError:
        return None


def _target_wheels_for_release(
    version: str, files: list[dict[str, Any]], targets: dict[str, Any]
) -> dict[str, dict[str, Any]] | None:
    by_name = {
        f.get("filename"): f
        for f in files
        if isinstance(f, dict)
        and not f.get("yanked", False)
        and f.get("packagetype") in (None, "bdist_wheel")
    }
    matched: dict[str, dict[str, Any]] = {}
    for target, cfg in targets.items():
        wheel_tag = cfg["wheel_tag"]
        expected = f"{plugin_package.DIST_PREFIX}-{version}-{wheel_tag}.whl"
        file_info = by_name.get(expected)
        if file_info is None:
            return None
        sha256 = (file_info.get("digests") or {}).get("sha256", "")
        size = file_info.get("size")
        url = file_info.get("url", "")
        if not sha256 or type(size) is not int or size <= 0 or not url:
            return None
        matched[target] = {
            "filename": expected,
            "url": url,
            "size": size,
            "sha256": sha256,
        }
    return matched


def select_latest_stable(
    pypi_data: dict[str, Any], targets: dict[str, Any]
) -> dict[str, Any]:
    """Pick the highest stable X.Y.Z PyPI release covering every locked target."""
    releases = pypi_data.get("releases")
    if not isinstance(releases, dict):
        raise RuntimeError("unexpected PyPI response: missing releases mapping")

    best_version: tuple[int, int, int] | None = None
    best_payload: dict[str, Any] | None = None

    for version, files in releases.items():
        parsed = _stable_version(str(version))
        if parsed is None or not isinstance(files, list) or not files:
            continue
        wheels = _target_wheels_for_release(str(version), files, targets)
        if wheels is None:
            continue
        if best_version is None or parsed > best_version:
            best_version = parsed
            best_payload = {
                "version": str(version),
                "pypi_url": plugin_package.pypi_version_url(str(version)),
                "wheels": wheels,
            }

    if best_payload is None:
        raise RuntimeError("no stable PyPI releases found covering all locked targets")
    return best_payload


def detect_latest_stable(lock: dict[str, Any] | None = None) -> dict[str, Any]:
    if lock is None:
        lock = plugin_package.load_lock()
    pypi_data = plugin_package.fetch_pypi_json()
    return select_latest_stable(pypi_data, lock["targets"])


def compare_with_lock(latest: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    lock_version = lock.get("package", {}).get("version", "")
    latest_version = latest["version"]
    lock_tuple = _stable_version(lock_version)
    latest_tuple = _stable_version(latest_version)

    if latest_tuple is not None and lock_tuple is not None:
        if latest_tuple != lock_tuple:
            update_available = latest_tuple > lock_tuple
        else:
            update_available = any(
                latest["wheels"][t]["sha256"] != lock["targets"][t]["sha256"]
                or latest["wheels"][t]["size"] != lock["targets"][t]["size"]
                or latest["wheels"][t]["url"] != lock["targets"][t]["url"]
                for t in lock["targets"]
            )
    else:
        update_available = latest_version != lock_version

    return {
        "plugin": plugin_package.PROVIDER,
        "package_name": plugin_package.PACKAGE_NAME,
        "lock_version": lock_version,
        "latest_version": latest_version,
        "update_available": update_available,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect new stable onnxruntime-ep-webgpu releases on PyPI.")
    parser.add_argument("--lock", type=Path, default=LOCK_PATH, help="Path to ort/plugin-lock.json.")
    parser.add_argument("--json", action="store_true", help="Output JSON.")
    args = parser.parse_args()

    try:
        lock = plugin_package.load_lock(args.lock)
        latest = detect_latest_stable(lock)
    except (RuntimeError, ValueError, OSError, KeyError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    comparison = compare_with_lock(latest, lock)

    if args.json:
        print(json.dumps({"latest": latest, "comparison": comparison}, indent=2))
    else:
        print(f"Latest stable: {latest['version']}")
        print(f"Lock:          {comparison['lock_version']}")
        if comparison["update_available"]:
            print(f"UPDATE AVAILABLE: {comparison['lock_version']} -> {comparison['latest_version']}")
        else:
            print("Plugin lock is up to date.")

    return 0 if comparison["update_available"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
