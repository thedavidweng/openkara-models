#!/usr/bin/env python3
"""Native-host verification for runtime-only measurement.

A report must be produced on a host whose operating system and CPU
architecture actually match the catalog target triple. This module normalizes
``platform`` values, maps each supported target to its expected host, detects
macOS Rosetta translation, and validates the declared environment. It is not
an attestation: a report can still lie about its host; the validator only
rejects internally inconsistent or translated-host claims, and repository
review remains required.
"""

from __future__ import annotations

import ctypes
import os
import platform
import sys
from typing import Any

# target triple -> (os_name, arch)
TARGET_HOSTS: dict[str, tuple[str, str]] = {
    "aarch64-apple-darwin": ("darwin", "aarch64"),
    "x86_64-apple-darwin": ("darwin", "x86_64"),
    "aarch64-unknown-linux-gnu": ("linux", "aarch64"),
    "x86_64-unknown-linux-gnu": ("linux", "x86_64"),
    "x86_64-pc-windows-msvc": ("windows", "x86_64"),
}

_ENVIRONMENT_KEYS = (
    "os", "os_name", "arch", "machine", "python", "cpu_count",
    "target_triple", "rosetta_translated",
)


class HostMismatch(ValueError):
    """The current (or declared) host cannot produce native evidence for a target."""


def normalize_arch(machine: str) -> str:
    value = machine.strip().lower()
    if value in {"arm64", "aarch64"}:
        return "aarch64"
    if value in {"x86_64", "amd64", "x64"}:
        return "x86_64"
    return value


def expected_host(target: str) -> tuple[str, str]:
    try:
        return TARGET_HOSTS[target]
    except KeyError as exc:
        raise HostMismatch(f"unsupported native measurement target: {target}") from exc


def rosetta_translated() -> bool:
    """Return True when this macOS process is running under Rosetta.

    ``sysctl.proc_translated`` exists on Apple Silicon; its absence on Intel or
    older macOS means no translation. Any other sysctl failure raises instead
    of guessing.
    """
    if sys.platform != "darwin":
        return False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        function = libc.sysctlbyname
    except (OSError, AttributeError) as exc:
        raise RuntimeError(f"cannot load sysctlbyname for Rosetta detection: {exc}") from exc
    function.argtypes = [
        ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p, ctypes.c_size_t,
    ]
    function.restype = ctypes.c_int
    value = ctypes.c_int32(0)
    size = ctypes.c_size_t(ctypes.sizeof(value))
    result = function(b"sysctl.proc_translated", ctypes.byref(value),
                      ctypes.byref(size), None, 0)
    if result == 0:
        return value.value == 1
    errno = ctypes.get_errno()
    if errno == 2:  # ENOENT: key absent (Intel/older macOS), not translated
        return False
    raise RuntimeError(f"cannot determine Rosetta translation state (errno {errno})")


def build_environment(target: str, system: str, machine: str,
                      translated: bool, *,
                      platform_label: str | None = None,
                      python_version: str | None = None,
                      cpu_count: int | None = None) -> dict[str, Any]:
    """Validate host facts against the target and return the report environment."""
    expected_os, expected_arch = expected_host(target)
    os_name = system.strip().lower()
    arch = normalize_arch(machine)
    if os_name != expected_os:
        raise HostMismatch(
            f"target {target} requires {expected_os}, host reports {os_name}"
        )
    if arch != expected_arch:
        raise HostMismatch(
            f"target {target} requires {expected_arch}, host reports {arch} ({machine})"
        )
    if translated:
        raise HostMismatch(
            "macOS process is Rosetta-translated; translated execution cannot "
            "produce native target evidence"
        )
    environment = {
        "os": platform_label if platform_label is not None else platform.platform(),
        "os_name": os_name,
        "arch": arch,
        "machine": machine,
        "python": python_version if python_version is not None else platform.python_version(),
        "cpu_count": cpu_count if cpu_count is not None else (os.cpu_count() or 1),
        "target_triple": target,
        "rosetta_translated": False,
    }
    if set(environment) != set(_ENVIRONMENT_KEYS):
        raise AssertionError("environment keys drifted")
    return environment


def describe_host(target: str) -> dict[str, Any]:
    """Describe and verify the actual host for ``target`` (fail closed)."""
    system = platform.system()
    machine = platform.machine()
    translated = rosetta_translated() if system.strip().lower() == "darwin" else False
    return build_environment(target, system, machine, translated)


def environment_errors(environment: dict[str, Any], target: str) -> list[str]:
    """Validate a report's declared environment against the report target."""
    errors: list[str] = []
    try:
        expected_os, expected_arch = expected_host(target)
    except HostMismatch as exc:
        return [str(exc)]
    if environment.get("target_triple") != target:
        errors.append("report environment target_triple differs from the report target")
    if environment.get("os_name") != expected_os:
        errors.append(
            f"report environment OS {environment.get('os_name')!r} != {expected_os!r}"
        )
    if environment.get("arch") != expected_arch:
        errors.append(
            f"report environment arch {environment.get('arch')!r} != {expected_arch!r}"
        )
    if environment.get("rosetta_translated") is not False:
        errors.append("report environment is Rosetta-translated or does not declare it false")
    machine = environment.get("machine")
    if not isinstance(machine, str) or not machine.strip():
        errors.append("report environment machine is empty")
    elif normalize_arch(machine) != expected_arch:
        errors.append(
            f"report environment machine {machine!r} does not match arch {expected_arch!r}"
        )
    return errors
