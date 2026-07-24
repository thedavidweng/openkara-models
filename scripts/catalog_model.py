#!/usr/bin/env python3
"""Typed models and invariant checks for the OpenKara artifact catalog.

This module is the single Python authority for catalog entity shapes. The JSON
Schema files in ``catalog/`` enforce field-level structure; this module enforces
the cross-field invariants that JSON Schema cannot express (artifact identity,
compatibility-edge referential integrity, target/provider consistency, immutable
URLs, and cross-release generation monotonicity).

See ``docs/catalog-contract.md`` for the consumer contract.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

SCHEMA_VERSION_RELEASE = "openkara.catalog/release-v1"
SCHEMA_VERSION_CHANNEL = "openkara.catalog/channel-v1"

CATALOG_DIR = Path(__file__).resolve().parents[1] / "catalog"
RELEASE_SCHEMA_PATH = CATALOG_DIR / "schema-release-v1.json"
CHANNEL_SCHEMA_PATH = CATALOG_DIR / "schema-channel-v1.json"
FIXTURES_DIR = CATALOG_DIR / "fixtures"

#: Every runtime target OpenKara ships. A model may be portable (``None``) but
#: a runtime always targets exactly one of these.
SUPPORTED_TARGET_TRIPLES: frozenset[str] = frozenset(
    {
        "aarch64-apple-darwin",
        "x86_64-apple-darwin",
        "x86_64-unknown-linux-gnu",
        "aarch64-unknown-linux-gnu",
        "x86_64-pc-windows-msvc",
    }
)

TARGET_ARCH_OS: dict[str, tuple[str, str]] = {
    "aarch64-apple-darwin": ("aarch64", "macos"),
    "x86_64-apple-darwin": ("x86_64", "macos"),
    "x86_64-unknown-linux-gnu": ("x86_64", "linux"),
    "aarch64-unknown-linux-gnu": ("aarch64", "linux"),
    "x86_64-pc-windows-msvc": ("x86_64", "windows"),
}

RELEASE_ID_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{3}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ARTIFACT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

#: URL must contain an immutable segment: a GitHub release download tag, or a
#: content-addressed (40/64 hex) path. This rejects mutable "latest" URLs.
_IMMUTABLE_URL_RE = re.compile(
    r"^https://"
    r"(?:.*/releases/download/[A-Za-z0-9_.-]+/"  # GitHub release tag
    r"|.*\b[0-9a-f]{40}\b"  # git SHA content addressing
    r"|.*\b[0-9a-f]{64}\b"  # sha256 content addressing
    r")"
)


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class ArtifactKind(str, Enum):
    MODEL = "model"
    RUNTIME = "runtime"
    BUNDLE = "bundle"


class Profile(str, Enum):
    BALANCED = "balanced"
    QUALITY = "quality"
    KARAOKE_2STEM = "karaoke_2stem"


class TensorInterface(str, Enum):
    WAVEFORM = "waveform"
    SPECTRAL_CORE = "spectral-core"


class Precision(str, Enum):
    FP32 = "fp32"
    MIXED_FP16 = "mixed-fp16"
    INT8_SELECTIVE = "int8-selective"


class GraphFormat(str, Enum):
    ONNX = "onnx"
    ORT = "ort"


class ExecutionProvider(str, Enum):
    CPU = "cpu"
    XNNPACK = "xnnpack"
    COREML = "coreml"
    DIRECTML = "directml"


class StemProfile(str, Enum):
    FOUR_STEM = "four-stem"
    KARAOKE_2STEM = "karaoke_2stem"


class ChannelName(str, Enum):
    STABLE = "stable"
    CANDIDATE = "candidate"


class CompatibilityStatus(str, Enum):
    SUPPORTED = "supported"
    DEPRECATED = "deprecated"


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


@dataclass
class ValidationError:
    code: str
    message: str
    path: str = ""

    def __str__(self) -> str:
        suffix = f" at {self.path}" if self.path else ""
        return f"[{self.code}] {self.message}{suffix}"


class CatalogIntegrityError(Exception):
    """Raised when a catalog document fails invariant validation."""

    def __init__(self, errors: list[ValidationError]):
        self.errors = errors
        super().__init__("; ".join(str(e) for e in errors) if errors else "catalog integrity error")


# --------------------------------------------------------------------------- #
# Typed accessors
# --------------------------------------------------------------------------- #


@dataclass
class ArtifactRef:
    """Index of artifacts in a release manifest by id and kind."""

    by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    runtimes: dict[str, dict[str, Any]] = field(default_factory=dict)
    bundles: dict[str, dict[str, Any]] = field(default_factory=dict)


def index_artifacts(manifest: dict[str, Any]) -> ArtifactRef:
    ref = ArtifactRef()
    for kind_list, bucket in (
        ("models", ref.models),
        ("runtimes", ref.runtimes),
        ("bundles", ref.bundles),
    ):
        for art in manifest.get("artifacts", {}).get(kind_list, []):
            aid = art.get("artifact_id")
            if aid is not None:
                ref.by_id[aid] = art
                bucket[aid] = art
    return ref


# --------------------------------------------------------------------------- #
# Invariant checks
# --------------------------------------------------------------------------- #


def _is_immutable_url(url: str) -> bool:
    return bool(_IMMUTABLE_URL_RE.match(url))


def validate_release_invariants(manifest: dict[str, Any]) -> list[ValidationError]:
    """Return a list of cross-field invariant violations for a release manifest.

    An empty list means the manifest satisfies every invariant this module
    enforces. JSON Schema structure must already have been checked separately.
    """
    errors: list[ValidationError] = []
    ref = index_artifacts(manifest)

    # 1. Artifact ID uniqueness across all kinds.
    seen: dict[str, str] = {}  # id -> kind where first seen
    for kind_list in ("models", "runtimes", "bundles"):
        for art in manifest.get("artifacts", {}).get(kind_list, []):
            aid = art.get("artifact_id")
            if aid is None:
                continue
            if aid in seen:
                errors.append(
                    ValidationError(
                        "duplicate_artifact_id",
                        f"artifact_id {aid!r} appears in both {seen[aid]!r} and {kind_list!r}",
                        f"artifacts.{kind_list}[]",
                    )
                )
            else:
                seen[aid] = kind_list

    # 2. Immutable download URLs and per-file digests for every artifact.
    for kind_list in ("models", "runtimes", "bundles"):
        for art in manifest.get("artifacts", {}).get(kind_list, []):
            aid = art.get("artifact_id", "<unknown>")
            url = art.get("download_url", "")
            if url and not _is_immutable_url(url):
                errors.append(
                    ValidationError(
                        "mutable_download_url",
                        f"artifact {aid!r} download_url is not immutable "
                        "(must contain a release tag or content-addressed segment)",
                        f"artifacts.{kind_list}[{aid}].download_url",
                    )
                )
            digests = art.get("extracted_file_digests", {}) or {}
            for fname, entry in digests.items():
                if not isinstance(entry, dict):
                    continue
                if entry.get("sha256") is None or entry.get("size") is None:
                    errors.append(
                        ValidationError(
                            "missing_file_digest",
                            f"artifact {aid!r} extracted file {fname!r} missing size or sha256",
                            f"artifacts.{kind_list}[{aid}].extracted_file_digests.{fname}",
                        )
                    )

    # 3. Model -> compatible_runtime_ids must reference existing runtimes.
    for aid, model in ref.models.items():
        for rid in model.get("model", {}).get("compatible_runtime_ids", []):
            if rid not in ref.runtimes:
                errors.append(
                    ValidationError(
                        "dangling_runtime_ref",
                        f"model {aid!r} compatible_runtime_ids references unknown runtime {rid!r}",
                        f"artifacts.models[{aid}].model.compatible_runtime_ids",
                    )
                )

    # 4. Runtime -> supported_model_artifact_ids must reference existing models.
    for aid, runtime in ref.runtimes.items():
        for mid in runtime.get("runtime", {}).get("supported_model_artifact_ids", []):
            if mid not in ref.models:
                errors.append(
                    ValidationError(
                        "dangling_model_ref",
                        f"runtime {aid!r} supported_model_artifact_ids references unknown model {mid!r}",
                        f"artifacts.runtimes[{aid}].runtime.supported_model_artifact_ids",
                    )
                )

    # 5. Bundle referential integrity and target/provider consistency.
    for aid, bundle in ref.bundles.items():
        mid = bundle.get("model_artifact_id")
        rid = bundle.get("runtime_artifact_id")
        if mid not in ref.models:
            errors.append(
                ValidationError(
                    "dangling_model_ref",
                    f"bundle {aid!r} model_artifact_id {mid!r} is not a known model",
                    f"artifacts.bundles[{aid}].model_artifact_id",
                )
            )
        if rid not in ref.runtimes:
            errors.append(
                ValidationError(
                    "dangling_runtime_ref",
                    f"bundle {aid!r} runtime_artifact_id {rid!r} is not a known runtime",
                    f"artifacts.bundles[{aid}].runtime_artifact_id",
                )
            )
        runtime = ref.runtimes.get(rid)
        if runtime is not None:
            rt_target = runtime.get("target_triple")
            b_target = bundle.get("target_triple")
            if b_target != rt_target:
                errors.append(
                    ValidationError(
                        "target_mismatch",
                        f"bundle {aid!r} target {b_target!r} != runtime {rid!r} target {rt_target!r}",
                        f"artifacts.bundles[{aid}].target_triple",
                    )
                )
            providers = set(runtime.get("runtime", {}).get("execution_providers", []))
            ep = bundle.get("execution_provider")
            if ep not in providers:
                errors.append(
                    ValidationError(
                        "provider_mismatch",
                        f"bundle {aid!r} execution_provider {ep!r} not in runtime {rid!r} providers {sorted(providers)!r}",
                        f"artifacts.bundles[{aid}].execution_provider",
                    )
                )

    # 6. Compatibility edges: referential integrity + target/provider consistency.
    for i, edge in enumerate(manifest.get("compatibility", [])):
        mid = edge.get("model_artifact_id")
        rid = edge.get("runtime_artifact_id")
        if mid not in ref.models:
            errors.append(
                ValidationError(
                    "dangling_model_ref",
                    f"compatibility[{i}] model_artifact_id {mid!r} is not a known model",
                    f"compatibility[{i}].model_artifact_id",
                )
            )
        if rid not in ref.runtimes:
            errors.append(
                ValidationError(
                    "dangling_runtime_ref",
                    f"compatibility[{i}] runtime_artifact_id {rid!r} is not a known runtime",
                    f"compatibility[{i}].runtime_artifact_id",
                )
            )
        runtime = ref.runtimes.get(rid)
        model = ref.models.get(mid)
        if runtime is not None and model is not None:
            e_target = edge.get("target_triple")
            rt_target = runtime.get("target_triple")
            if e_target != rt_target:
                errors.append(
                    ValidationError(
                        "target_mismatch",
                        f"compatibility[{i}] target {e_target!r} != runtime {rid!r} target {rt_target!r}",
                        f"compatibility[{i}].target_triple",
                    )
                )
            m_target = model.get("target_triple")
            if m_target is not None and e_target is not None and m_target != e_target:
                errors.append(
                    ValidationError(
                        "target_mismatch",
                        f"compatibility[{i}] target {e_target!r} != model {mid!r} target {m_target!r}",
                        f"compatibility[{i}].target_triple",
                    )
                )
            providers = set(runtime.get("runtime", {}).get("execution_providers", []))
            ep = edge.get("execution_provider")
            if ep not in providers:
                errors.append(
                    ValidationError(
                        "provider_mismatch",
                        f"compatibility[{i}] execution_provider {ep!r} not in runtime {rid!r} providers {sorted(providers)!r}",
                        f"compatibility[{i}].execution_provider",
                    )
                )

    # 7. Model/runtime declarations and compatibility edges must be reciprocal.
    if manifest.get("generation", 0) >= 3:
        edge_pairs = {
            (edge.get("model_artifact_id"), edge.get("runtime_artifact_id"))
            for edge in manifest.get("compatibility", [])
            if edge.get("status") == "supported"
        }
        for mid, model in ref.models.items():
            declared = set(model.get("model", {}).get("compatible_runtime_ids", []))
            supporting = {
                rid for rid, runtime in ref.runtimes.items()
                if mid in set(runtime.get("runtime", {}).get("supported_model_artifact_ids", []))
            }
            if supporting and not declared:
                errors.append(ValidationError(
                    "empty_runtime_compatibility",
                    f"model {mid!r} has supporting runtimes but compatible_runtime_ids is empty",
                    f"artifacts.models[{mid}].model.compatible_runtime_ids",
                ))
            for rid in declared & set(ref.runtimes):
                supported_models = set(
                    ref.runtimes[rid].get("runtime", {}).get("supported_model_artifact_ids", [])
                )
                if mid not in supported_models:
                    errors.append(ValidationError(
                        "asymmetric_runtime_support",
                        f"model {mid!r} declares runtime {rid!r}, but the runtime does not declare the model",
                        f"artifacts.models[{mid}].model.compatible_runtime_ids",
                    ))
                if (mid, rid) not in edge_pairs:
                    errors.append(ValidationError(
                        "missing_compatibility_edge",
                        f"model {mid!r} and runtime {rid!r} are reciprocal but have no supported compatibility edge",
                        "compatibility",
                    ))
            for rid in supporting:
                if rid not in declared:
                    errors.append(ValidationError(
                        "asymmetric_model_support",
                        f"runtime {rid!r} declares model {mid!r}, but the model does not declare the runtime",
                        f"artifacts.runtimes[{rid}].runtime.supported_model_artifact_ids",
                    ))
        for mid, rid in edge_pairs:
            model = ref.models.get(mid)
            runtime = ref.runtimes.get(rid)
            if model is None or runtime is None:
                continue
            model_runtimes = set(model.get("model", {}).get("compatible_runtime_ids", []))
            runtime_models = set(runtime.get("runtime", {}).get("supported_model_artifact_ids", []))
            if rid not in model_runtimes or mid not in runtime_models:
                errors.append(ValidationError(
                    "asymmetric_compatibility_edge",
                    f"supported edge {mid!r} -> {rid!r} is not represented by both artifact declarations",
                    "compatibility",
                ))
    

    # 8. Deprecation replacement must point at an existing artifact in this release.
    for kind_list in ("models", "runtimes", "bundles"):
        for art in manifest.get("artifacts", {}).get(kind_list, []):
            dep = art.get("deprecation") or {}
            if dep.get("deprecated"):
                repl = dep.get("replacement_artifact_id")
                if repl is not None and repl not in ref.by_id:
                    errors.append(
                        ValidationError(
                            "dangling_replacement",
                            f"artifact {art.get('artifact_id')!r} deprecation.replacement_artifact_id "
                            f"{repl!r} is not a known artifact in this release",
                            f"artifacts.{kind_list}[{art.get('artifact_id')}].deprecation.replacement_artifact_id",
                        )
                    )

    # 9. arch/os must be consistent with target_triple when both are present.
    for kind_list in ("models", "runtimes", "bundles"):
        for art in manifest.get("artifacts", {}).get(kind_list, []):
            t = art.get("target_triple")
            if t is None:
                continue
            expected_arch, expected_os = TARGET_ARCH_OS.get(t, (None, None))
            if expected_arch is not None and art.get("arch") is not None and art.get("arch") != expected_arch:
                errors.append(
                    ValidationError(
                        "target_arch_mismatch",
                        f"artifact {art.get('artifact_id')!r} arch {art.get('arch')!r} != target {t!r} arch {expected_arch!r}",
                        f"artifacts.{kind_list}[{art.get('artifact_id')}].arch",
                    )
                )
            if expected_os is not None and art.get("os") is not None and art.get("os") != expected_os:
                errors.append(
                    ValidationError(
                        "target_os_mismatch",
                        f"artifact {art.get('artifact_id')!r} os {art.get('os')!r} != target {t!r} os {expected_os!r}",
                        f"artifacts.{kind_list}[{art.get('artifact_id')}].os",
                    )
                )

    return errors


def validate_channel_invariants(pointer: dict[str, Any]) -> list[ValidationError]:
    """Return invariant violations for a stable/candidate channel pointer."""
    errors: list[ValidationError] = []
    url = pointer.get("release_manifest_url", "")
    if url and not _is_immutable_url(url):
        errors.append(
            ValidationError(
                "mutable_manifest_url",
                "release_manifest_url is not immutable "
                "(must contain a release tag or content-addressed segment)",
                "release_manifest_url",
            )
        )
    sha = pointer.get("release_manifest_sha256", "")
    if sha and not SHA256_RE.match(sha):
        errors.append(
            ValidationError(
                "bad_manifest_digest",
                "release_manifest_sha256 is not a 64-char lowercase hex digest",
                "release_manifest_sha256",
            )
        )
    return errors


def assert_generations_monotonic(manifests: Iterable[dict[str, Any]]) -> None:
    """Assert a sequence of release manifests has non-decreasing generation and
    strictly increasing release_id when sorted by release_id.

    Used by the generator (issue #18 PR 2) and by multi-manifest tests.
    """
    items = sorted(manifests, key=lambda m: m.get("release_id", ""))
    prev_gen: int | None = None
    prev_id: str | None = None
    for m in items:
        rid = m.get("release_id", "")
        gen = m.get("generation")
        if prev_id is not None and rid == prev_id:
            raise CatalogIntegrityError(
                [ValidationError("duplicate_release_id", f"release_id {rid!r} appears twice")]
            )
        if prev_gen is not None and gen is not None and gen < prev_gen:
            raise CatalogIntegrityError(
                [
                    ValidationError(
                        "non_monotonic_generation",
                        f"generation {gen} < previous {prev_gen} at release_id {rid!r}",
                    )
                ]
            )
        prev_gen = gen
        prev_id = rid


__all__ = [
    "SCHEMA_VERSION_RELEASE",
    "SCHEMA_VERSION_CHANNEL",
    "CATALOG_DIR",
    "RELEASE_SCHEMA_PATH",
    "CHANNEL_SCHEMA_PATH",
    "FIXTURES_DIR",
    "SUPPORTED_TARGET_TRIPLES",
    "TARGET_ARCH_OS",
    "ArtifactKind",
    "Profile",
    "TensorInterface",
    "Precision",
    "GraphFormat",
    "ExecutionProvider",
    "StemProfile",
    "ChannelName",
    "CompatibilityStatus",
    "ValidationError",
    "CatalogIntegrityError",
    "ArtifactRef",
    "index_artifacts",
    "validate_release_invariants",
    "validate_channel_invariants",
    "assert_generations_monotonic",
]


if __name__ == "__main__":
    sys.exit(0)
