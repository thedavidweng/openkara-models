#!/usr/bin/env python3
"""Validate the model source lock (``models/source-lock.json``, issue #20).

The model lock is the immutable model source authority. Checks that:
  - Both current models are present.
  - Weights are pinned by immutable HuggingFace revision commit SHAs
    (timestamp-only state is forbidden).
  - The Demucs package revision and checkpoint signatures are declared.
  - Conversion toolchain identity and output contract are declared.
  - The lock never duplicates catalog-owned artifact URLs/sizes/digests
    (one dependency, one authority).

The ORT source lock has its own validator (``validate_source_lock.py``);
they are separate scripts so a model-lock change never triggers the
five-target ORT source build workflow.

Usage::

    python scripts/validate_model_source_lock.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODEL_LOCK_PATH = ROOT / "models" / "source-lock.json"

REQUIRED_MODELS = {"htdemucs", "htdemucs_ft"}

CONVERSION_REQUIRED_FIELDS = {
    "script", "workflow", "loader", "opset", "ir_version",
    "torch_requirement", "onnx_requirement",
}

OUTPUT_CONTRACT_REQUIRED_FIELDS = {
    "input_semantics", "output_semantics", "sample_rate", "segment_frames",
    "stems", "precision", "tensor_interface",
}

# The spectral-core export path (issue #23) is a second conversion authority
# per model: same pinned weights and demucs package, different exporter and
# tensor boundary. It must declare the contract version it implements.
SPECTRAL_CORE_REQUIRED_FIELDS = CONVERSION_REQUIRED_FIELDS | {
    "contract", "artifact_id", "output_contract",
}
SPECTRAL_CONTRACT_VERSION = "openkara.spectral-contract/v1"

# Catalog-owned artifact identity fields that must NOT be duplicated in the
# model lock (one dependency, one authority — issue #20).
FORBIDDEN_CATALOG_FIELDS = {"download_url", "url", "sha256", "byte_size", "size"}

# Mutable timestamp state is forbidden as a weights identity (issue #20
# invariant: immutable commit SHAs, never lastModified).
FORBIDDEN_TIMESTAMP_FIELDS = {"last_modified", "lastModified", "timestamp"}


def _is_hex_sha(value: Any, length: int = 40) -> bool:
    if not isinstance(value, str) or len(value) != length:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def validate_model_lock(lock: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    if lock.get("lock_version") != "openkara.model-source-lock/v1":
        errors.append(f"unexpected lock_version: {lock.get('lock_version')}")

    models = lock.get("models", {})
    missing = REQUIRED_MODELS - set(models.keys())
    if missing:
        errors.append(f"missing models: {sorted(missing)}")
    extra = set(models.keys()) - REQUIRED_MODELS
    if extra:
        errors.append(f"unknown models: {sorted(extra)}")

    for name, entry in models.items():
        prefix = f"models.{name}"

        if not entry.get("artifact_id"):
            errors.append(f"{prefix}.artifact_id missing")

        weights = entry.get("weights", {})
        if not weights.get("repo") or "/" not in weights.get("repo", ""):
            errors.append(f"{prefix}.weights.repo missing or not owner/name")
        if not _is_hex_sha(weights.get("commit_sha")):
            errors.append(f"{prefix}.weights.commit_sha must be a 40-char hex commit SHA")
        for forbidden in FORBIDDEN_TIMESTAMP_FIELDS & set(weights.keys()):
            errors.append(
                f"{prefix}.weights.{forbidden}: timestamp state is forbidden; "
                "pin an immutable commit SHA instead"
            )

        pkg = entry.get("demucs_package", {})
        if not pkg.get("version"):
            errors.append(f"{prefix}.demucs_package.version missing")
        sigs = pkg.get("checkpoint_signatures")
        if not isinstance(sigs, list) or not sigs:
            errors.append(f"{prefix}.demucs_package.checkpoint_signatures missing or empty")

        conv = entry.get("conversion", {})
        missing_conv = CONVERSION_REQUIRED_FIELDS - set(conv.keys())
        if missing_conv:
            errors.append(f"{prefix}.conversion: missing fields: {sorted(missing_conv)}")

        contract = entry.get("output_contract", {})
        missing_contract = OUTPUT_CONTRACT_REQUIRED_FIELDS - set(contract.keys())
        if missing_contract:
            errors.append(f"{prefix}.output_contract: missing fields: {sorted(missing_contract)}")

        spectral = entry.get("spectral_core", {})
        if not spectral:
            errors.append(f"{prefix}.spectral_core missing (issue #23 export authority)")
        else:
            missing_spectral = SPECTRAL_CORE_REQUIRED_FIELDS - set(spectral.keys())
            if missing_spectral:
                errors.append(
                    f"{prefix}.spectral_core: missing fields: {sorted(missing_spectral)}"
                )
            if spectral.get("contract") != SPECTRAL_CONTRACT_VERSION:
                errors.append(
                    f"{prefix}.spectral_core.contract must be "
                    f"{SPECTRAL_CONTRACT_VERSION}, got {spectral.get('contract')!r}"
                )
            sc_contract = spectral.get("output_contract", {})
            missing_sc = OUTPUT_CONTRACT_REQUIRED_FIELDS - set(sc_contract.keys())
            if missing_sc:
                errors.append(
                    f"{prefix}.spectral_core.output_contract: missing fields: "
                    f"{sorted(missing_sc)}"
                )
            if sc_contract.get("tensor_interface") != "spectral-core":
                errors.append(
                    f"{prefix}.spectral_core.output_contract.tensor_interface "
                    "must be 'spectral-core'"
                )

        def _reject_catalog_fields(obj: Any, path: str) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in FORBIDDEN_CATALOG_FIELDS:
                        errors.append(
                            f"{path}.{k}: catalog-owned artifact field must not be "
                            "duplicated in the model source lock"
                        )
                    _reject_catalog_fields(v, f"{path}.{k}")

        _reject_catalog_fields(entry, prefix)

    return errors


def main() -> int:
    if not MODEL_LOCK_PATH.is_file():
        print(f"ERROR: model source lock not found: {MODEL_LOCK_PATH}", file=sys.stderr)
        return 1

    with MODEL_LOCK_PATH.open("r", encoding="utf-8") as fh:
        lock = json.load(fh)

    errors = validate_model_lock(lock)
    if errors:
        print("ERROR: model source lock validation failed:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1

    n = len(lock.get("models", {}))
    demucs_version = lock["models"]["htdemucs"]["demucs_package"]["version"]
    print(f"OK: model source lock valid ({n} models, demucs {demucs_version})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
