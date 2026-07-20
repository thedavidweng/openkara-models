#!/usr/bin/env python3
"""Generate catalog validation fixtures from the hand-authored valid base.

The valid fixtures under ``catalog/fixtures/valid/`` are the canonical,
hand-authored examples. This script derives every invalid fixture from them
via a single explicit mutation each, and writes a sibling
``<name>.expected.json`` sidecar declaring the expected failure stage and
error codes. ``scripts/validate_catalog.py --all-fixtures`` checks both.

Regenerate after editing the valid base or the mutation table::

    python scripts/generate_catalog_fixtures.py
    git diff --exit-code catalog/fixtures   # CI guard: fixtures must be fresh

The generated JSON files carry a header comment in this docstring's spirit;
do not hand-edit them — edit the mutation table here instead.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "catalog" / "fixtures"
VALID = FIXTURES / "valid"
INVALID = FIXTURES / "invalid"

Mutation = tuple[str, str, list[str], Callable[[dict[str, Any]], None]]
# (name, expected_stage, expected_error_codes, mutate_fn)
# expected_stage ∈ {"schema", "invariant", "either"}
# expected_error_codes: invariant codes or "schema:<substring>" matched loosely.


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _dump(path: Path, doc: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def _dump_expected(path: Path, stage: str, codes: list[str]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(
            {"expected_stage": stage, "expected_error_codes": codes},
            fh,
            indent=2,
            ensure_ascii=False,
        )
        fh.write("\n")


# --------------------------------------------------------------------------- #
# Release-manifest mutations
# --------------------------------------------------------------------------- #

RELEASE_MUTATIONS: list[Mutation] = [
    (
        "missing-file-digest",
        "schema",
        [],
        lambda d: d["artifacts"]["models"][0]["extracted_file_digests"][
            "htdemucs.onnx"
        ].pop("sha256"),
    ),
    (
        "duplicate-artifact-id",
        "invariant",
        ["duplicate_artifact_id"],
        lambda d: d["artifacts"]["bundles"][0].__setitem__(
            "artifact_id", d["artifacts"]["models"][0]["artifact_id"]
        ),
    ),
    (
        "dangling-runtime-ref",
        "invariant",
        ["dangling_runtime_ref"],
        lambda d: d["artifacts"]["models"][0]["model"][
            "compatible_runtime_ids"
        ].__setitem__(0, "ort.does-not-exist"),
    ),
    (
        "dangling-model-ref",
        "invariant",
        ["dangling_model_ref"],
        lambda d: d["artifacts"]["runtimes"][0]["runtime"][
            "supported_model_artifact_ids"
        ].__setitem__(0, "model.does-not-exist"),
    ),
    (
        "bundle-target-mismatch",
        "invariant",
        ["target_mismatch"],
        lambda d: d["artifacts"]["bundles"][0].__setitem__(
            "target_triple", "x86_64-unknown-linux-gnu"
        ),
    ),
    (
        "bundle-provider-mismatch",
        "invariant",
        ["provider_mismatch"],
        lambda d: d["artifacts"]["bundles"][0].__setitem__(
            "execution_provider", "directml"
        ),
    ),
    (
        "mutable-download-url",
        "invariant",
        ["mutable_download_url"],
        lambda d: d["artifacts"]["models"][0].__setitem__(
            "download_url", "https://example.com/latest/htdemucs.onnx"
        ),
    ),
    (
        "dangling-replacement",
        "invariant",
        ["dangling_replacement"],
        lambda d: (
            d["artifacts"]["models"][0]["deprecation"].__setitem__("deprecated", True),
            d["artifacts"]["models"][0]["deprecation"].__setitem__(
                "replacement_artifact_id", "gone.artifact"
            ),
        ),
    ),
    (
        "target-arch-mismatch",
        "invariant",
        ["target_arch_mismatch"],
        lambda d: d["artifacts"]["runtimes"][0].__setitem__("arch", "x86_64"),
    ),
    (
        "unknown-top-level-field",
        "schema",
        [],
        lambda d: d.__setitem__("release_digest", "deadbeef"),
    ),
    (
        "missing-required-field",
        "schema",
        [],
        lambda d: d.pop("producer"),
    ),
    (
        "unknown-schema-version",
        "schema",
        [],
        lambda d: d.__setitem__("schema_version", "openkara.catalog/v999"),
    ),
    (
        "generation-zero",
        "schema",
        [],
        lambda d: d.__setitem__("generation", 0),
    ),
    (
        "bad-sha256",
        "schema",
        [],
        lambda d: d["artifacts"]["models"][0].__setitem__(
            "archive_digest", "not-a-sha256"
        ),
    ),
    (
        "compatibility-dangling-runtime",
        "invariant",
        ["dangling_runtime_ref"],
        lambda d: d["compatibility"][0].__setitem__(
            "runtime_artifact_id", "ort.does-not-exist"
        ),
    ),
    (
        "compatibility-provider-mismatch",
        "invariant",
        ["provider_mismatch"],
        lambda d: d["compatibility"][0].__setitem__(
            "execution_provider", "directml"
        ),
    ),
]


# --------------------------------------------------------------------------- #
# Channel-pointer mutations
# --------------------------------------------------------------------------- #

CHANNEL_MUTATIONS: list[Mutation] = [
    (
        "channel-mutable-url",
        "invariant",
        ["mutable_manifest_url"],
        lambda d: d.__setitem__(
            "release_manifest_url",
            "https://example.com/latest/release-manifest.json",
        ),
    ),
    (
        "channel-bad-digest",
        "schema",
        [],
        lambda d: d.__setitem__("release_manifest_sha256", "not-hex"),
    ),
    (
        "channel-unknown-field",
        "schema",
        [],
        lambda d: d.__setitem__("release_manifest_digest", "deadbeef"),
    ),
]


def main() -> int:
    INVALID.mkdir(parents=True, exist_ok=True)
    base_release = _load(VALID / "release-minimal.json")
    base_channel = _load(VALID / "channel-stable.json")

    for name, stage, codes, mutate in RELEASE_MUTATIONS:
        doc = copy.deepcopy(base_release)
        mutate(doc)
        _dump(INVALID / f"release-{name}.json", doc)
        _dump_expected(INVALID / f"release-{name}.expected.json", stage, codes)

    for name, stage, codes, mutate in CHANNEL_MUTATIONS:
        doc = copy.deepcopy(base_channel)
        mutate(doc)
        _dump(INVALID / f"channel-{name}.json", doc)
        _dump_expected(INVALID / f"channel-{name}.expected.json", stage, codes)

    print(
        f"generated {len(RELEASE_MUTATIONS)} release + "
        f"{len(CHANNEL_MUTATIONS)} channel invalid fixtures in {INVALID}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
