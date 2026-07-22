#!/usr/bin/env python3
"""Generate the ORT reduced-build operator config from stable catalog models.

Scans every ONNX model (and ORT-format bundle, when present) referenced by a
candidate catalog release manifest, extracts the full operator set
(domain + op_type) from each graph, and emits:

  * ``ort/required-operators.config`` — the ORT ``create_reduced_build_config``
    format (``<domain>;<op1>,op2,...`` per line) consumed by ORT's
    ``reduce_op_kernels.py`` / ``build.py --include_ops_by_config``.
  * ``ort/required-operators.json`` — a provenance sidecar recording which
    models contributed, the union operator set, the source lock ref, and the
    generator version. This is the verifiable record; the ``.config`` file is
    the build input.

The config is the union of operators across every stable model and ORT-format
bundle in the candidate catalog. Because the CPU execution provider is always
included in every target build, the union operator set also covers every
operator the CPU fallback path may need to run (any node the selected EP does
not claim falls back to CPU, and CPU kernels are built for exactly the model's
operator set). The comparison script
(``scripts/compare_runtime_builds.py``) verifies the reduced runtime can
actually load each model and that the fallback node count is unchanged.

Usage::

    # From a catalog release manifest (resolves model download URLs):
    python scripts/generate_required_operators.py \\
        --catalog catalog/releases/<release-id>.json \\
        --download-dir ort/model-cache

    # From explicit model files (e.g. locally converted):
    python scripts/generate_required_operators.py \\
        --models models/htdemucs.onnx models/htdemucs_ft.onnx

    # Write to a specific path:
    python scripts/generate_required_operators.py --models ... --output ort/required-operators
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import onnx

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "ort" / "source-lock.json"
DEFAULT_OUTPUT_STEM = ROOT / "ort" / "required-operators"
GENERATOR_VERSION = "openkara.required-operators/v1"


def extract_operators(model_path: Path) -> dict[str, set[str]]:
    """Return ``{domain: {op_type, ...}}`` for every node in the model graph.

    Recurses into subgraphs (if/loop bodies) so that control-flow operators are
    included. The empty domain (``""``) maps to the default ``ai.onnx`` domain.
    """
    model = onnx.load(str(model_path), load_external_data=False)
    domains: dict[str, set[str]] = {}

    def _walk(graph: onnx.GraphProto) -> None:
        for node in graph.node:
            domain = node.domain or ""
            domains.setdefault(domain, set()).add(node.op_type)
        for node in graph.node:
            for attr in node.attribute:
                if attr.g and attr.g.node:
                    _walk(attr.g)
                for sub in attr.graphs:
                    if sub and sub.node:
                        _walk(sub)

    _walk(model.graph)
    return domains


def merge_operator_sets(per_model: list[tuple[str, dict[str, set[str]]]]) -> dict[str, set[str]]:
    merged: dict[str, set[str]] = {}
    for _, domains in per_model:
        for domain, ops in domains.items():
            merged.setdefault(domain, set()).update(ops)
    return merged


def write_ort_config(union: dict[str, set[str]], config_path: Path) -> None:
    """Write the ORT reduced-build config file.

    Format: one line per domain, ``<domain>;<op1>,op2,...``. The default
    ``ai.onnx`` domain is written with an empty domain prefix (``;Conv,Add,...``)
    matching ORT's ``create_reduced_build_config.py`` output. Domains are sorted
    for deterministic output; ops within a domain are sorted.
    """
    lines: list[str] = []
    for domain in sorted(union):
        ops = sorted(union[domain])
        lines.append(f"{domain};{','.join(ops)}")
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return size, h.hexdigest()


def _resolve_catalog_models(manifest_path: Path, download_dir: Path) -> list[Path]:
    """Resolve every ONNX model artifact in a catalog manifest to a local file.

    Downloads each model's ``download_url`` into ``download_dir`` if not already
    present, verifying the SHA-256 against the manifest's ``archive_digest``.
    ORT-format bundles (``model.format == "ort"``) are also included; their
    operator set is extracted the same way (ORT format wraps an ONNX graph).
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = manifest.get("artifacts", {}).get("models", [])
    bundles = manifest.get("artifacts", {}).get("bundles", [])
    download_dir.mkdir(parents=True, exist_ok=True)
    resolved: list[Path] = []

    for art in models:
        fmt = art.get("model", {}).get("format", "onnx")
        if fmt not in ("onnx", "ort"):
            continue
        url = art["download_url"]
        fname = art["filename"]
        dest = download_dir / fname
        expected_sha = art["archive_digest"]
        if not dest.is_file() or _sha256_file(dest)[1] != expected_sha:
            print(f"  downloading {art['artifact_id']} from {url}...")
            with urlopen(url) as resp, dest.open("wb") as out:  # noqa: S310
                out.write(resp.read())
            got = _sha256_file(dest)[1]
            if got != expected_sha:
                raise RuntimeError(
                    f"downloaded model {fname} sha256 mismatch "
                    f"(expected {expected_sha}, got {got})"
                )
        resolved.append(dest)

    for bundle in bundles:
        # ORT-format bundles ship an .ort model file; extract ops from it.
        fmt = bundle.get("bundle", {}).get("model_format", "onnx")
        if fmt != "ort":
            continue
        url = bundle["download_url"]
        fname = bundle["filename"]
        dest = download_dir / fname
        expected_sha = bundle["archive_digest"]
        if not dest.is_file() or _sha256_file(dest)[1] != expected_sha:
            print(f"  downloading bundle {bundle['artifact_id']} from {url}...")
            with urlopen(url) as resp, dest.open("wb") as out:  # noqa: S310
                out.write(resp.read())
            got = _sha256_file(dest)[1]
            if got != expected_sha:
                raise RuntimeError(
                    f"downloaded bundle {fname} sha256 mismatch "
                    f"(expected {expected_sha}, got {got})"
                )
        resolved.append(dest)

    return resolved


def generate(model_paths: list[Path], output_stem: Path) -> dict[str, Any]:
    """Generate the config + sidecar. Returns the sidecar dict."""
    per_model: list[tuple[str, dict[str, set[str]]]] = []
    for p in model_paths:
        domains = extract_operators(p)
        size, sha = _sha256_file(p)
        per_model.append((p.name, domains))
        print(f"  {p.name}: {sum(len(o) for o in domains.values())} ops across "
              f"{len(domains)} domain(s)")

    union = merge_operator_sets(per_model)
    config_path = output_stem.with_suffix(".config")
    sidecar_path = output_stem.with_suffix(".json")
    write_ort_config(union, config_path)

    lock_ref: dict[str, Any] = {}
    if LOCK_PATH.is_file():
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        lock_ref = {
            "upstream_tag": lock["upstream"]["tag"],
            "upstream_commit_sha": lock["upstream"]["commit_sha"],
        }

    union_serializable = {d: sorted(ops) for d, ops in sorted(union.items())}
    sidecar = {
        "schema_version": GENERATOR_VERSION,
        "source_lock": lock_ref,
        "contributors": [
            {
                "filename": name,
                "operator_domains": {d: sorted(ops) for d, ops in sorted(domains.items())},
            }
            for name, domains in per_model
        ],
        "union_operator_set": union_serializable,
        "config_file": {
            "path": config_path.name,
            "sha256": _sha256_file(config_path)[1],
        },
    }
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"  config:  {config_path} ({len(union_serializable)} domain(s))")
    print(f"  sidecar: {sidecar_path}")
    return sidecar


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ORT reduced-build operator config.")
    parser.add_argument("--models", nargs="*", type=Path, default=None,
                        help="ONNX/ORT model files to scan.")
    parser.add_argument("--catalog", type=Path, default=None,
                        help="Catalog release manifest to resolve models from.")
    parser.add_argument("--download-dir", type=Path, default=ROOT / "ort" / "model-cache",
                        help="Directory for downloaded catalog models.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_STEM,
                        help="Output stem (writes <stem>.config and <stem>.json).")
    args = parser.parse_args()

    if args.models:
        model_paths = args.models
    elif args.catalog:
        if not args.catalog.is_file():
            print(f"ERROR: catalog not found: {args.catalog}", file=sys.stderr)
            return 1
        model_paths = _resolve_catalog_models(args.catalog, args.download_dir)
    else:
        parser.error("provide --models or --catalog")

    if not model_paths:
        print("ERROR: no models to scan", file=sys.stderr)
        return 1

    print(f"Generating required-operator config from {len(model_paths)} model(s)...")
    generate(model_paths, args.output)
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
