#!/usr/bin/env python3
"""Qualify a runtime-only stable promotion (read-only, no network writes).

A runtime-only promotion moves the stable pointer to an already-published
candidate release whose model artifacts are byte-identical to the current
stable release. It is authorised only by:

1. an external qualification record at
   ``catalog/qualifications/<release-id>.json`` (outside the immutable
   manifest, so there is no manifest/evidence hash cycle);
2. immutable native-comparison reports, produced by
   ``scripts/run_runtime_only_comparison.py`` on each native target, covering
   every supported compatibility edge exactly once with strict provider
   assignment, no fallback, exact output shapes, finite outputs, numeric
   old/new deltas and the existing non-regression budgets;
3. immutable OpenKara app-integration reports covering every app-applicable
   edge (non-deprecated spectral-core, the only production path in
   OpenKara#172) with the loaded library digest, provider assignment, output
   shapes and finite audio-pipeline output, and explicitly listing every
   non-applicable supported edge in ``skipped_edges``. Native reports still
   cover every supported edge, so waveform model↔runtime capability is never
   dropped.

Missing, duplicate, stale, synthetic or unmeasured evidence fails closed. The
record itself only carries refs; report bytes are downloaded and verified.
Hashes and refs establish immutability and schema conformance, not that a
producer actually executed inference or the app; see
``docs/runtime-only-promotion.md`` for the trust boundary.

After promotion the record is historical: ``--historical`` revalidates its
committed evidence without requiring the current stable/candidate pointers to
still equal the qualification-time ones. Promotion itself always requires the
current pointers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import enforce_quality_gates as gates  # noqa: E402
import native_host  # noqa: E402
import runtime_headers  # noqa: E402
from validate_catalog import validate_document  # noqa: E402
from runtime_interfaces import InterfaceError, parse_output_shapes  # noqa: E402
from validate_release_evidence import load_report  # noqa: E402

SCHEMA_PATH = ROOT / "catalog" / "schema-runtime-only-v1.json"
RECORD_DIR = ROOT / "catalog" / "qualifications"
RELEASES_DIR = ROOT / "catalog" / "releases"
CHANNELS_DIR = ROOT / "catalog" / "channels"
NATIVE_SCHEMA_VERSION = "openkara.runtime-only-native-report/v1"
APP_SCHEMA_VERSION = "openkara.openkara-app-integration/v1"
LIBRARY_NAMES = {"libonnxruntime.so", "libonnxruntime.dylib", "onnxruntime.dll"}
# OpenKara's production separation path is spectral-core only: the waveform
# graph path was removed in OpenKara#172. A catalog waveform edge still
# describes model<->runtime capability, not app loadability. App reports cover
# only supported edges whose model is non-deprecated and uses one of these
# interfaces; every other supported edge must be listed explicitly in the
# report's skipped_edges instead of being silently dropped.
APP_LOADABLE_TENSOR_INTERFACES = frozenset({"spectral-core"})


def _defs() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))["$defs"]


def _validate(doc: dict[str, Any], definition: str) -> None:
    jsonschema.Draft202012Validator(
        {"$ref": f"#/$defs/{definition}", "$defs": _defs()}
    ).validate(doc)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _family(artifact_id: str) -> str:
    marker = "-openkara-"
    if marker not in artifact_id:
        raise ValueError(f"runtime artifact id has no target family: {artifact_id}")
    return artifact_id.split(marker, 1)[1]


def _runtime_families(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    families: dict[str, dict[str, Any]] = {}
    for artifact in manifest["artifacts"]["runtimes"]:
        family = _family(artifact["artifact_id"])
        if family in families:
            raise ValueError(f"duplicate runtime family: {family}")
        families[family] = artifact
    return families


def load_record(release_id: str, root: Path = ROOT) -> dict[str, Any]:
    """Load and schema-validate the external qualification record."""
    record = _load(root / "catalog" / "qualifications" / f"{release_id}.json")
    _validate(record, "qualification")
    return record


def _manifest_bytes(root: Path, release_id: str) -> bytes:
    return (root / "catalog" / "releases" / f"{release_id}.json").read_bytes()


def _check_record_bindings(manifest: dict[str, Any], manifest_bytes: bytes,
                           record: dict[str, Any], root: Path, *,
                           require_current_pointers: bool,
                           expected_baseline: dict[str, Any] | None = None) -> list[str]:
    errors: list[str] = []
    digest = _sha256(manifest_bytes)
    if record["release_id"] != manifest["release_id"]:
        errors.append("record release_id differs from the manifest")
    if record["generation"] != manifest["generation"]:
        errors.append("record generation differs from the manifest")
    if record["manifest"] != {"sha256": digest, "size": len(manifest_bytes)}:
        errors.append("record does not bind these exact candidate manifest bytes")

    baseline_bytes = _manifest_bytes(root, record["baseline_release_id"])
    if {"sha256": _sha256(baseline_bytes), "size": len(baseline_bytes)} != record["baseline_manifest"]:
        errors.append("record baseline bytes differ from the committed baseline manifest")

    stable_path = root / "catalog" / "channels" / "stable.json"
    if not stable_path.is_file():
        errors.append("stable channel pointer is missing")
        stable = {}
    else:
        stable = _load(stable_path)
    if require_current_pointers:
        candidate = _load(root / "catalog" / "channels" / "candidate.json")
        if stable.get("release_id") != record["baseline_release_id"]:
            errors.append("record baseline is not the current stable release")
        if stable.get("release_manifest_sha256") != record["baseline_manifest"]["sha256"] or \
                stable.get("release_manifest_size") != record["baseline_manifest"]["size"]:
            errors.append("record baseline does not bind the current stable pointer")
        if candidate.get("release_id") != manifest["release_id"] or \
                candidate.get("release_manifest_sha256") != digest or \
                candidate.get("release_manifest_size") != len(manifest_bytes):
            errors.append("candidate channel does not point at this exact candidate manifest")
    else:
        # Historical re-validation after promotion: the record still names the
        # stable release it was qualified against. A stable-pointer transition
        # guard supplies that previous pointer explicitly.
        if expected_baseline is not None:
            if expected_baseline.get("release_id") != record["baseline_release_id"] or \
                    expected_baseline.get("release_manifest_sha256") != record["baseline_manifest"]["sha256"] or \
                    expected_baseline.get("release_manifest_size") != record["baseline_manifest"]["size"]:
                errors.append("record baseline is not the promoted-from stable pointer")
        if stable.get("release_id") == manifest["release_id"]:
            if stable.get("release_manifest_sha256") != digest or \
                    stable.get("release_manifest_size") != len(manifest_bytes):
                errors.append("stable pointer does not bind this exact release manifest")

    baseline = json.loads(baseline_bytes)
    result = validate_document(baseline, schema="release")
    if not result.ok:
        errors.append(f"baseline manifest invalid: {result.all_codes()}")
    if baseline.get("generation", 0) >= manifest.get("generation", 0):
        errors.append("baseline generation is not older than the candidate")
    return errors


def _model_semantics(model_block: dict[str, Any]) -> dict[str, Any]:
    """Model contract fields that must be identical; catalog pointers may change."""
    return {key: value for key, value in model_block.items() if key != "compatible_runtime_ids"}


def _check_model_identity(manifest: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    candidate_models = {a["artifact_id"]: a for a in manifest["artifacts"]["models"]}
    baseline_models = {a["artifact_id"]: a for a in baseline["artifacts"]["models"]}
    if set(candidate_models) != set(baseline_models):
        return ["runtime-only promotion requires the identical model artifact set"]
    errors: list[str] = []
    for artifact_id, artifact in candidate_models.items():
        other = baseline_models[artifact_id]
        if artifact["archive_digest"] != other["archive_digest"] or \
                artifact["byte_size"] != other["byte_size"]:
            errors.append(f"{artifact_id}: archive bytes differ from the stable release")
        if artifact["extracted_file_digests"] != other["extracted_file_digests"]:
            errors.append(f"{artifact_id}: extracted model bytes differ from the stable release")
        if _model_semantics(artifact["model"]) != _model_semantics(other["model"]):
            errors.append(f"{artifact_id}: model semantics differ from the stable release")
    return errors


def _check_runtime_families(manifest: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    candidate_by_family = _runtime_families(manifest)
    baseline_by_family = _runtime_families(baseline)
    errors: list[str] = []
    if set(candidate_by_family) != set(baseline_by_family):
        errors.append("runtime target/reduced families differ from the stable release")
    for family, artifact in candidate_by_family.items():
        version = str(artifact.get("runtime", {}).get("version", "")).lstrip("v")
        if version and f"-{version}-" not in artifact["artifact_id"]:
            errors.append(f"{family}: runtime.version does not match the artifact id")
        other = baseline_by_family.get(family)
        if other is None:
            continue
        if artifact["target_triple"] != other["target_triple"]:
            errors.append(f"{family}: target triple changed")
        if artifact.get("runtime", {}).get("execution_providers") != \
                other.get("runtime", {}).get("execution_providers"):
            errors.append(f"{family}: execution provider set changed")
    if not any(
        artifact["archive_digest"] != baseline_by_family[family]["archive_digest"]
        for family, artifact in candidate_by_family.items()
        if family in baseline_by_family
    ):
        errors.append("runtime-only promotion requires at least one different runtime archive")
    return errors


def _check_edge_preservation(manifest: dict[str, Any], baseline: dict[str, Any],
                             supported: dict[tuple, dict[str, Any]]) -> list[str]:
    """Reject dropping or unsupporting any previously supported stable edge."""
    errors: list[str] = []
    candidate_by_family = _runtime_families(manifest)
    for edge in baseline["compatibility"]:
        if edge["status"] != "supported":
            continue
        family = _family(edge["runtime_artifact_id"])
        candidate_runtime = candidate_by_family.get(family)
        if candidate_runtime is None:
            errors.append(f"candidate drops runtime family {family}")
            continue
        key = (edge["model_artifact_id"], candidate_runtime["artifact_id"],
               edge["target_triple"], edge["execution_provider"])
        if key not in supported:
            errors.append(
                "candidate drops a previously supported stable edge: "
                f"{edge['model_artifact_id']} / {family} / {edge['target_triple']} / "
                f"{edge['execution_provider']}"
            )
    return errors


def _supported_edges(manifest: dict[str, Any]) -> dict[tuple, dict[str, Any]]:
    models = {a["artifact_id"] for a in manifest["artifacts"]["models"]}
    runtimes = {a["artifact_id"]: a for a in manifest["artifacts"]["runtimes"]}
    edges: dict[tuple, dict[str, Any]] = {}
    for edge in manifest["compatibility"]:
        if edge["status"] != "supported":
            continue
        key = (edge["model_artifact_id"], edge["runtime_artifact_id"],
               edge["target_triple"], edge["execution_provider"])
        if key in edges:
            raise ValueError(f"duplicate compatibility edge: {key}")
        if key[0] not in models or key[1] not in runtimes:
            raise ValueError(f"dangling compatibility edge: {key}")
        runtime = runtimes[key[1]]
        if runtime["target_triple"] != key[2]:
            raise ValueError(f"edge target differs from runtime target: {key}")
        if key[3] not in runtime.get("runtime", {}).get("execution_providers", []):
            raise ValueError(f"edge provider is not shipped by the runtime: {key}")
        edges[key] = edge
    if not edges:
        raise ValueError("candidate manifest has no supported compatibility edges")
    return edges


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _check_metrics(metrics: dict[str, Any], provider: str, expected_shapes: list[list[int]],
                   frames: int, runtime: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if metrics["requested_provider"] != provider:
        problems.append("requested provider mismatch")
    if metrics["assigned_provider"] != provider:
        problems.append(f"provider assignment {metrics['assigned_provider']!r} != {provider!r}")
    if metrics["used_fallback"] is not False:
        problems.append("whole-session CPU fallback was used or not measured")
    for field in ("provider_node_count", "total_node_count"):
        if type(metrics[field]) is not int or metrics[field] <= 0:
            problems.append(f"{field} must be > 0")
    # An accelerated provider may keep part of the graph on the CPU EP; the
    # issue forbids CPU fallback standing in for the provider, not a recorded
    # partial partition. Require the partition to be explicit and accounted for.
    if provider != "cpu":
        cpu_nodes = metrics["fallback_node_count"]
        if type(cpu_nodes) is not int or cpu_nodes < 0:
            problems.append(
                f"accelerator CPU partition node count must be >= 0, got {cpu_nodes!r}"
            )
        elif type(metrics["provider_node_count"]) is int and \
                type(metrics["total_node_count"]) is int and \
                metrics["provider_node_count"] + cpu_nodes != metrics["total_node_count"]:
            problems.append(
                "accelerator partition accounting mismatch: provider + CPU nodes "
                f"({metrics['provider_node_count']} + {cpu_nodes}) != total "
                f"({metrics['total_node_count']})"
            )
    if metrics["finite_output"] is not True:
        problems.append("outputs were not finite")
    if metrics["output_shapes"] != expected_shapes:
        problems.append("measured output shapes differ from the catalog contract")
    if metrics["frames"] != frames:
        problems.append("measured frame count differs from the model segment")
    if metrics["installed_size"] != sum(
        value["size"] for value in runtime["extracted_file_digests"].values()
    ):
        problems.append("installed size differs from the declared runtime files")
    if metrics["archive_bytes"] != runtime["byte_size"] or \
            metrics["archive_sha256"] != runtime["archive_digest"]:
        problems.append("measured archive does not match the catalog runtime artifact")
    problems.extend(runtime_headers.verify_provenance(
        metrics["header_provenance"], runtime,
    ))
    for field in ("cold_load_s", "first_window_s", "warm_median_s", "warm_p95_s",
                  "rtf_warm", "peak_rss_kb"):
        if not _finite_number(metrics[field]) or metrics[field] < 0:
            problems.append(f"{field} must be a finite non-negative number")
    if type(metrics["warm_iters"]) is not int or metrics["warm_iters"] < 1:
        problems.append("warm_iters must be >= 1")
    if metrics["warm_p95_s"] < metrics["warm_median_s"]:
        problems.append("warm_p95_s is below warm_median_s")
    if _finite_number(metrics["rtf_warm"]) and _finite_number(metrics["warm_median_s"]):
        duration = frames / 44100
        if not math.isclose(metrics["rtf_warm"], metrics["warm_median_s"] / duration,
                            rel_tol=1e-9, abs_tol=1e-12):
            problems.append("rtf_warm does not match warm_median_s / duration")
    return problems


def _budget_problems(baseline: dict[str, Any], candidate: dict[str, Any],
                     comparison: dict[str, Any]) -> list[str]:
    budgets = {budget["metric"]: budget for budget in gates._load_json(gates.BUDGETS_PATH)["budgets"]}
    problems: list[str] = []
    for metric, value in (
        ("mse", comparison["output_mse"]),
        ("max_abs_error", comparison["output_max_abs"]),
    ):
        problem = gates._check_budget(metric, value, 0, budgets[metric], "release")
        if problem:
            problems.append(problem)
    for metric in ("rtf_warm", "cold_load_s", "warm_median_s", "peak_rss_kb", "installed_size"):
        problem = gates._check_budget(metric, candidate[metric], baseline[metric],
                                      budgets[metric], "release")
        if problem:
            problems.append(problem)
    for metric, value in (
        ("shape_match", comparison["output_shapes_equal"]),
        ("onnx_has_nan", not comparison["finite"]),
        ("onnx_has_inf", not comparison["finite"]),
    ):
        problem = gates._check_budget(metric, value, None, budgets[metric], "release")
        if problem:
            problems.append(problem)
    return problems


def _load_report_doc(ref: dict[str, Any], definition: str, expected_version: str,
                     download: Callable[[str], bytes]) -> dict[str, Any]:
    doc = load_report(ref, download)
    _validate(doc, definition)
    if doc["schema_version"] != expected_version:
        raise ValueError(f"unexpected report schema version: {doc['schema_version']!r}")
    return doc


def _native_checks(manifest: dict[str, Any], baseline: dict[str, Any],
                   record: dict[str, Any], supported: dict[tuple, dict[str, Any]],
                   docs: list[dict[str, Any]]) -> tuple[list[str], dict[tuple, list[list[int]]]]:
    errors: list[str] = []
    candidate_models = {a["artifact_id"]: a for a in manifest["artifacts"]["models"]}
    candidate_runtimes = {a["artifact_id"]: a for a in manifest["artifacts"]["runtimes"]}
    baseline_by_family = _runtime_families(baseline)
    seen: dict[tuple, list[list[int]]] = {}
    candidate_digest = record["manifest"]["sha256"]
    baseline_digest = record["baseline_manifest"]["sha256"]

    for doc in docs:
        if doc["candidate_release_id"] != manifest["release_id"] or \
                doc["generation"] != manifest["generation"] or \
                doc["candidate_manifest_sha256"] != candidate_digest:
            errors.append(f"{doc['report_id']}: candidate binding mismatch")
            continue
        if doc["baseline_release_id"] != record["baseline_release_id"] or \
                doc["baseline_manifest_sha256"] != baseline_digest:
            errors.append(f"{doc['report_id']}: baseline binding mismatch")
            continue
        if doc["status"] != "passed":
            errors.append(f"{doc['report_id']}: native report status is {doc['status']!r}")
        environment = doc["environment"]
        if environment["target_triple"] != doc["target_triple"]:
            errors.append(
                f"{doc['report_id']}: environment target_triple differs from the report target"
            )
        errors.extend(
            f"{doc['report_id']}: {problem}"
            for problem in native_host.environment_errors(environment, doc["target_triple"])
        )
        for edge in doc["edges"]:
            identity = edge["identity"]
            key = (identity["model_artifact_id"], identity["candidate_runtime_artifact_id"],
                   identity["target_triple"], identity["execution_provider"])
            if key in seen:
                errors.append(f"{doc['report_id']}: duplicate edge {key}")
                continue
            if key not in supported:
                errors.append(f"{doc['report_id']}: edge is not supported by the candidate: {key}")
                continue
            if identity["target_triple"] != doc["target_triple"]:
                errors.append(f"{doc['report_id']}: edge target differs from the report target")
            model = candidate_models[key[0]]
            candidate_runtime = candidate_runtimes[key[1]]
            baseline_runtime = baseline_by_family.get(_family(key[1]))
            if baseline_runtime is None:
                errors.append(f"{doc['report_id']}: no baseline runtime for {key[1]}")
                continue
            expected_identity = {
                "model_artifact_id": key[0],
                "baseline_runtime_artifact_id": baseline_runtime["artifact_id"],
                "candidate_runtime_artifact_id": key[1],
                "target_triple": key[2],
                "execution_provider": key[3],
                "model_digest": model["archive_digest"],
                "model_size": model["byte_size"],
                "baseline_runtime_digest": baseline_runtime["archive_digest"],
                "baseline_runtime_size": baseline_runtime["byte_size"],
                "candidate_runtime_digest": candidate_runtime["archive_digest"],
                "candidate_runtime_size": candidate_runtime["byte_size"],
            }
            if identity != expected_identity:
                errors.append(f"{doc['report_id']}: edge identity differs from the manifests")
                continue
            declared = {
                name: (value["sha256"], value["size"])
                for name, value in model["extracted_file_digests"].items()
            }
            rows = edge["model_files"]
            observed = {(row["path"], row["sha256"], row["size"]) for row in rows}
            if len(rows) != len(observed) or observed != {
                (name, sha, size) for name, (sha, size) in declared.items()
            }:
                errors.append(f"{doc['report_id']}: model_files differ from the candidate model")
                continue
            expected_shapes = edge["expected_output_shapes"]
            try:
                catalog_shapes = parse_output_shapes(
                    model["model"]["output_semantics"], model["model"]["tensor_interface"],
                )
            except InterfaceError as exc:
                errors.append(f"{model['artifact_id']}: {exc}")
                continue
            if expected_shapes != catalog_shapes:
                errors.append(
                    f"{doc['report_id']}: expected_output_shapes differ from the model contract"
                )
            problems: list[str] = []
            if edge["checks"]:
                problems.extend(edge["checks"])
            for label, runtime in (("baseline", baseline_runtime), ("candidate", candidate_runtime)):
                metrics = edge[label]
                if metrics is None:
                    problems.append(f"{label}: no measurement")
                else:
                    problems.extend(f"{label}: {problem}" for problem in _check_metrics(
                        metrics, key[3], expected_shapes, model["model"]["segment_frames"], runtime,
                    ))
            comparison = edge["comparison"]
            if comparison["output_shapes_equal"] is not True or \
                    comparison["finite"] is not True or comparison["passed"] is not True:
                problems.append("numeric comparison did not pass")
            for field in ("output_max_abs", "output_mse"):
                if not _finite_number(comparison[field]) or comparison[field] < 0:
                    problems.append(f"{field} must be a finite non-negative number")
            if edge["baseline"] is not None and edge["candidate"] is not None:
                problems.extend(_budget_problems(edge["baseline"], edge["candidate"], comparison))
            if problems:
                errors.extend(f"{doc['report_id']} {key}: {problem}" for problem in problems)
            seen[key] = expected_shapes

    missing = set(supported) - set(seen)
    if missing:
        errors.append(f"native reports do not cover {len(missing)} supported edge(s), e.g. {sorted(missing)[0]}")
    return errors, seen


def _app_applicable_edges(manifest: dict[str, Any],
                          supported: dict[tuple, dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    """Return supported edges the current OpenKara production path can load.

    OpenKara resolves only non-deprecated spectral-core models. The remaining
    supported edges stay in the candidate for native API compatibility, but
    they are not app-applicable and must be reported as explicit skips.
    """
    models = {a["artifact_id"]: a for a in manifest["artifacts"]["models"]}
    applicable: dict[tuple, dict[str, Any]] = {}
    for key, edge in supported.items():
        model = models[key[0]]
        if model.get("deprecation", {}).get("deprecated"):
            continue
        if model.get("model", {}).get("tensor_interface") in APP_LOADABLE_TENSOR_INTERFACES:
            applicable[key] = edge
    return applicable


def _app_checks(manifest: dict[str, Any], record: dict[str, Any],
                supported: dict[tuple, dict[str, Any]],
                native_shapes: dict[tuple, list[list[int]]],
                docs: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    candidate_models = {a["artifact_id"]: a for a in manifest["artifacts"]["models"]}
    candidate_runtimes = {a["artifact_id"]: a for a in manifest["artifacts"]["runtimes"]}
    applicable = _app_applicable_edges(manifest, supported)
    seen: set[tuple] = set()
    skipped_seen: dict[tuple, dict[str, Any]] = {}

    for doc in docs:
        if doc["candidate_release_id"] != manifest["release_id"] or \
                doc["generation"] != manifest["generation"] or \
                doc["candidate_manifest_sha256"] != record["manifest"]["sha256"]:
            errors.append(f"{doc['report_id']}: candidate binding mismatch")
            continue
        if doc["status"] != "passed":
            errors.append(f"{doc['report_id']}: app report status is {doc['status']!r}")
        for skipped in doc["skipped_edges"]:
            key = (skipped["model_artifact_id"], skipped["runtime_artifact_id"],
                   skipped["target_triple"], skipped["execution_provider"])
            if key in skipped_seen:
                errors.append(f"{doc['report_id']}: duplicate skipped edge {key}")
                continue
            if key in seen:
                errors.append(f"{doc['report_id']}: edge is both measured and skipped: {key}")
                continue
            if key not in supported:
                errors.append(
                    f"{doc['report_id']}: skipped edge is not supported by the candidate: {key}"
                )
                continue
            if key in applicable:
                errors.append(
                    f"{doc['report_id']}: app-applicable edge was skipped: {key}"
                )
                continue
            if skipped["target_triple"] != doc["target_triple"]:
                errors.append(
                    f"{doc['report_id']}: skipped edge target differs from the report target"
                )
            model = candidate_models[key[0]]
            interface = model.get("model", {}).get("tensor_interface")
            deprecated = model.get("deprecation", {}).get("deprecated")
            if skipped["tensor_interface"] != interface:
                errors.append(
                    f"{doc['report_id']}: skipped edge {key} declares tensor_interface "
                    f"{skipped['tensor_interface']!r} but the candidate model is {interface!r}"
                )
            if skipped["deprecated"] is not deprecated:
                errors.append(
                    f"{doc['report_id']}: skipped edge {key} declares deprecated "
                    f"{skipped['deprecated']!r} but the candidate model is {deprecated!r}"
                )
            in_scope = {
                "interface": interface not in APP_LOADABLE_TENSOR_INTERFACES,
                "deprecated": deprecated is True,
            }
            if in_scope.get(skipped["scope"]) is not True:
                errors.append(
                    f"{doc['report_id']}: skipped edge {key} declares scope "
                    f"{skipped['scope']!r}, which the candidate manifest does not support "
                    f"(interface {interface!r}, deprecated {deprecated!r})"
                )
            skipped_seen[key] = skipped
        for edge in doc["edges"]:
            identity = edge["identity"]
            key = (identity["model_artifact_id"], identity["runtime_artifact_id"],
                   identity["target_triple"], identity["execution_provider"])
            if key in seen:
                errors.append(f"{doc['report_id']}: duplicate edge {key}")
                continue
            if key in skipped_seen:
                errors.append(f"{doc['report_id']}: edge is both skipped and measured: {key}")
                continue
            if key not in supported:
                errors.append(f"{doc['report_id']}: edge is not supported by the candidate: {key}")
                continue
            if key not in applicable:
                errors.append(
                    f"{doc['report_id']}: app measured a non-applicable edge: {key}"
                )
                continue
            if identity["target_triple"] != doc["target_triple"]:
                errors.append(f"{doc['report_id']}: edge target differs from the report target")
            model = candidate_models[key[0]]
            runtime = candidate_runtimes[key[1]]
            expected_identity = {
                "model_artifact_id": key[0],
                "runtime_artifact_id": key[1],
                "target_triple": key[2],
                "execution_provider": key[3],
                "model_digest": model["archive_digest"],
                "model_size": model["byte_size"],
                "runtime_digest": runtime["archive_digest"],
                "runtime_size": runtime["byte_size"],
            }
            if identity != expected_identity:
                errors.append(f"{doc['report_id']}: edge identity differs from the manifests")
                continue
            libraries = [
                name for name in runtime["extracted_file_digests"]
                if name.replace("\\", "/").rsplit("/", 1)[-1].lower() in LIBRARY_NAMES
            ]
            if len(libraries) != 1:
                errors.append(f"{doc['report_id']}: runtime does not declare exactly one library")
                continue
            library = libraries[0]
            loaded = edge["loaded_library"]
            if loaded["sha256"] != runtime["extracted_file_digests"][library]["sha256"] or \
                    loaded["name"].replace("\\", "/").rsplit("/", 1)[-1].lower() != \
                    library.replace("\\", "/").rsplit("/", 1)[-1].lower():
                errors.append(f"{doc['report_id']}: loaded library differs from the runtime archive")
            provider = edge["provider"]
            if provider["requested"] != key[3]:
                errors.append(f"{doc['report_id']}: requested provider mismatch")
            if provider["assigned"] != key[3]:
                errors.append(f"{doc['report_id']}: provider assignment mismatch")
            if provider["used_fallback"] is not False:
                errors.append(f"{doc['report_id']}: app used whole-session CPU fallback")
            if type(provider["provider_node_count"]) is not int or provider["provider_node_count"] <= 0 or \
                    type(provider["total_node_count"]) is not int or provider["total_node_count"] <= 0:
                errors.append(f"{doc['report_id']}: provider node counts must be > 0")
            elif key[3] != "cpu":
                cpu_nodes = provider["fallback_node_count"]
                if type(cpu_nodes) is not int or cpu_nodes < 0:
                    errors.append(
                        f"{doc['report_id']}: app CPU partition node count must be >= 0"
                    )
                elif provider["provider_node_count"] + cpu_nodes != provider["total_node_count"]:
                    errors.append(
                        f"{doc['report_id']}: app accelerator partition accounting "
                        "mismatch: provider + CPU nodes != total"
                    )
            if edge["finite_output"] is not True:
                errors.append(f"{doc['report_id']}: app output was not finite")
            if edge["output_shapes"] != native_shapes.get(key):
                errors.append(f"{doc['report_id']}: app output shapes differ from native measurement")
            pipeline = edge["audio_pipeline"]
            if pipeline["sample_rate"] != model["model"]["sample_rate"]:
                errors.append(f"{doc['report_id']}: app sample rate differs from the model contract")
            if pipeline["frames_processed"] < model["model"]["segment_frames"]:
                errors.append(f"{doc['report_id']}: app processed less than one full window")
            if pipeline["stems"] != 4 or pipeline["finite"] is not True:
                errors.append(f"{doc['report_id']}: app pipeline must produce four finite stems")
            if edge["checks"]:
                errors.extend(f"{doc['report_id']}: {problem}" for problem in edge["checks"])
            seen.add(key)

    missing = set(applicable) - seen
    if missing:
        errors.append(
            f"app reports do not cover {len(missing)} app-applicable edge(s), "
            f"e.g. {sorted(missing)[0]}"
        )
    non_applicable = set(supported) - set(applicable)
    unaccounted = non_applicable - set(skipped_seen)
    if unaccounted:
        errors.append(
            f"app reports do not explicitly skip {len(unaccounted)} non-applicable edge(s), "
            f"e.g. {sorted(unaccounted)[0]}"
        )
    extra_skipped = set(skipped_seen) - non_applicable
    if extra_skipped:
        errors.append(
            f"app reports skip {len(extra_skipped)} edge(s) that are not non-applicable, "
            f"e.g. {sorted(extra_skipped)[0]}"
        )
    return errors


def _check_producer_repo(manifest: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """The candidate must keep the baseline's trusted catalog repository."""
    candidate_repo = manifest.get("producer", {}).get("repo")
    baseline_repo = baseline.get("producer", {}).get("repo")
    if not baseline_repo:
        return ["baseline manifest has no producer repository"]
    if candidate_repo != baseline_repo:
        return [
            f"candidate producer repository {candidate_repo!r} differs from "
            f"the trusted stable repository {baseline_repo!r}"
        ]
    return []


def _check_report_repositories(record: dict[str, Any], trusted_repo: str) -> list[str]:
    """Reports must live in immutable assets of the baseline catalog repository."""
    prefix = f"https://github.com/{trusted_repo}/releases/download/"
    return [
        f"report asset is outside {trusted_repo}: {ref['url']}"
        for ref in [*record["native_reports"], *record["app_reports"]]
        if not ref["url"].startswith(prefix)
    ]


def qualify_runtime_only(manifest: dict[str, Any], manifest_bytes: bytes,
                         record: dict[str, Any],
                         download: Callable[[str], bytes],
                         root: Path = ROOT, *,
                         require_current_pointers: bool = True,
                         expected_baseline: dict[str, Any] | None = None) -> list[str]:
    """Return qualification errors; an empty list authorises runtime-only promotion."""
    try:
        _validate(record, "qualification")
        manifest_result = validate_document(manifest, schema="release")
        if not manifest_result.ok:
            return [
                f"candidate manifest invalid: {code}"
                for code in manifest_result.all_codes()
            ]
        errors = _check_record_bindings(
            manifest, manifest_bytes, record, root,
            require_current_pointers=require_current_pointers,
            expected_baseline=expected_baseline,
        )
        baseline = json.loads(_manifest_bytes(root, record["baseline_release_id"]))
        errors.extend(_check_producer_repo(manifest, baseline))
        errors.extend(_check_report_repositories(
            record, baseline.get("producer", {}).get("repo", ""),
        ))
        if errors:
            return errors
        errors.extend(_check_model_identity(manifest, baseline))
        errors.extend(_check_runtime_families(manifest, baseline))
        supported = _supported_edges(manifest)
        errors.extend(_check_edge_preservation(manifest, baseline, supported))
        if errors:
            return errors

        native_docs = [
            _load_report_doc(ref, "native_report", NATIVE_SCHEMA_VERSION, download)
            for ref in record["native_reports"]
        ]
        native_errors, native_shapes = _native_checks(
            manifest, baseline, record, supported, native_docs
        )
        errors.extend(native_errors)

        app_docs = [
            _load_report_doc(ref, "app_report", APP_SCHEMA_VERSION, download)
            for ref in record["app_reports"]
        ]
        errors.extend(_app_checks(manifest, record, supported, native_shapes, app_docs))
        return errors
    except (
        KeyError, TypeError, ValueError, OSError, json.JSONDecodeError,
        jsonschema.ValidationError,
    ) as exc:
        return [f"runtime-only qualification: {exc}"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, help="candidate release_id to qualify")
    parser.add_argument(
        "--historical", action="store_true",
        help="revalidate a committed record after promotion (does not require the "
             "current stable/candidate pointers to be the qualification-time ones)",
    )
    args = parser.parse_args()
    from publish_catalog_release import _download_report

    manifest_path = RELEASES_DIR / f"{args.release}.json"
    if not manifest_path.is_file():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1
    try:
        record = load_record(args.release)
    except OSError:
        print(f"ERROR: qualification record not found: {RECORD_DIR / (args.release + '.json')}",
              file=sys.stderr)
        return 1
    except jsonschema.ValidationError as exc:
        print(f"ERROR: invalid qualification record: {exc.message}", file=sys.stderr)
        return 1
    errors = qualify_runtime_only(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        manifest_path.read_bytes(), record, _download_report,
        require_current_pointers=not args.historical,
    )
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    print(f"OK: runtime-only qualification passed for {args.release}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
