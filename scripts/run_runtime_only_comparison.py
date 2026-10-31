#!/usr/bin/env python3
"""Produce the native runtime-only comparison report.

Runtime-only promotion compares the **stable** runtime archive against the
**candidate** runtime archive while feeding both exactly the same published
model bytes. This producer measures every supported compatibility edge for one
target on the current native host, for each requested provider, then records
the strict native measurements, output-shape/finiteness results, numeric
old-vs-new deltas, and the same non-regression budgets used by the strict
gate.

The report binds both manifests and every artifact digest; it is an immutable
release asset, referenced (never embedded) by
``catalog/qualifications/<release-id>.json``. A failed edge is recorded and
the command exits non-zero: there is no option to omit an edge, substitute a
Python wheel, or pass a skipped/fallback provider.

Usage::

    python scripts/run_runtime_only_comparison.py \\
        --candidate-release catalog/releases/2026-10-30-001.json \\
        --baseline-release catalog/releases/2026-08-12-001.json \\
        --artifacts-dir /path/to/candidate-and-baseline-artifacts \\
        --target aarch64-apple-darwin \\
        --report runtime-only-native-aarch64-apple-darwin.json

The artifacts directory must contain every referenced model and runtime
archive under its manifest filename. Nothing is downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import archive_utils  # noqa: E402
import enforce_quality_gates as gates  # noqa: E402
import native_host  # noqa: E402
from native_quality import NativeRuntime  # noqa: E402
from runtime_interfaces import InterfaceError, measurement_spec  # noqa: E402
import runtime_headers  # noqa: E402
from validate_catalog import validate_document  # noqa: E402

SCHEMA_VERSION = "openkara.runtime-only-native-report/v1"
FRAMES = 343980
PROVIDERS = ("cpu", "coreml", "xnnpack", "directml")
# Accelerated providers may leave part of the graph on the CPU EP. That is a
# recorded partition, not a fallback standing in for the provider. A candidate
# may not move substantially more work onto the CPU than the stable baseline.
CPU_PARTITION_MAX_FACTOR = 1.5


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _manifest_binding(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    manifest = json.loads(raw)
    result = validate_document(manifest, schema="release")
    if not result.ok:
        raise ValueError(f"invalid release manifest {path.name}: {result.all_codes()}")
    return manifest, hashlib.sha256(raw).hexdigest()


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
            raise ValueError(f"duplicate runtime family in manifest: {family}")
        families[family] = artifact
    return families


def _verify_artifact(directory: Path, artifact: dict[str, Any]) -> Path:
    path = directory / artifact["filename"]
    if not path.is_file():
        raise FileNotFoundError(f"missing artifact: {path}")
    size, digest = _sha256_file(path)
    if size != artifact["byte_size"] or digest != artifact["archive_digest"]:
        raise ValueError(f"{artifact['artifact_id']}: artifact digest/size mismatch")
    return path


def _materialize_model(archive: Path, artifact: dict[str, Any], destination: Path) -> tuple[Path, list[dict[str, Any]]]:
    """Extract one model artifact and verify every declared extracted file."""
    declared = artifact["extracted_file_digests"]
    destination.mkdir(parents=True, exist_ok=True)
    if artifact["filename"].endswith((".tar.gz", ".tgz", ".zip")):
        archive_utils.safe_extract(archive, destination)
    else:
        target = destination / artifact["filename"]
        shutil.copyfile(archive, target)
    files = [path for path in destination.rglob("*") if path.is_file()]
    actual = {
        path.relative_to(destination).as_posix(): {"sha256": _sha256_file(path)[1], "size": path.stat().st_size}
        for path in files
    }
    if actual != declared:
        raise ValueError(
            f"{artifact['artifact_id']}: extracted files differ from declared digests"
        )
    graphs = [name for name in declared if name.endswith(".onnx")]
    if len(graphs) != 1:
        raise ValueError(
            f"{artifact['artifact_id']}: exactly one ONNX graph is supported, found {graphs}"
        )
    records = [
        {"path": name, "sha256": value["sha256"], "size": value["size"]}
        for name, value in sorted(declared.items())
    ]
    return destination / graphs[0], records


def _model_semantics(model_block: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in model_block.items() if key != "compatible_runtime_ids"}


def _edge_plan(manifest: dict[str, Any], target: str,
               providers: list[str]) -> list[dict[str, Any]]:
    """Return the manifest edges to measure, rejecting partial provider coverage."""
    edges = [
        edge for edge in manifest["compatibility"]
        if edge["status"] == "supported" and edge["target_triple"] == target
    ]
    if not edges:
        raise ValueError(f"candidate has no supported edges for target {target}")
    selected_providers = sorted(set(providers)) if providers else sorted(
        {edge["execution_provider"] for edge in edges}
    )
    available = {edge["execution_provider"] for edge in edges}
    extra = set(selected_providers) - available
    if extra:
        raise ValueError(f"providers without supported edges: {sorted(extra)}")
    selected = [edge for edge in edges if edge["execution_provider"] in selected_providers]
    expected = {(edge["model_artifact_id"], edge["execution_provider"]) for edge in edges}
    covered = {(edge["model_artifact_id"], edge["execution_provider"]) for edge in selected}
    if covered != expected:
        raise ValueError(
            "every supported edge for the target must be measured; do not filter providers"
        )
    return sorted(
        selected,
        key=lambda edge: (
            edge["model_artifact_id"], edge["runtime_artifact_id"],
            edge["execution_provider"],
        ),
    )


def _identity(model: dict[str, Any], baseline: dict[str, Any], candidate: dict[str, Any],
              target: str, provider: str) -> dict[str, Any]:
    return {
        "model_artifact_id": model["artifact_id"],
        "baseline_runtime_artifact_id": baseline["artifact_id"],
        "candidate_runtime_artifact_id": candidate["artifact_id"],
        "target_triple": target,
        "execution_provider": provider,
        "model_digest": model["archive_digest"],
        "model_size": model["byte_size"],
        "baseline_runtime_digest": baseline["archive_digest"],
        "baseline_runtime_size": baseline["byte_size"],
        "candidate_runtime_digest": candidate["archive_digest"],
        "candidate_runtime_size": candidate["byte_size"],
    }


def _check_measurement(metrics: dict[str, Any], provider: str,
                       expected_shapes: list[list[int]]) -> list[str]:
    problems: list[str] = []
    if metrics.get("requested_provider") != provider:
        problems.append("requested provider mismatch")
    if metrics.get("assigned_provider") != provider:
        problems.append(f"provider assignment {metrics.get('assigned_provider')!r} != {provider!r}")
    if metrics.get("used_fallback") is not False:
        problems.append("whole-session CPU fallback was used or not measured")
    for field in ("provider_node_count", "total_node_count"):
        value = metrics.get(field)
        if type(value) is not int or value <= 0:
            problems.append(f"{field} must be > 0, got {value!r}")
    # An accelerated provider may partition part of the graph to the CPU EP;
    # the issue forbids CPU fallback standing in for the provider, not a
    # recorded partial partition. Require the partition to be explicitly
    # accounted for and compared, never described as full acceleration.
    if provider != "cpu":
        cpu_nodes = metrics.get("fallback_node_count")
        if type(cpu_nodes) is not int or cpu_nodes < 0:
            problems.append(
                f"accelerator CPU partition node count must be >= 0, got {cpu_nodes!r}"
            )
        elif metrics.get("provider_node_count") + cpu_nodes != metrics.get("total_node_count"):
            problems.append(
                "accelerator partition accounting mismatch: provider + CPU nodes "
                f"({metrics.get('provider_node_count')} + {cpu_nodes}) != total "
                f"({metrics.get('total_node_count')})"
            )
    if metrics.get("finite_output") is not True:
        problems.append("outputs were not finite")
    if metrics.get("output_shapes") != [list(shape) for shape in expected_shapes]:
        problems.append("measured output shapes differ from the catalog contract")
    for field in ("cold_load_s", "first_window_s", "warm_median_s", "warm_p95_s", "rtf_warm"):
        value = metrics.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            problems.append(f"{field} must be a finite non-negative number")
    if type(metrics.get("peak_rss_kb")) not in (int, float) or not math.isfinite(metrics["peak_rss_kb"]) \
            or metrics["peak_rss_kb"] < 0:
        problems.append("peak_rss_kb must be a finite non-negative number")
    return problems


def _compare_outputs(baseline: list[Any], candidate: list[Any]) -> dict[str, Any]:
    import numpy as np

    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("baseline and candidate output counts differ or are empty")
    shapes_equal, finite = True, True
    max_abs, squared_error, count = 0.0, 0.0, 0
    for old, new in zip(baseline, candidate):
        if old.shape != new.shape:
            shapes_equal = False
            continue
        if not np.isfinite(old).all() or not np.isfinite(new).all():
            finite = False
        diff = old.astype(np.float64) - new.astype(np.float64)
        max_abs = max(max_abs, float(np.abs(diff).max()))
        squared_error += float((diff ** 2).sum())
        count += diff.size
    mse = squared_error / count if count else 0.0
    return {
        "output_shapes_equal": shapes_equal,
        "finite": finite,
        "output_max_abs": max_abs,
        "output_mse": mse,
    }


def _budget_checks(baseline: dict[str, Any], candidate: dict[str, Any],
                   comparison: dict[str, Any]) -> list[str]:
    budgets = {budget["metric"]: budget for budget in gates._load_json(gates.BUDGETS_PATH)["budgets"]}
    problems: list[str] = []
    for metric in ("mse", "max_abs_error"):
        problem = gates._check_budget(
            metric, comparison["output_mse" if metric == "mse" else "output_max_abs"],
            0, budgets[metric], "release",
        )
        if problem:
            problems.append(problem)
    for metric in ("rtf_warm", "cold_load_s", "warm_median_s", "peak_rss_kb", "installed_size"):
        problem = gates._check_budget(
            metric, candidate[metric], baseline[metric], budgets[metric], "release",
        )
        if problem:
            problems.append(problem)
    # Same accounting as the baseline: a candidate accelerated runtime must not
    # move substantially more of the graph onto the CPU EP than stable did.
    provider = candidate.get("requested_provider") or baseline.get("requested_provider")
    if provider != "cpu":
        base_cpu = baseline.get("fallback_node_count")
        cand_cpu = candidate.get("fallback_node_count")
        if type(base_cpu) is not int or type(cand_cpu) is not int:
            problems.append("accelerator CPU partition node counts must be integers")
        elif cand_cpu > max(base_cpu, 1) * CPU_PARTITION_MAX_FACTOR:
            problems.append(
                f"candidate CPU partition {cand_cpu} exceeds baseline {base_cpu} "
                f"by more than {CPU_PARTITION_MAX_FACTOR}x"
            )
    for metric, value in (
        ("shape_match", comparison["output_shapes_equal"]),
        ("onnx_has_nan", not comparison["finite"]),
        ("onnx_has_inf", not comparison["finite"]),
    ):
        problem = gates._check_budget(metric, value, None, budgets[metric], "release")
        if problem:
            problems.append(problem)
    return problems


def _failing_row(identity: dict[str, Any], model_files: list[dict[str, Any]],
                 expected_shapes: list[list[int]], error: str) -> dict[str, Any]:
    return {
        "identity": identity,
        "model_files": model_files,
        "expected_output_shapes": expected_shapes,
        "baseline": None,
        "candidate": None,
        "comparison": {
            "output_shapes_equal": False,
            "finite": False,
            "output_max_abs": 0.0,
            "output_mse": 0.0,
            "passed": False,
        },
        "checks": [error],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-release", required=True, type=Path)
    parser.add_argument("--baseline-release", required=True, type=Path)
    parser.add_argument("--artifacts-dir", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--provider", action="append", choices=PROVIDERS, default=[])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    if args.warmup < 0 or args.iters < 1:
        parser.error("--warmup must be >= 0 and --iters >= 1")
    if not args.artifacts_dir.is_dir():
        parser.error(f"artifacts directory not found: {args.artifacts_dir}")

    candidate, candidate_sha = _manifest_binding(args.candidate_release)
    baseline, baseline_sha = _manifest_binding(args.baseline_release)
    if candidate["release_id"] == baseline["release_id"]:
        parser.error("candidate and baseline must be different releases")
    if candidate["generation"] <= baseline["generation"]:
        parser.error("candidate generation must be newer than the baseline")

    candidate_models = {a["artifact_id"]: a for a in candidate["artifacts"]["models"]}
    baseline_models = {a["artifact_id"]: a for a in baseline["artifacts"]["models"]}
    if set(candidate_models) != set(baseline_models):
        parser.error("runtime-only comparison requires identical model artifact sets")
    for artifact_id, artifact in candidate_models.items():
        other = baseline_models[artifact_id]
        for field in ("archive_digest", "byte_size", "extracted_file_digests"):
            if artifact[field] != other[field]:
                parser.error(f"{artifact_id}: model bytes differ between releases")
        if _model_semantics(artifact["model"]) != _model_semantics(other["model"]):
            parser.error(f"{artifact_id}: model semantics differ between releases")

    try:
        selected = _edge_plan(candidate, args.target, args.provider)
    except ValueError as exc:
        parser.error(str(exc))
    providers = sorted({edge["execution_provider"] for edge in selected})
    try:
        environment = native_host.describe_host(args.target)
    except (native_host.HostMismatch, RuntimeError) as exc:
        parser.error(f"native host verification failed: {exc}")

    candidate_runtimes = {a["artifact_id"]: a for a in candidate["artifacts"]["runtimes"]}
    baseline_by_family = _runtime_families(baseline)
    candidate_by_family = _runtime_families(candidate)
    if set(candidate_by_family) != set(baseline_by_family):
        parser.error("candidate and baseline runtime target/reduced families differ")

    now = datetime.now(timezone.utc)
    created_at = now.isoformat().replace("+00:00", "Z")
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "report_id": (
            f"runtime-only-native-{candidate['release_id']}-{args.target}-"
            f"{'-'.join(providers)}-{stamp}"
        ),
        "created_at": created_at,
        "candidate_release_id": candidate["release_id"],
        "generation": candidate["generation"],
        "candidate_manifest_sha256": candidate_sha,
        "baseline_release_id": baseline["release_id"],
        "baseline_manifest_sha256": baseline_sha,
        "target_triple": args.target,
        "providers": providers,
        "environment": environment,
        "status": "failed",
        "edges": [],
    }

    failures = 0
    with tempfile.TemporaryDirectory(prefix="runtime-only-") as temporary:
        work_root = Path(temporary)
        runtimes: dict[tuple[str, str], NativeRuntime] = {}
        pins: dict[str, Any] = {}
        # Materialize one model at a time and release it as soon as the sorted
        # edge list moves on, so peak disk stays bounded instead of growing
        # with every extracted graph. Coverage is unchanged.
        current_model_id: str | None = None
        model_path: Path | None = None
        model_files: list[dict[str, Any]] = []
        spec: Any = None

        for edge in selected:
            model_id = edge["model_artifact_id"]
            runtime_id = edge["runtime_artifact_id"]
            provider = edge["execution_provider"]
            candidate_runtime = candidate_runtimes[runtime_id]
            baseline_runtime = baseline_by_family.get(_family(runtime_id))
            if baseline_runtime is None:
                raise SystemExit(f"no baseline runtime for candidate family {_family(runtime_id)}")
            identity = _identity(
                candidate_models[model_id], baseline_runtime, candidate_runtime,
                args.target, provider,
            )
            if model_id != current_model_id:
                if current_model_id is not None:
                    shutil.rmtree(work_root / "models" / current_model_id, ignore_errors=True)
                model_archive = _verify_artifact(args.artifacts_dir, candidate_models[model_id])
                model_path, model_files = _materialize_model(
                    model_archive, candidate_models[model_id],
                    work_root / "models" / model_id,
                )
                try:
                    spec = measurement_spec(
                        candidate_models[model_id]["model"], model_path, FRAMES,
                    )
                except InterfaceError as exc:
                    raise SystemExit(f"{model_id}: {exc}") from exc
                current_model_id = model_id

            expected_shapes = [list(shape) for shape in spec.output_shapes]
            checks: list[str] = []
            try:
                for artifact, label in (
                    (baseline_runtime, "baseline"), (candidate_runtime, "candidate"),
                ):
                    archive = _verify_artifact(args.artifacts_dir, artifact)
                    key = (artifact["artifact_id"], provider)
                    if key not in runtimes:
                        if artifact["artifact_id"] not in pins:
                            pins[artifact["artifact_id"]] = runtime_headers.for_archive(archive)[0]
                        runtimes[key] = NativeRuntime(
                            archive, None, args.target, provider,
                            work_root / "runtimes" / f"{artifact['artifact_id']}-{provider}",
                            pinned_headers=pins[artifact["artifact_id"]],
                        )
                    declared_size = sum(value["size"] for value in artifact["extracted_file_digests"].values())
                    if runtimes[key].archive["installed_size"] != declared_size:
                        checks.append(f"{label}: installed size differs from declared extracted files")

                baseline_outputs, baseline_metrics, _ = runtimes[
                    (baseline_runtime["artifact_id"], provider)
                ].measure_raw(model_path, spec.inputs, expected_shapes, FRAMES, args.warmup, args.iters)
                candidate_outputs, candidate_metrics, _ = runtimes[
                    (candidate_runtime["artifact_id"], provider)
                ].measure_raw(model_path, spec.inputs, expected_shapes, FRAMES, args.warmup, args.iters)
                checks.extend(f"baseline: {problem}" for problem in
                              _check_measurement(baseline_metrics, provider, expected_shapes))
                checks.extend(f"candidate: {problem}" for problem in
                              _check_measurement(candidate_metrics, provider, expected_shapes))
                comparison = _compare_outputs(baseline_outputs, candidate_outputs)
                checks.extend(_budget_checks(baseline_metrics, candidate_metrics, comparison))
                comparison["passed"] = (
                    comparison["output_shapes_equal"] and comparison["finite"] and not checks
                )
                row = {
                    "identity": identity,
                    "model_files": model_files,
                    "expected_output_shapes": expected_shapes,
                    "baseline": baseline_metrics,
                    "candidate": candidate_metrics,
                    "comparison": comparison,
                    "checks": checks,
                }
            except (ValueError, OSError, RuntimeError, KeyError) as exc:
                row = _failing_row(identity, model_files, expected_shapes, str(exc))
            if row["checks"]:
                failures += 1
            report["edges"].append(row)

    report["status"] = "passed" if failures == 0 else "failed"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"{report['status'].upper()}: {len(report['edges'])} edges, {failures} failed")
    print(f"report: {args.report}")
    if failures:
        for row in report["edges"]:
            if row["checks"]:
                print(f"  {row['identity']['model_artifact_id']} / "
                      f"{row['identity']['execution_provider']}: {row['checks'][0]}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
