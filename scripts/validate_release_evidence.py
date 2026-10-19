#!/usr/bin/env python3
"""Qualify immutable release evidence before stable publication (no network writes)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Callable

import jsonschema
import enforce_quality_gates as gates
from run_native_smoke import validation_failures
from validate_quality_report import validate_report as validate_quality
from validate_runtime_quality_report import validate_report as validate_runtime

ROOT = Path(__file__).resolve().parents[1]
EDGE_FIELDS = ("model_artifact_id", "runtime_artifact_id", "target_triple", "execution_provider")
SEPARATION_DIRECTIONS = {
    "si_sdr_db": "higher-is-better", "sdr_db": "higher-is-better",
    "leakage_db": "lower-is-better", "mrstft_loss": "lower-is-better",
    "transient_error": "lower-is-better", "low_frequency_error_db": "lower-is-better",
    "chunk_boundary_error": "lower-is-better",
}
REF_SCHEMA = {"$ref": "#/$defs/report_ref", "$defs": json.loads(
    (ROOT / "catalog/schema-release-v1.json").read_text())["$defs"]}


def load_report(ref: dict, download: Callable[[str], bytes]) -> dict:
    """The hash and length bind the parsed report to the immutable catalog ref."""
    jsonschema.validate(ref, REF_SCHEMA)
    raw = download(ref["url"])
    if len(raw) != ref["size"] or hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise ValueError("report digest/size mismatch")
    doc = json.loads(raw)
    if not isinstance(doc, dict) or any(doc.get(k) != ref[k] for k in ("report_id", "schema_version")):
        raise ValueError("report identity mismatch")
    return doc


def _key(edge: dict) -> tuple:
    return tuple(edge[k] for k in EDGE_FIELDS)


def _index(doc: dict, manifest: dict, kind: str, expected: set[tuple]) -> dict:
    schema = {
        "type": "object", "additionalProperties": False,
        "required": ["schema_version", "report_id", "release_id", "generation", "edges"],
        "properties": {
            "schema_version": {"const": f"openkara.release-{kind}/v1"},
            "report_id": {"type": "string", "minLength": 1},
            "release_id": {"const": manifest["release_id"]},
            "generation": {"const": manifest["generation"]},
            "edges": {"type": "array", "minItems": 1, "items": {
                "type": "object", "required": ["identity"],
                "properties": {"identity": {
                    "type": "object", "additionalProperties": False,
                    "required": [*EDGE_FIELDS, "model_digest", "model_size", "runtime_digest", "runtime_size"],
                    "properties": {
                        **{k: {"type": "string", "minLength": 1} for k in EDGE_FIELDS},
                        **{k: {"type": "string", "pattern": "^[a-f0-9]{64}$"} for k in ("model_digest", "runtime_digest")},
                        **{k: {"type": "integer", "minimum": 1} for k in ("model_size", "runtime_size")},
                    },
                }},
            }},
        },
    }
    jsonschema.validate(doc, schema)
    rows = {_key(row["identity"]): row for row in doc["edges"]}
    if len(rows) != len(doc["edges"]) or set(rows) != expected:
        raise ValueError(f"{kind}: missing, duplicate or extra compatibility edges")
    return rows


def _fixture_coverage(report: dict, fixtures: dict) -> None:
    rows = report["results"]
    ids = [r["fixture_id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(fixtures):
        raise ValueError("missing, duplicate or extra corpus fixtures")
    for row in rows:
        fixture = fixtures[row["fixture_id"]]
        if row["tier"] != fixture["tier"] or row["category"] != fixture["category"]:
            raise ValueError("fixture tier/category mismatch")
        if "mse_threshold" in row:
            allowed = fixture.get("mse_budget", {}).get("release", report.get("mse_threshold"))
            if row["mse_threshold"] != allowed:
                raise ValueError("unreviewed per-fixture MSE budget")


def _separation(rows: list[dict], real_ids: set[str]) -> dict:
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
        raise ValueError("measured reference-stem separation results required")
    ids = [r.get("fixture_id") for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != real_ids:
        raise ValueError("separation reference-stem corpus coverage mismatch")
    for row in rows:
        for metric in SEPARATION_DIRECTIONS:
            value = row.get(metric)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{metric}: finite measured separation metric required")
    return {m: (min if d == "higher-is-better" else max)(r[m] for r in rows)
            for m, d in SEPARATION_DIRECTIONS.items()}


def qualify_release(manifest: dict, download: Callable[[str], bytes]) -> list[str]:
    """Require one measured, identity-bound result for every advertised edge."""
    try:
        refs = manifest.get("gates", {})
        docs = {k: load_report(refs[f"{k}_report"], download)
                for k in ("quality", "performance", "compatibility")}
        edges = {_key(e) for e in manifest["compatibility"] if e["status"] == "supported"}
        if not edges:
            raise ValueError("no supported compatibility edges")
        indexed = {k: _index(v, manifest, k, edges) for k, v in docs.items()}
        corpus_raw = (ROOT / "quality/corpus-manifest.json").read_bytes()
        corpus = json.loads(corpus_raw)
        jsonschema.validate(corpus, json.loads((Path(__file__).resolve().parents[1] /
                            "quality/corpus-manifest-v1.json").read_text()))
        fixtures = {f["fixture_id"]: f for f in corpus["fixtures"]}
        if len(fixtures) != len(corpus["fixtures"]):
            raise ValueError("duplicate corpus fixture IDs")
        real_ids = {k for k, f in fixtures.items() if f["kind"] == "real-audio"}
        if not real_ids:
            raise ValueError("licensed real reference-stem corpus is not frozen")
        for fid in real_ids:
            f = fixtures[fid]
            if f["license"] in {"", "generated", "pending", "unknown"} or not f["generator"].get("sha256"):
                raise ValueError(f"{fid}: licensed immutable real-audio source required")
        baselines = gates._load_json(gates.BASELINES_PATH)
        default_id = baselines["default_artifact_id"]
        models = {a["artifact_id"]: a for a in manifest["artifacts"]["models"]}
        runtimes = {a["artifact_id"]: a for a in manifest["artifacts"]["runtimes"]}
        errors = []
        for key in sorted(edges):
            try:
                q, p, c = (indexed[k][key] for k in ("quality", "performance", "compatibility"))
                model, runtime = models[key[0]], runtimes[key[1]]
                expected = dict(zip(EDGE_FIELDS, key)) | {
                    "model_digest": model["archive_digest"], "model_size": model["byte_size"],
                    "runtime_digest": runtime["archive_digest"], "runtime_size": runtime["byte_size"],
                }
                if any(row["identity"] != expected for row in (q, p, c)):
                    raise ValueError("candidate model/runtime digest, size or edge identity mismatch")
                if c.get("status") != "passed" or any(c.get(f"{k}_report_id") != docs[k]["report_id"]
                                                        for k in ("quality", "performance")):
                    raise ValueError("compatibility qualification/result identity mismatch")
                if q.get("corpus_sha256") != hashlib.sha256(corpus_raw).hexdigest():
                    raise ValueError("reviewed corpus digest mismatch")
                quality, performance, native = q["report"], p["report"], c["native_smoke"]
                for report, validator in ((quality, validate_quality), (performance, validate_runtime)):
                    problems = validator(report)
                    if problems or report["tier"] != "release":
                        raise ValueError(f"invalid release report: {problems}")
                    _fixture_coverage(report, fixtures)
                archive = performance["runtime_archive"]
                if (archive["sha256"], archive["size"]) != (runtime["archive_digest"], runtime["byte_size"]):
                    raise ValueError("performance runtime archive mismatch")
                if p.get("measurement_backend") != "native-c-api":
                    raise ValueError("native C API performance measurements required; Python wheel is not evidence")
                if native.get("target") != key[2] or validation_failures(native, key[3]):
                    raise ValueError("strict native provider qualification failed")
                if any(native.get(k, {}).get(f) != v for k, f, v in (
                    ("model_digest", "sha256", model["archive_digest"]),
                    ("model_digest", "size", model["byte_size"]),
                    ("runtime_archive", "sha256", runtime["archive_digest"]),
                    ("runtime_archive", "size", runtime["byte_size"]),
                )):
                    raise ValueError("native smoke candidate bytes mismatch")
                matches = [b for b in baselines["stable_baselines"]
                           if b["artifact_id"] == default_id and b["target"] == key[2] and b["provider"] == key[3]]
                if len(matches) != 1 or not gates._is_baseline_frozen(matches[0]):
                    raise ValueError("missing, duplicate or NOT FROZEN current-default edge baseline")
                baseline = matches[0]
                if p.get("benchmark_environment") != baseline["benchmark_environment"]:
                    raise ValueError("candidate/baseline benchmark environment mismatch")
                for kind in ("quality", "runtime"):
                    ref = q[f"baseline_{kind}_report"]
                    if (ref["report_id"], ref["sha256"]) != (
                        baseline[f"baseline_{kind}_report_id"], baseline[f"baseline_{kind}_report_digest"]
                    ):
                        raise ValueError("frozen baseline report identity mismatch")
                    old = load_report(ref, download)
                    if old.get("identity") != {"model_digest": baseline["model_digest"],
                                               "model_size": baseline["artifact_size_bytes"],
                                               "runtime_id": baseline["runtime_id"],
                                               "runtime_digest": baseline["runtime_digest"],
                                               "runtime_size": baseline["runtime_size"],
                                               "target": baseline["target"], "provider": baseline["provider"]}:
                        raise ValueError("baseline model/runtime/target/provider mismatch")
                    if old.get("corpus_sha256") != q["corpus_sha256"]:
                        raise ValueError("baseline corpus differs from candidate corpus")
                    if old.get("benchmark_environment") != baseline["benchmark_environment"]:
                        raise ValueError("baseline report environment mismatch")
                    if old.get("metrics") != baseline["baseline_metrics"]:
                        raise ValueError("baseline report metrics differ from frozen measurements")
                candidate = gates._aggregate_quality(quality) | gates._aggregate_runtime(performance)
                candidate.update(_separation(q["separation"], real_ids))
                problems = gates._per_fixture_budget_violations(quality)
                budgets = gates._load_json(gates.BUDGETS_PATH)["budgets"] + [
                    {"metric": m, "direction": d, "release_max_delta": 0}
                    for m, d in SEPARATION_DIRECTIONS.items()
                ]
                for budget in budgets:
                    metric = budget["metric"]
                    if metric not in baseline["baseline_metrics"]:
                        problems.append(f"baseline missing {metric}")
                        continue
                    problem = gates._check_budget(metric, candidate.get(metric), baseline["baseline_metrics"][metric], budget, "release")
                    if problem:
                        problems.append(problem)
                pareto = gates._check_pareto(candidate, baseline["baseline_metrics"], budgets)
                if pareto:
                    problems.append(pareto)
                if key[0] != default_id and not any(candidate[m] < baseline["baseline_metrics"][m] for m in
                           ("installed_size", "cold_load_s", "warm_median_s", "peak_rss_kb")) and not any(
                    (candidate[m] > baseline["baseline_metrics"][m] if d == "higher-is-better"
                     else candidate[m] < baseline["baseline_metrics"][m]) for m, d in SEPARATION_DIRECTIONS.items()
                ):
                    problems.append("Pareto: target-specific candidate must improve size, load, latency, memory or quality")
                errors.extend(f"{key}: {problem}" for problem in problems)
            except (KeyError, TypeError, ValueError, jsonschema.ValidationError) as exc:
                errors.append(f"{key}: {exc}")
        return errors
    except (KeyError, TypeError, ValueError, OSError, jsonschema.ValidationError) as exc:
        return [f"release evidence: {exc}"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    from publish_catalog_release import _download_report
    errors = qualify_release(json.loads(args.manifest.read_text()), _download_report)
    for error in errors:
        print(error, file=sys.stderr)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
