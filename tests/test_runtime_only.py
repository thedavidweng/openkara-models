"""Tests for runtime-only promotion evidence (no native measurement, no network).

These tests build synthetic catalog documents and synthetic reports that
satisfy the schemas, then exercise the fail-closed checks. Synthetic reports
are test fixtures here and are never evidence for a real promotion.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import native_host  # noqa: E402
import runtime_interfaces as interfaces  # noqa: E402
import validate_runtime_only as validator  # noqa: E402

BASELINE_ID = "2026-08-12-001"
CANDIDATE_ID = "2099-01-01-001"
FRAMES = 343980


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _waveform_model(tmp_path: Path, frames: int = 8, dual: bool = False) -> Path:
    inputs = [helper.make_tensor_value_info("mix", TensorProto.FLOAT, [1, 2, frames])]
    if dual:
        outputs = [
            helper.make_tensor_value_info("stems", TensorProto.FLOAT, [1, 4, 2, frames]),
            helper.make_tensor_value_info("accompaniment", TensorProto.FLOAT, [1, 2, 2, frames]),
        ]
    else:
        outputs = [helper.make_tensor_value_info("stems", TensorProto.FLOAT, [1, 4, 2, frames])]
    graph = helper.make_graph([], "stub", inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    path = tmp_path / "stub.onnx"
    onnx.save(model, str(path))
    return path


def test_parse_shape_literal_accepts_bracketed_capture() -> None:
    assert interfaces.parse_shape_literal("[1,4,2,343980]") == [1, 4, 2, 343980]
    assert interfaces.parse_shape_literal("1, 4, 2") == [1, 4, 2]
    with pytest.raises(interfaces.InterfaceError):
        interfaces.parse_shape_literal("[1,0,2]")


def test_parse_output_shapes_supports_all_declared_forms() -> None:
    assert interfaces.parse_output_shapes(
        "[1,4,2,343980] drums/bass/other/vocals", "waveform"
    ) == [[1, 4, 2, 343980]]
    assert interfaces.parse_output_shapes(
        "outputs[0]=[1,4,2,343980] stems; outputs[1]=[1,2,2,343980] vocals/accompaniment",
        "waveform",
    ) == [[1, 4, 2, 343980], [1, 2, 2, 343980]]
    assert interfaces.parse_output_shapes(
        "spectral_out=[1,4,2,2,2048,336] pre-ISTFT; time_out=[1,4,2,343980] time branch",
        "spectral-core",
    ) == [[1, 4, 2, 2, 2048, 336], [1, 4, 2, 343980]]


def test_measurement_spec_waveform_uses_one_deterministic_input(tmp_path) -> None:
    path = _waveform_model(tmp_path)
    model = {
        "sample_rate": 44100,
        "segment_frames": 8,
        "tensor_interface": "waveform",
        "output_semantics": "[1,4,2,8] drums/bass/other/vocals",
    }
    spec = interfaces.measurement_spec(model, path, 8)
    assert list(spec.inputs) == ["mix"]
    assert spec.inputs["mix"].shape == (1, 2, 8)
    assert spec.output_shapes == [[1, 4, 2, 8]]
    assert spec.output_shape_string == "[1,4,2,8]"


def test_measurement_spec_rejects_semantics_graph_mismatch(tmp_path) -> None:
    path = _waveform_model(tmp_path, dual=True)
    model = {
        "sample_rate": 44100,
        "segment_frames": 8,
        "tensor_interface": "waveform",
        "output_semantics": "[1,4,2,8] drums/bass/other/vocals",
    }
    with pytest.raises(interfaces.InterfaceError):
        interfaces.measurement_spec(model, path, 8)


# --------------------------------------------------------------------------- #
# Synthetic qualification fixture
# --------------------------------------------------------------------------- #


def _bump_runtime_ids(manifest: dict) -> dict:
    mapping = {}
    for artifact in manifest["artifacts"]["runtimes"]:
        old = artifact["artifact_id"]
        new = old.replace("1.27.1", "1.31.0")
        mapping[old] = new
        artifact["artifact_id"] = new
        artifact["filename"] = artifact["filename"].replace("1.27.1", "1.31.0")
        artifact["archive_digest"] = _sha(new.encode())
        artifact["byte_size"] += 1
        artifact["extracted_file_digests"] = {
            "libonnxruntime.so": {"sha256": _sha(("lib-" + new).encode()), "size": 1000}
        }
        artifact["runtime"]["version"] = "v1.31.0"
    for artifact in manifest["artifacts"]["runtimes"]:
        replacement = (artifact.get("deprecation") or {}).get("replacement_artifact_id")
        if replacement in mapping:
            artifact["deprecation"]["replacement_artifact_id"] = mapping[replacement]
    for artifact in manifest["artifacts"]["models"]:
        artifact["model"]["compatible_runtime_ids"] = [
            mapping[value] for value in artifact["model"]["compatible_runtime_ids"]
        ]
        replacement = (artifact.get("deprecation") or {}).get("replacement_artifact_id")
        if replacement in mapping:
            artifact["deprecation"]["replacement_artifact_id"] = mapping[replacement]
    manifest["compatibility"] = [
        edge | {"runtime_artifact_id": mapping[edge["runtime_artifact_id"]]}
        for edge in manifest["compatibility"]
    ]
    return mapping


def _pinned_provenance() -> dict:
    lock = json.loads((ROOT / "ort" / "pinned-headers.json").read_text())
    entry = lock["headers"]["v1.27.1"]
    return {
        "source": "pinned",
        "upstream_commit_sha": entry["upstream"]["commit_sha"],
        "ort_api_version": entry["ort_api_version"],
        "files": [
            {"name": name, "sha256": value["sha256"], "size": value["size"]}
            for name, value in sorted(entry["files"].items())
        ],
    }


def _environment(target: str) -> dict:
    os_name, arch = native_host.TARGET_HOSTS[target]
    machine = "arm64" if (os_name == "darwin" and arch == "aarch64") else (
        "AMD64" if os_name == "windows" else arch
    )
    return {
        "os": "test",
        "os_name": os_name,
        "arch": arch,
        "machine": machine,
        "python": "3.11",
        "cpu_count": 1,
        "target_triple": target,
        "rosetta_translated": False,
    }


def _measurement(runtime: dict, shapes: list[list[int]], provider: str) -> dict:
    installed = sum(value["size"] for value in runtime["extracted_file_digests"].values())
    duration = FRAMES / 44100
    return {
        "cold_load_s": 0.1,
        "first_window_s": 0.1,
        "warm_median_s": 0.1,
        "warm_p95_s": 0.1,
        "warm_iters": 5,
        "peak_rss_kb": 100.0,
        "rss_delta_kb": 0.0,
        "requested_provider": provider,
        "assigned_provider": provider,
        "used_fallback": False,
        "provider_node_count": 10,
        "total_node_count": 10,
        "fallback_node_count": 0,
        "finite_output": True,
        "output_shapes": shapes,
        "installed_size": installed,
        "archive_bytes": runtime["byte_size"],
        "archive_sha256": runtime["archive_digest"],
        "header_provenance": _pinned_provenance(),
        "frames": FRAMES,
        "audio_duration_s": duration,
        "rtf_warm": 0.1 / duration,
    }


def _report_ref(url: str, payload: bytes, report_id: str) -> dict:
    return {
        "url": url,
        "size": len(payload),
        "sha256": _sha(payload),
        "report_id": report_id,
        "schema_version": json.loads(payload)["schema_version"],
    }


def _fixture(tmp_path: Path) -> tuple[dict, bytes, dict, dict[str, bytes]]:
    baseline = json.loads((ROOT / "catalog" / "releases" / f"{BASELINE_ID}.json").read_text())
    candidate = copy.deepcopy(baseline)
    candidate["release_id"] = CANDIDATE_ID
    candidate["generation"] = baseline["generation"] + 1
    _bump_runtime_ids(candidate)

    releases = tmp_path / "catalog" / "releases"
    channels = tmp_path / "catalog" / "channels"
    releases.mkdir(parents=True)
    channels.mkdir(parents=True)
    baseline_bytes = json.dumps(baseline, indent=2, sort_keys=True).encode() + b"\n"
    candidate_bytes = json.dumps(candidate, indent=2, sort_keys=True).encode() + b"\n"
    (releases / f"{BASELINE_ID}.json").write_bytes(baseline_bytes)
    (releases / f"{CANDIDATE_ID}.json").write_bytes(candidate_bytes)
    (channels / "stable.json").write_text(json.dumps({
        "release_id": BASELINE_ID,
        "release_manifest_sha256": _sha(baseline_bytes),
        "release_manifest_size": len(baseline_bytes),
    }))
    (channels / "candidate.json").write_text(json.dumps({
        "release_id": CANDIDATE_ID,
        "release_manifest_sha256": _sha(candidate_bytes),
        "release_manifest_size": len(candidate_bytes),
    }))

    baseline_runtimes = {a["artifact_id"]: a for a in baseline["artifacts"]["runtimes"]}
    candidate_models = {a["artifact_id"]: a for a in candidate["artifacts"]["models"]}
    candidate_runtimes = {a["artifact_id"]: a for a in candidate["artifacts"]["runtimes"]}

    groups: dict[tuple[str, str], list[dict]] = {}
    for edge in candidate["compatibility"]:
        if edge["status"] != "supported":
            continue
        groups.setdefault((edge["target_triple"], edge["execution_provider"]), []).append(edge)

    payloads: dict[str, bytes] = {}
    native_refs, app_refs = [], []
    for (target, provider), edges in sorted(groups.items()):
        native_edges, app_edges, skipped_edges = [], [], []
        for edge in edges:
            model_id = edge["model_artifact_id"]
            candidate_runtime = candidate_runtimes[edge["runtime_artifact_id"]]
            baseline_runtime = next(
                runtime for runtime in baseline_runtimes.values()
                if validator._family(runtime["artifact_id"]) ==
                validator._family(candidate_runtime["artifact_id"])
            )
            model = candidate_models[model_id]
            shapes = interfaces.parse_output_shapes(
                model["model"]["output_semantics"], model["model"]["tensor_interface"]
            )
            identity = {
                "model_artifact_id": model_id,
                "baseline_runtime_artifact_id": baseline_runtime["artifact_id"],
                "candidate_runtime_artifact_id": candidate_runtime["artifact_id"],
                "target_triple": target,
                "execution_provider": provider,
                "model_digest": model["archive_digest"],
                "model_size": model["byte_size"],
                "baseline_runtime_digest": baseline_runtime["archive_digest"],
                "baseline_runtime_size": baseline_runtime["byte_size"],
                "candidate_runtime_digest": candidate_runtime["archive_digest"],
                "candidate_runtime_size": candidate_runtime["byte_size"],
            }
            native_edges.append({
                "identity": identity,
                "model_files": [
                    {"path": name, "sha256": value["sha256"], "size": value["size"]}
                    for name, value in sorted(model["extracted_file_digests"].items())
                ],
                "expected_output_shapes": shapes,
                "baseline": _measurement(baseline_runtime, shapes, provider),
                "candidate": _measurement(candidate_runtime, shapes, provider),
                "comparison": {
                    "output_shapes_equal": True, "finite": True,
                    "output_max_abs": 0.0, "output_mse": 0.0, "passed": True,
                },
                "checks": [],
            })
            library = next(
                name for name in candidate_runtime["extracted_file_digests"]
                if name.rsplit("/", 1)[-1].lower() in validator.LIBRARY_NAMES
            )
            app_edge = {
                "identity": {
                    "model_artifact_id": model_id,
                    "runtime_artifact_id": candidate_runtime["artifact_id"],
                    "target_triple": target,
                    "execution_provider": provider,
                    "model_digest": model["archive_digest"],
                    "model_size": model["byte_size"],
                    "runtime_digest": candidate_runtime["archive_digest"],
                    "runtime_size": candidate_runtime["byte_size"],
                },
                "loaded_library": {
                    "name": library,
                    "sha256": candidate_runtime["extracted_file_digests"][library]["sha256"],
                },
                "provider": {
                    "requested": provider,
                    "assigned": provider,
                    "used_fallback": False,
                    "provider_node_count": 10,
                    "total_node_count": 10,
                    "fallback_node_count": 0,
                },
                "output_shapes": shapes,
                "finite_output": True,
                "audio_pipeline": {
                    "frames_processed": FRAMES,
                    "sample_rate": 44100,
                    "stems": 4,
                    "finite": True,
                },
                "checks": [],
            }
            if not model.get("deprecation", {}).get("deprecated") and \
                    model["model"]["tensor_interface"] in validator.APP_LOADABLE_TENSOR_INTERFACES:
                app_edges.append(app_edge)
            else:
                interface = model["model"]["tensor_interface"]
                deprecated = bool(model.get("deprecation", {}).get("deprecated"))
                skipped_edges.append({
                    "model_artifact_id": model_id,
                    "runtime_artifact_id": candidate_runtime["artifact_id"],
                    "target_triple": target,
                    "execution_provider": provider,
                    "tensor_interface": interface,
                    "deprecated": deprecated,
                    "scope": (
                        "deprecated"
                        if interface in validator.APP_LOADABLE_TENSOR_INTERFACES and deprecated
                        else "interface"
                    ),
                })

        slug = f"{target}-{provider}"
        native_report = {
            "schema_version": validator.NATIVE_SCHEMA_VERSION,
            "report_id": f"native-{CANDIDATE_ID}-{slug}",
            "created_at": "2099-01-01T00:00:00Z",
            "candidate_release_id": CANDIDATE_ID,
            "generation": candidate["generation"],
            "candidate_manifest_sha256": _sha(candidate_bytes),
            "baseline_release_id": BASELINE_ID,
            "baseline_manifest_sha256": _sha(baseline_bytes),
            "target_triple": target,
            "providers": [provider],
            "environment": _environment(target),
            "status": "passed",
            "edges": native_edges,
        }
        app_report = {
            "schema_version": validator.APP_SCHEMA_VERSION,
            "report_id": f"app-{CANDIDATE_ID}-{slug}",
            "created_at": "2099-01-01T00:00:00Z",
            "candidate_release_id": CANDIDATE_ID,
            "generation": candidate["generation"],
            "candidate_manifest_sha256": _sha(candidate_bytes),
            "openkara": {"version": "test", "commit": "a" * 40},
            "target_triple": target,
            "status": "passed",
            "edges": app_edges,
            "skipped_edges": skipped_edges,
        }
        for kind, report, refs in (
            ("native", native_report, native_refs),
            ("app", app_report, app_refs),
        ):
            payload = json.dumps(report, indent=2, sort_keys=True).encode() + b"\n"
            url = (
                "https://github.com/thedavidweng/openkara-models/releases/download/"
                f"qual-{CANDIDATE_ID}/{report['report_id']}.json"
            )
            payloads[url] = payload
            refs.append(_report_ref(url, payload, report["report_id"]))

    record = {
        "schema_version": "openkara.runtime-only-qualification/v1",
        "release_id": CANDIDATE_ID,
        "generation": candidate["generation"],
        "manifest": {"sha256": _sha(candidate_bytes), "size": len(candidate_bytes)},
        "baseline_release_id": BASELINE_ID,
        "baseline_manifest": {"sha256": _sha(baseline_bytes), "size": len(baseline_bytes)},
        "native_reports": native_refs,
        "app_reports": app_refs,
        "created_at": "2099-01-01T00:00:00Z",
    }
    return candidate, candidate_bytes, record, payloads


def _replace_payload(record: dict, payloads: dict[str, bytes], url: str, doc: dict) -> None:
    payload = json.dumps(doc, indent=2, sort_keys=True).encode() + b"\n"
    payloads[url] = payload
    for ref in record["native_reports"] + record["app_reports"]:
        if ref["url"] == url:
            ref["size"] = len(payload)
            ref["sha256"] = _sha(payload)
            ref["report_id"] = doc["report_id"]
            ref["schema_version"] = doc["schema_version"]


def _qualify(tmp_path: Path, mutate=None) -> list[str]:
    candidate, candidate_bytes, record, payloads = _fixture(tmp_path)
    if mutate is not None:
        mutate(candidate, record, payloads)
    return validator.qualify_runtime_only(
        candidate, candidate_bytes, record, lambda url: payloads[url], root=tmp_path
    )


def test_synthetic_runtime_only_evidence_qualifies(tmp_path) -> None:
    assert _qualify(tmp_path) == []


def test_missing_native_edge_fails(tmp_path) -> None:
    def mutate(_candidate, record, payloads):
        url = next(url for url in payloads if "/native-" in url)
        doc = json.loads(payloads[url])
        doc["edges"].pop()
        _replace_payload(record, payloads, url, doc)

    assert any("native reports do not cover" in error for error in _qualify(tmp_path, mutate))


def test_app_fallback_cannot_qualify(tmp_path) -> None:
    def mutate(_candidate, record, payloads):
        url = next(url for url in payloads if "/app-" in url)
        doc = json.loads(payloads[url])
        doc["edges"][0]["provider"]["used_fallback"] = True
        _replace_payload(record, payloads, url, doc)

    errors = _qualify(tmp_path, mutate)
    assert any("fallback" in error for error in errors)


def test_app_report_must_explicitly_skip_non_applicable_edges(tmp_path) -> None:
    def mutate(_candidate, record, payloads):
        url = next(url for url in payloads if "/app-" in url)
        doc = json.loads(payloads[url])
        assert doc["skipped_edges"], "fixture must contain non-applicable waveform edges"
        doc["skipped_edges"].pop()
        _replace_payload(record, payloads, url, doc)

    errors = _qualify(tmp_path, mutate)
    assert any("do not explicitly skip" in error for error in errors)


def test_app_report_cannot_skip_app_applicable_edge(tmp_path) -> None:
    def mutate(_candidate, record, payloads):
        url = next(
            url for url in payloads
            if "/app-" in url and len(json.loads(payloads[url])["edges"]) > 1
        )
        doc = json.loads(payloads[url])
        edge = doc["edges"].pop()
        identity = edge["identity"]
        doc["skipped_edges"].append({
            "model_artifact_id": identity["model_artifact_id"],
            "runtime_artifact_id": identity["runtime_artifact_id"],
            "target_triple": identity["target_triple"],
            "execution_provider": identity["execution_provider"],
            "tensor_interface": "spectral-core",
            "deprecated": False,
            "scope": "interface",
        })
        _replace_payload(record, payloads, url, doc)

    errors = _qualify(tmp_path, mutate)
    assert any("app-applicable edge was skipped" in error for error in errors)


def test_app_report_cannot_measure_non_applicable_edge(tmp_path) -> None:
    def mutate(_candidate, record, payloads):
        url = next(
            url for url in payloads
            if "/app-" in url and json.loads(payloads[url])["skipped_edges"]
        )
        doc = json.loads(payloads[url])
        skipped = doc["skipped_edges"].pop()
        edge = copy.deepcopy(doc["edges"][0])
        edge["identity"].update({
            "model_artifact_id": skipped["model_artifact_id"],
            "runtime_artifact_id": skipped["runtime_artifact_id"],
            "target_triple": skipped["target_triple"],
            "execution_provider": skipped["execution_provider"],
        })
        doc["edges"].append(edge)
        _replace_payload(record, payloads, url, doc)

    errors = _qualify(tmp_path, mutate)
    assert any("app measured a non-applicable edge" in error for error in errors)


def test_app_report_requires_skipped_edges_field(tmp_path) -> None:
    def mutate(_candidate, record, payloads):
        url = next(url for url in payloads if "/app-" in url)
        doc = json.loads(payloads[url])
        doc.pop("skipped_edges")
        _replace_payload(record, payloads, url, doc)

    errors = _qualify(tmp_path, mutate)
    assert any("skipped_edges" in error for error in errors)


def test_manifest_binding_mismatch_fails(tmp_path) -> None:
    def mutate(_candidate, record, _payloads):
        record["manifest"]["sha256"] = "f" * 64

    assert any("candidate manifest bytes" in error for error in _qualify(tmp_path, mutate))


def test_changed_model_bytes_cannot_use_runtime_only(tmp_path) -> None:
    def mutate(candidate, _record, _payloads):
        candidate["artifacts"]["models"][0]["archive_digest"] = "f" * 64

    assert any("archive bytes differ" in error for error in _qualify(tmp_path, mutate))


def test_committed_pin_matches_provenance_verification() -> None:
    import runtime_headers

    provenance = _pinned_provenance()
    artifact = {"runtime": {"ort_c_api_level": "27"}, "extracted_file_digests": {}}
    assert runtime_headers.verify_provenance(provenance, artifact) == []

    tampered = copy.deepcopy(provenance)
    tampered["files"][0]["sha256"] = "f" * 64
    assert runtime_headers.verify_provenance(tampered, artifact)


def test_packaged_provenance_must_match_artifact_header_digests() -> None:
    import runtime_headers

    provenance = {
        "source": "packaged",
        "upstream_commit_sha": "a" * 40,
        "ort_api_version": 27,
        "files": [
            {"name": "onnxruntime_c_api.h", "sha256": "b" * 64, "size": 10},
        ],
    }
    artifact = {
        "runtime": {"ort_c_api_level": "27"},
        "extracted_file_digests": {
            "onnxruntime_c_api.h": {"sha256": "b" * 64, "size": 10},
        },
    }
    assert runtime_headers.verify_provenance(provenance, artifact) == []
    artifact["extracted_file_digests"]["onnxruntime_c_api.h"]["sha256"] = "c" * 64
    assert runtime_headers.verify_provenance(provenance, artifact)


# --------------------------------------------------------------------------- #
# Regression tests for review findings
# --------------------------------------------------------------------------- #


def _channel_pointer(release_id: str, digest: str, size: int, generation: int,
                     previous_release_id: str | None) -> dict:
    repo = "thedavidweng/openkara-models"
    return {
        "schema_version": "openkara.catalog/channel-v1",
        "channel": "stable",
        "generation": generation,
        "release_id": release_id,
        "release_manifest_url": (
            f"https://github.com/{repo}/releases/download/"
            f"infra-{release_id}/release-manifest.json"
        ),
        "release_manifest_sha256": digest,
        "release_manifest_size": size,
        "updated_at": "2099-01-01T00:00:00Z",
        "previous_release_id": previous_release_id,
    }


def _remote_ok(pointer: dict, manifest_bytes: bytes) -> list[str]:
    return []


def _promoted_pointers(record: dict) -> tuple[dict, dict]:
    baseline = json.loads((ROOT / "catalog" / "releases" / f"{BASELINE_ID}.json").read_text())
    base = _channel_pointer(
        BASELINE_ID, record["baseline_manifest"]["sha256"],
        record["baseline_manifest"]["size"], baseline["generation"], None,
    )
    head = _channel_pointer(
        CANDIDATE_ID, record["manifest"]["sha256"], record["manifest"]["size"],
        record["generation"], BASELINE_ID,
    )
    return base, head


def test_record_remains_valid_after_promotion(tmp_path) -> None:
    candidate, candidate_bytes, record, payloads = _fixture(tmp_path)
    download = lambda url: payloads[url]  # noqa: E731

    assert validator.qualify_runtime_only(
        candidate, candidate_bytes, record, download, root=tmp_path,
    ) == []

    base, head = _promoted_pointers(record)
    (tmp_path / "catalog" / "channels" / "stable.json").write_text(json.dumps(head))
    assert validator.qualify_runtime_only(
        candidate, candidate_bytes, record, download, root=tmp_path,
    ), "current-pointer validation must fail after promotion"
    assert validator.qualify_runtime_only(
        candidate, candidate_bytes, record, download, root=tmp_path,
        require_current_pointers=False, expected_baseline=base,
    ) == []


def test_stable_pointer_transition_requires_evidence(tmp_path) -> None:
    import verify_stable_pointer_change as guard

    candidate, candidate_bytes, record, payloads = _fixture(tmp_path)
    base, head = _promoted_pointers(record)
    (tmp_path / "catalog" / "channels" / "stable.json").write_text(json.dumps(head))
    download = lambda url: payloads[url]  # noqa: E731

    unauthorised = guard.verify_transition(
        base, head, candidate, candidate_bytes, None, download, root=tmp_path,
        remote_manifest_check=_remote_ok,
    )
    assert unauthorised, "direct stable pointer advance without evidence must fail"

    assert guard.verify_transition(
        base, head, candidate, candidate_bytes, record, download, root=tmp_path,
        remote_manifest_check=_remote_ok,
    ) == []

    wrong_lineage = dict(head, previous_release_id="2020-01-01-001")
    assert guard.verify_transition(
        base, wrong_lineage, candidate, candidate_bytes, record, download, root=tmp_path,
        remote_manifest_check=_remote_ok,
    ), "pointer lineage must match the promoted-from stable release"


def test_accelerator_explicit_cpu_partition_is_allowed(tmp_path) -> None:
    def mutate(_candidate, record, payloads):
        changed = False
        for url in [item for item in payloads if "/native-" in item]:
            doc = json.loads(payloads[url])
            for edge in doc["edges"]:
                if edge["identity"]["execution_provider"] == "cpu":
                    continue
                for label in ("baseline", "candidate"):
                    edge[label]["provider_node_count"] = 7
                    edge[label]["fallback_node_count"] = 3
                    edge[label]["total_node_count"] = 10
                changed = True
            if changed:
                _replace_payload(record, payloads, url, doc)
        assert changed, "fixture has no accelerated native edge"

    assert _qualify(tmp_path, mutate) == [], "an explicit, accounted CPU partition must qualify"


def test_accelerator_partition_accounting_mismatch_fails(tmp_path) -> None:
    def mutate(_candidate, record, payloads):
        for url in [item for item in payloads if "/native-" in item]:
            doc = json.loads(payloads[url])
            edge = next(
                (item for item in doc["edges"]
                 if item["identity"]["execution_provider"] != "cpu"),
                None,
            )
            if edge is not None:
                edge["candidate"]["provider_node_count"] = 7
                edge["candidate"]["fallback_node_count"] = 5
                edge["candidate"]["total_node_count"] = 10
                _replace_payload(record, payloads, url, doc)
                return
        raise AssertionError("fixture has no accelerated native edge")

    errors = _qualify(tmp_path, mutate)
    assert any("partition accounting mismatch" in error for error in errors), errors


def test_app_accelerator_partition_accounting_mismatch_fails(tmp_path) -> None:
    def mutate(_candidate, record, payloads):
        for url in [item for item in payloads if "/app-" in item]:
            doc = json.loads(payloads[url])
            edge = next(
                (item for item in doc["edges"]
                 if item["identity"]["execution_provider"] != "cpu"),
                None,
            )
            if edge is not None:
                edge["provider"]["provider_node_count"] = 7
                edge["provider"]["fallback_node_count"] = 5
                edge["provider"]["total_node_count"] = 10
                _replace_payload(record, payloads, url, doc)
                return
        raise AssertionError("fixture has no accelerated app edge")

    errors = _qualify(tmp_path, mutate)
    assert any("partition accounting" in error for error in errors), errors


def test_candidate_cannot_drop_a_stable_supported_edge(tmp_path) -> None:
    def mutate(candidate, _record, _payloads):
        from collections import Counter

        pairs = Counter(
            (edge["model_artifact_id"], edge["runtime_artifact_id"])
            for edge in candidate["compatibility"] if edge["status"] == "supported"
        )
        model_id, runtime_id = next(pair for pair, count in pairs.items() if count == 1)
        candidate["compatibility"] = [
            edge for edge in candidate["compatibility"]
            if not (edge["model_artifact_id"] == model_id
                    and edge["runtime_artifact_id"] == runtime_id)
        ]
        for model in candidate["artifacts"]["models"]:
            if model["artifact_id"] == model_id:
                model["model"]["compatible_runtime_ids"].remove(runtime_id)
        for runtime in candidate["artifacts"]["runtimes"]:
            if runtime["artifact_id"] == runtime_id:
                runtime["runtime"]["supported_model_artifact_ids"].remove(model_id)

    errors = _qualify(tmp_path, mutate)
    assert any("drops a previously supported stable edge" in error for error in errors), errors


def test_reports_must_live_in_the_catalog_repository(tmp_path) -> None:
    def mutate(_candidate, record, _payloads):
        record["native_reports"][0]["url"] = (
            "https://github.com/attacker/elsewhere/releases/download/tag/report.json"
        )

    assert any("outside" in error for error in _qualify(tmp_path, mutate))


def test_producer_edge_plan_reads_manifest_runtime_ids() -> None:
    import run_runtime_only_comparison as producer

    manifest = json.loads((ROOT / "catalog" / "releases" / f"{BASELINE_ID}.json").read_text())
    plan = producer._edge_plan(manifest, "aarch64-apple-darwin", ["cpu", "coreml"])
    assert plan
    assert {edge["execution_provider"] for edge in plan} == {"cpu", "coreml"}
    assert all("runtime_artifact_id" in edge for edge in plan)
    with pytest.raises(ValueError):
        producer._edge_plan(manifest, "aarch64-apple-darwin", ["cpu"])


def test_native_report_environment_has_no_hostname_field() -> None:
    schema = json.loads((ROOT / "catalog" / "schema-runtime-only-v1.json").read_text())
    environment = schema["$defs"]["native_report"]["properties"]["environment"]
    assert "host" not in environment["properties"]
    assert "host" not in environment["required"]


def test_native_host_rejects_rosetta_translation() -> None:
    with pytest.raises(native_host.HostMismatch, match="Rosetta"):
        native_host.build_environment(
            "x86_64-apple-darwin", "Darwin", "x86_64", True,
            platform_label="test", python_version="3.11", cpu_count=1,
        )


def test_native_host_maps_apple_silicon_and_rejects_arch_mismatch() -> None:
    environment = native_host.build_environment(
        "aarch64-apple-darwin", "Darwin", "arm64", False,
        platform_label="test", python_version="3.11", cpu_count=1,
    )
    assert environment["os_name"] == "darwin"
    assert environment["arch"] == "aarch64"
    assert environment["rosetta_translated"] is False
    with pytest.raises(native_host.HostMismatch, match="requires aarch64"):
        native_host.build_environment(
            "aarch64-apple-darwin", "Darwin", "x86_64", True,
            platform_label="test", python_version="3.11", cpu_count=1,
        )
    with pytest.raises(native_host.HostMismatch, match="requires linux"):
        native_host.build_environment(
            "x86_64-unknown-linux-gnu", "Darwin", "x86_64", False,
            platform_label="test", python_version="3.11", cpu_count=1,
        )


def test_validator_rejects_rosetta_or_wrong_arch_environment(tmp_path) -> None:
    def mutate(_candidate, record, payloads):
        for url in [item for item in payloads if "/native-" in item]:
            doc = json.loads(payloads[url])
            if doc["target_triple"].startswith(("aarch64-apple", "x86_64-apple")):
                doc["environment"]["rosetta_translated"] = True
                _replace_payload(record, payloads, url, doc)
                return
        raise AssertionError("fixture has no macOS native report")

    errors = _qualify(tmp_path, mutate)
    assert any("Rosetta" in error for error in errors), errors


def test_validator_rejects_environment_target_mismatch(tmp_path) -> None:
    def mutate(_candidate, record, payloads):
        url = next(item for item in payloads if "/native-" in item)
        doc = json.loads(payloads[url])
        doc["environment"]["target_triple"] = "x86_64-unknown-linux-gnu"
        _replace_payload(record, payloads, url, doc)

    errors = _qualify(tmp_path, mutate)
    assert any("target_triple" in error for error in errors), errors


def test_stable_pointer_guard_checks_generation_and_url(tmp_path) -> None:
    import verify_stable_pointer_change as guard

    candidate, candidate_bytes, record, payloads = _fixture(tmp_path)
    base, head = _promoted_pointers(record)
    (tmp_path / "catalog" / "channels" / "stable.json").write_text(json.dumps(head))
    download = lambda url: payloads[url]  # noqa: E731

    bad_generation = dict(head, generation=head["generation"] + 1)
    errors = guard.verify_transition(
        base, bad_generation, candidate, candidate_bytes, record, download, root=tmp_path,
        remote_manifest_check=_remote_ok,
    )
    assert any("generation" in error for error in errors), errors

    bad_url = dict(head, release_manifest_url="https://example.invalid/release-manifest.json")
    errors = guard.verify_transition(
        base, bad_url, candidate, candidate_bytes, record, download, root=tmp_path,
        remote_manifest_check=_remote_ok,
    )
    assert any("release_manifest_url" in error for error in errors), errors


def test_candidate_cannot_change_the_trusted_catalog_repository(tmp_path) -> None:
    def mutate(candidate, record, _payloads):
        candidate["producer"]["repo"] = "attacker/elsewhere"
        for ref in [*record["native_reports"], *record["app_reports"]]:
            ref["url"] = ref["url"].replace(
                "thedavidweng/openkara-models", "attacker/elsewhere",
            )

    errors = _qualify(tmp_path, mutate)
    assert any("trusted stable repository" in error for error in errors), errors


def test_remote_manifest_asset_must_match_local_bytes() -> None:
    import hashlib
    import verify_stable_pointer_change as guard

    manifest_bytes = b'{"release_id": "2099-01-01-001"}\n'
    pointer = {
        "release_manifest_url": (
            "https://github.com/thedavidweng/openkara-models/releases/download/"
            "infra-2099-01-01-001/release-manifest.json"
        )
    }
    size = len(manifest_bytes)
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    good_assets = lambda repo, tag: [{"name": "release-manifest.json", "size": size}]  # noqa: E731
    good_download = lambda repo, tag, name: digest  # noqa: E731
    good_state = lambda repo, tag: {  # noqa: E731
        "isDraft": False, "publishedAt": "2099-01-01T00:00:00Z",
    }

    assert guard.remote_manifest_errors(
        pointer, manifest_bytes,
        release_assets=good_assets, download_sha256=good_download,
        release_state=good_state,
    ) == []

    assert guard.remote_manifest_errors(
        pointer, manifest_bytes,
        release_assets=good_assets, download_sha256=good_download,
        release_state=lambda repo, tag: {"isDraft": True, "publishedAt": None},
    ), "draft release must fail"
    assert guard.remote_manifest_errors(
        pointer, manifest_bytes,
        release_assets=good_assets, download_sha256=good_download,
        release_state=lambda repo, tag: {"isDraft": False, "publishedAt": None},
    ), "unpublished release must fail"
    assert guard.remote_manifest_errors(
        pointer, manifest_bytes,
        release_assets=lambda repo, tag: [], download_sha256=good_download,
        release_state=good_state,
    ), "missing remote asset must fail"
    assert guard.remote_manifest_errors(
        pointer, manifest_bytes,
        release_assets=lambda repo, tag: [{"name": "release-manifest.json", "size": size + 1}],
        download_sha256=good_download, release_state=good_state,
    ), "remote size mismatch must fail"
    assert guard.remote_manifest_errors(
        pointer, manifest_bytes,
        release_assets=good_assets, download_sha256=lambda repo, tag, name: "f" * 64,
        release_state=good_state,
    ), "remote digest mismatch must fail"


def test_transition_propagates_remote_manifest_failure(tmp_path) -> None:
    import verify_stable_pointer_change as guard

    candidate, candidate_bytes, record, payloads = _fixture(tmp_path)
    base, head = _promoted_pointers(record)
    (tmp_path / "catalog" / "channels" / "stable.json").write_text(json.dumps(head))
    errors = guard.verify_transition(
        base, head, candidate, candidate_bytes, record,
        lambda url: payloads[url], root=tmp_path,
        remote_manifest_check=lambda pointer, data: ["remote pointer asset mismatch"],
    )
    assert "remote pointer asset mismatch" in errors
