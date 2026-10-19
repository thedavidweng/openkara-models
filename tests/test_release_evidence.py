"""Small synthetic contract tests; these are not measured release qualification."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import validate_release_evidence as evidence
import enforce_quality_gates as gates
from test_quality_gates import _quality_report, _runtime_report, _frozen_baseline
from test_native_smoke import _valid_smoke_report


@pytest.fixture
def case(tmp_path, monkeypatch):
    quality, performance = _quality_report(), _runtime_report()
    for report in (quality, performance):
        report["tier"] = "release"
        report["results"][0].update(fixture_id="real-test", tier="release", category="male-vocals")
    identity = dict(zip(evidence.EDGE_FIELDS, ("model", "runtime", "target", "cpu")))
    identity.update(model_digest="a" * 64, model_size=100, runtime_digest="b" * 64, runtime_size=200)
    performance["runtime_archive"].update(sha256="b" * 64, size=200)
    quality["runtime_archive"] = copy.deepcopy(performance["runtime_archive"])
    for report in (quality, performance):
        report.update(measurement_backend="native-c-api", target="target", provider="cpu")
        report["model_digest"] = {"sha256": "a" * 64, "size": 100}
    for row in performance["results"]:
        row["rtf_warm"] = row["warm_median_s"] / row["audio_duration_s"]
        row["end_to_end_s"] = row["rtf_first"] * row["audio_duration_s"]
    corpus = {"schema_version": "openkara.corpus-manifest/v1", "fixtures": [{"fixture_id": "real-test", "kind": "real-audio", "tier": "release",
                           "category": "male-vocals", "license": "CC0-1.0", "channels": 2, "sample_rate": 44100, "frames": 343980, "generator": {"type": "fetch", "sha256": "c" * 64, "size": 1000, "url": "https://example.test/references.npz"}, "reference_stems": {"format": "npz-float32-v1", "order": ["drums", "bass", "other", "vocals"]}}]}
    (tmp_path / "quality").mkdir()
    (tmp_path / "quality/corpus-manifest.json").write_text(json.dumps(corpus))
    monkeypatch.setattr(evidence, "ROOT", tmp_path)
    raw_corpus = (tmp_path / "quality/corpus-manifest.json").read_bytes()
    separation = {m: 1.0 for m in evidence.SEPARATION_DIRECTIONS}
    separation_row = separation | {"fixture_id": "real-test", "reference_bundle": {"sha256": "c" * 64, "size": 1000}}
    quality["separation"] = [separation_row]
    baseline = _frozen_baseline()
    baseline.update(artifact_id="default", target="target", provider="cpu")
    baseline["baseline_metrics"] = gates._aggregate_quality(quality) | gates._aggregate_runtime(performance) | separation
    baseline["baseline_metrics"]["installed_size"] *= 1.1
    blobs = {}
    def ref(doc):
        raw = json.dumps(doc).encode()
        url = f"https://github.com/test/test/releases/download/evidence/{doc['report_id']}.json"
        blobs[url] = raw
        return {"url": url, "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw),
                "report_id": doc["report_id"], "schema_version": doc["schema_version"]}
    baseline_refs = {}
    for kind in ("quality", "runtime"):
        doc = {"report_id": f"baseline-{kind}", "schema_version": "test-baseline/v1",
               "identity": {k: baseline[k] for k in ("model_digest", "runtime_id", "runtime_digest", "runtime_size", "target", "provider")} | {"model_size": baseline["artifact_size_bytes"]},
               "corpus_sha256": hashlib.sha256(raw_corpus).hexdigest(),
               "benchmark_environment": baseline["benchmark_environment"],
               "metrics": baseline["baseline_metrics"]}
        baseline_refs[f"baseline_{kind}_report"] = ref(doc)
        baseline[f"baseline_{kind}_report_id"] = doc["report_id"]
        baseline[f"baseline_{kind}_report_digest"] = baseline_refs[f"baseline_{kind}_report"]["sha256"]
    baseline_path = tmp_path / "baselines.json"
    baseline_path.write_text(json.dumps({"default_artifact_id": "default", "stable_baselines": [baseline],
        "required_fields_for_freeze": json.loads(gates.BASELINES_PATH.read_text())["required_fields_for_freeze"]}))
    monkeypatch.setattr(gates, "BASELINES_PATH", baseline_path)
    native = _valid_smoke_report("cpu") | {"target": "target",
             "model_digest": {"sha256": "a" * 64, "size": 100},
             "runtime_archive": {"sha256": "b" * 64, "size": 200}}
    payloads = {
        "quality": {"report": quality, "corpus_sha256": hashlib.sha256(raw_corpus).hexdigest(),
                    "separation": [separation_row], **baseline_refs},
        "performance": {"report": performance, "measurement_backend": "native-c-api", "benchmark_environment": baseline["benchmark_environment"]},
        "compatibility": {"native_smoke": native, "status": "passed", "quality_report_id": "quality", "performance_report_id": "performance"},
    }
    docs = {k: {"schema_version": f"openkara.release-{k}/v1", "report_id": k, "release_id": "test", "generation": 1,
                "edges": [{"identity": copy.deepcopy(identity), **row}]} for k, row in payloads.items()}
    manifest = {"release_id": "test", "generation": 1,
        "compatibility": [dict(zip(evidence.EDGE_FIELDS, ("model", "runtime", "target", "cpu"))) | {"status": "supported"}],
        "artifacts": {"models": [{"artifact_id": "model", "archive_digest": "a" * 64, "byte_size": 100}],
                      "runtimes": [{"artifact_id": "runtime", "archive_digest": "b" * 64, "byte_size": 200}]}}
    def run():
        manifest["gates"] = {f"{k}_report": ref(d) for k, d in docs.items()}
        return evidence.qualify_release(manifest, blobs.__getitem__)
    return manifest, docs, blobs, run


def test_complete_synthetic_contract(case):
    assert case[3]() == []


@pytest.mark.parametrize("kind", ["quality", "performance", "compatibility"])
@pytest.mark.parametrize("fault", ["missing", "duplicate", "extra", "digest", "size", "release", "generation"])
def test_exact_edge_and_candidate_identity(case, kind, fault):
    manifest, docs, _, run = case
    doc = docs[kind]
    if fault == "missing": doc["edges"] = []
    elif fault == "duplicate": doc["edges"] *= 2
    elif fault == "extra": doc["edges"].append(copy.deepcopy(doc["edges"][0])); doc["edges"][-1]["identity"]["execution_provider"] = "directml"
    elif fault == "digest": doc["edges"][0]["identity"]["model_digest"] = "f" * 64
    elif fault == "size": doc["edges"][0]["identity"]["runtime_size"] += 1
    elif fault == "release": doc["release_id"] = "other"
    elif fault == "generation": doc["generation"] += 1
    assert run()


@pytest.mark.parametrize("fault", ["skip", "wheel", "corpus", "fixture", "duplicate", "noninferior", "nan", "budget", "baseline", "native-bytes", "archive", "status", "report-id", "pareto"])
def test_rejects_unqualified_measurements(case, fault):
    _, docs, _, run = case
    q, p, c = (docs[k]["edges"][0] for k in ("quality", "performance", "compatibility"))
    if fault == "skip": c["native_smoke"]["session_creation"] = "skipped"
    elif fault == "wheel": p["measurement_backend"] = "python-wheel"
    elif fault == "corpus": q["corpus_sha256"] = "f" * 64
    elif fault == "fixture": q["report"]["results"] = []
    elif fault == "duplicate": p["report"]["results"] *= 2
    elif fault == "noninferior": q["separation"][0]["si_sdr_db"] = 0.9
    elif fault == "nan": q["separation"][0]["leakage_db"] = float("nan")
    elif fault == "budget": p["report"]["results"][0]["cold_load_s"] = 100
    elif fault == "baseline": q["baseline_runtime_report"]["sha256"] = "f" * 64
    elif fault == "native-bytes": c["native_smoke"]["model_digest"]["sha256"] = "f" * 64
    elif fault == "archive": p["report"]["runtime_archive"]["sha256"] = "f" * 64
    elif fault == "status": c["status"] = "not_run"
    elif fault == "report-id": c["quality_report_id"] = "unrelated"
    elif fault == "pareto": p["report"]["runtime_archive"]["installed_size"] *= 1.1
    assert run()


@pytest.mark.parametrize("fault", ["digest", "size", "identity"])
def test_report_bytes_are_verified(case, fault):
    manifest, _, blobs, run = case
    assert run() == []
    ref = manifest["gates"]["quality_report"]
    if fault == "digest": blobs[ref["url"]] += b" "
    elif fault == "size": ref["size"] += 1
    elif fault == "identity": ref["report_id"] = "unrelated"
    assert evidence.qualify_release(manifest, blobs.__getitem__)


@pytest.mark.parametrize("field", ["model_digest", "model_size", "runtime_id", "runtime_digest", "runtime_size", "target", "provider"])
def test_baseline_bytes_and_identity_cannot_be_substituted(case, field):
    manifest, docs, blobs, run = case
    assert run() == []
    q = docs["quality"]["edges"][0]
    ref = q["baseline_quality_report"]
    old = json.loads(blobs[ref["url"]])
    old["identity"][field] = "wrong"
    raw = json.dumps(old).encode()
    blobs[ref["url"]] = raw
    ref.update(sha256=hashlib.sha256(raw).hexdigest(), size=len(raw))
    baseline = json.loads(gates.BASELINES_PATH.read_text())
    baseline["stable_baselines"][0]["baseline_quality_report_digest"] = ref["sha256"]
    gates.BASELINES_PATH.write_text(json.dumps(baseline))
    assert any("baseline model/runtime/target/provider mismatch" in e for e in run())


@pytest.mark.parametrize("field,message", [
    ("corpus_sha256", "baseline corpus differs"),
    ("benchmark_environment", "baseline report environment mismatch"),
    ("metrics", "baseline report metrics differ"),
])
def test_baseline_measurements_are_bound(case, field, message):
    _, docs, blobs, run = case
    ref = docs["quality"]["edges"][0]["baseline_quality_report"]
    old = json.loads(blobs[ref["url"]])
    old[field] = "wrong"
    raw = json.dumps(old).encode()
    blobs[ref["url"]] = raw
    ref.update(sha256=hashlib.sha256(raw).hexdigest(), size=len(raw))
    baseline = json.loads(gates.BASELINES_PATH.read_text())
    baseline["stable_baselines"][0]["baseline_quality_report_digest"] = ref["sha256"]
    gates.BASELINES_PATH.write_text(json.dumps(baseline))
    assert any(message in e for e in run())


def test_default_remeasurement_need_not_beat_itself(case):
    manifest, docs, _, run = case
    baseline = json.loads(gates.BASELINES_PATH.read_text())
    baseline["default_artifact_id"] = "model"
    baseline["stable_baselines"][0]["artifact_id"] = "model"
    gates.BASELINES_PATH.write_text(json.dumps(baseline))
    docs["performance"]["edges"][0]["report"]["runtime_archive"]["installed_size"] *= 1.1
    assert run() == []


def test_missing_no_fallback_declaration_fails(case):
    del case[1]["compatibility"]["edges"][0]["native_smoke"]["used_fallback"]
    assert any("strict native provider qualification failed" in e for e in case[3]())


def test_rejects_environment_mismatch(case):
    _, docs, _, run = case
    docs["performance"]["edges"][0]["benchmark_environment"] = {"os": "different"}
    assert any("environment mismatch" in e for e in run())


def test_missing_evidence_stops_before_download():
    assert evidence.qualify_release({"gates": {}}, lambda _: pytest.fail("must not download"))


@pytest.mark.parametrize('kind', ['quality', 'performance'])
@pytest.mark.parametrize('fault', ['backend', 'provider', 'target', 'archive', 'model'])
def test_producer_report_must_match_native_edge(case, kind, fault):
    _, docs, _, run = case
    raw = docs[kind]['edges'][0]['report']
    if fault == 'backend': raw['measurement_backend'] = 'python-wheel-conversion-only'
    elif fault == 'archive': raw['runtime_archive']['size'] += 1
    elif fault == 'model': raw['model_digest']['sha256'] = 'f' * 64
    else: raw[fault] = 'other'
    expected = {'archive': 'producer runtime archive mismatch', 'model': 'producer model bytes mismatch'}.get(
        fault, 'native producer backend/target/provider identity mismatch')
    assert any(expected in error for error in run())
