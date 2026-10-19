"""Tiny generated arrays exercise ingestion, never qualify a real-audio release."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from corpus_audio import STEM_NAMES, load_fixture, separation_result


def bundle(tmp_path, mutation=None):
    rng = np.random.default_rng(7)
    stems = rng.normal(0, 0.1, (4, 2, 12000)).astype(np.float32)
    data = dict(mix=stems.sum(axis=0), stems=stems, sample_rate=np.array(44100), stem_names=np.array(STEM_NAMES))
    if mutation == "shape": data["stems"] = stems[..., :-1]
    elif mutation == "order": data["stem_names"] = np.array(STEM_NAMES[::-1])
    elif mutation == "rate": data["sample_rate"] = np.array(48000)
    elif mutation == "nonfinite": data["mix"][0, 0] = np.nan
    elif mutation == "sum": data["mix"] *= 2
    elif mutation == "dtype": data["stems"] = stems.astype(np.float64)
    path = tmp_path / "test.npz"
    np.savez(path, **data)
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    path.rename(tmp_path / f"{sha}.npz")
    fixture = dict(fixture_id="generated-contract-test", kind="real-audio", tier="release", category="noise",
                   channels=2, frames=12000, sample_rate=44100, license="CC0-1.0",
                   generator=dict(type="fetch", url="https://example.test/generated-test.npz", sha256=sha, size=len(raw)),
                   reference_stems=dict(format="npz-float32-v1", order=STEM_NAMES))
    return fixture


def test_load_and_measure_reference_bundle(tmp_path):
    fixture = bundle(tmp_path)
    mix, stems = load_fixture(fixture, tmp_path)
    assert mix.shape == (2, 12000)
    estimated = stems[None] + np.random.default_rng(1).normal(0, 0.01, (1, 4, 2, 12000))
    result = separation_result(fixture, stems, estimated, 4096)
    assert result["reference_bundle"]["sha256"] == fixture["generator"]["sha256"]
    assert len(result) == 9
    assert all(np.isfinite(v) for v in result.values() if isinstance(v, float))


@pytest.mark.parametrize("mutation", ["shape", "order", "rate", "nonfinite", "sum", "dtype"])
def test_bad_bundle_rejected(tmp_path, mutation):
    fixture = bundle(tmp_path, mutation)
    with pytest.raises(ValueError):
        load_fixture(fixture, tmp_path)


@pytest.mark.parametrize("field", ["sha256", "size"])
def test_bundle_integrity(tmp_path, field):
    fixture = bundle(tmp_path)
    path = tmp_path / f"{fixture['generator']['sha256']}.npz"
    if field == "sha256": path.write_bytes(b"corrupt" + path.read_bytes()[7:])
    else: fixture["generator"]["size"] += 1
    with pytest.raises(ValueError, match="digest/size"):
        load_fixture(fixture, tmp_path)


def test_no_download_or_fake_chunk_metric(tmp_path):
    fixture = bundle(tmp_path)
    with pytest.raises(ValueError, match="corpus-dir"):
        load_fixture(fixture)
    _, stems = load_fixture(fixture, tmp_path)
    with pytest.raises(ValueError, match="actual inference chunk"):
        separation_result(fixture, stems, stems[None], 343980)


def test_quality_producer_uses_real_reference(tmp_path, monkeypatch):
    import run_quality_suite as suite
    fixture = bundle(tmp_path)
    fixture["expected_output_shape"] = [1, 4, 2, 12000]
    _, stems = load_fixture(fixture, tmp_path)
    estimate = stems[None] + np.float32(0.01)
    monkeypatch.setattr(suite, "SEGMENT_FRAMES", 4096)
    monkeypatch.setattr(suite, "_compute_pytorch_reference", lambda *args: estimate)
    monkeypatch.setattr(suite, "_compute_onnx_output", lambda *args: estimate)
    result = suite.run_fixture(fixture, "mock-no-model", "mock-no-onnx", tmp_path)
    assert result["mse"] == 0
    assert result["separation"]["sdr_db"] > 10
    assert result["separation"]["reference_bundle"]["size"] == fixture["generator"]["size"]


def test_boundary_metric_known_answer_uses_actual_segment(tmp_path):
    fixture = bundle(tmp_path)
    _, stems = load_fixture(fixture, tmp_path)
    estimate = stems[None].astype(np.float64).copy()
    # Actual 4096-sample segment => hop 2048, radius 512, five disjoint regions.
    near = np.zeros(fixture['frames'], dtype=bool)
    for boundary in range(2048, fixture['frames'], 2048):
        near[boundary - 512:boundary + 512] = True
    estimate[..., near] += 0.01
    result = separation_result(fixture, stems, estimate, 4096)
    assert result['chunk_boundary_error'] == pytest.approx(0.0001, rel=1e-10)
