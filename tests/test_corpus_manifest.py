"""Tests for the quality corpus manifest and synthetic fixtures."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def test_synthetic_fixtures_silence() -> None:
    import synthetic_fixtures as sf
    out = sf.silence(channels=2, sample_rate=44100, frames=1024)
    assert out.shape == (2, 1024)
    assert out.dtype == np.float32
    assert np.all(out == 0)


def test_synthetic_fixtures_white_noise_deterministic() -> None:
    import synthetic_fixtures as sf
    a = sf.white_noise(channels=2, sample_rate=44100, frames=1024, seed=42)
    b = sf.white_noise(channels=2, sample_rate=44100, frames=1024, seed=42)
    c = sf.white_noise(channels=2, sample_rate=44100, frames=1024, seed=99)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_synthetic_fixtures_sine_sweep() -> None:
    import synthetic_fixtures as sf
    out = sf.sine_sweep(channels=2, sample_rate=44100, frames=44100, f0=20, f1=20000)
    assert out.shape == (2, 44100)
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))


def test_synthetic_fixtures_impulse() -> None:
    import synthetic_fixtures as sf
    out = sf.impulse(channels=2, sample_rate=44100, frames=1024, position=500)
    assert out.shape == (2, 1024)
    assert np.all(out[:, 500] == 1.0)
    assert np.sum(out) == 2.0  # one impulse per channel


def test_synthetic_fixtures_chunk_boundary() -> None:
    import synthetic_fixtures as sf
    out = sf.chunk_boundary(channels=2, sample_rate=44100, frames=2000, chunk_frames=1000)
    assert out.shape == (2, 2000)
    # Discontinuity at sample 1000.
    assert abs(out[0, 999] - out[0, 1000]) > 0.1


def test_synthetic_fixtures_channel_imbalance() -> None:
    import synthetic_fixtures as sf
    out = sf.channel_imbalance(channels=2, sample_rate=44100, frames=1024,
                               left_gain=1.0, right_gain=0.3)
    assert out.shape == (2, 1024)
    assert np.std(out[0]) > np.std(out[1])


def test_synthetic_fixtures_full_song() -> None:
    import synthetic_fixtures as sf
    out = sf.full_song(channels=2, sample_rate=44100, frames=3439800)
    assert out.shape == (2, 3439800)
    assert np.all(np.isfinite(out))


def test_generate_fixture_from_manifest() -> None:
    import synthetic_fixtures as sf
    fixture = {
        "fixture_id": "test",
        "kind": "synthetic",
        "tier": "pr",
        "category": "noise",
        "channels": 2,
        "sample_rate": 44100,
        "frames": 1024,
        "generator": {"type": "deterministic", "module": "synthetic_fixtures",
                      "function": "white_noise", "params": {"seed": 42}},
        "license": "generated",
    }
    out = sf.generate_fixture(fixture)
    assert out.shape == (2, 1024)


def test_validate_corpus_manifest_cli() -> None:
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "validate_corpus_manifest.py")],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_corpus_manifest_has_pr_and_release_fixtures() -> None:
    manifest = json.loads((ROOT / "quality" / "corpus-manifest.json").read_text())
    tiers = {f["tier"] for f in manifest["fixtures"]}
    assert "pr" in tiers
    assert "release" in tiers
    categories = {f["category"] for f in manifest["fixtures"]}
    # Issue #21 requires these categories.
    required = {"silence", "noise", "sine-sweep", "multitone", "impulse",
                "chunk-boundary", "clipping", "low-level", "channel-imbalance",
                "full-song"}
    assert required.issubset(categories), f"missing: {required - categories}"


def test_corpus_manifest_all_synthetic_are_deterministic() -> None:
    manifest = json.loads((ROOT / "quality" / "corpus-manifest.json").read_text())
    for f in manifest["fixtures"]:
        if f["kind"] == "synthetic":
            assert f["generator"]["type"] == "deterministic"
            assert "seed" in f["generator"].get("params", {}) or f["generator"]["function"] in (
                "silence", "impulse"
            )
