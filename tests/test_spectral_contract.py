"""Tests for the spectral contract v1 reference implementation and golden
vectors (issue #23 PR 1)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import generate_spectral_golden_vectors as golden  # noqa: E402
import spectral_reference as sr  # noqa: E402


# ---------------------------------------------------------------------------
# Contract constants
# ---------------------------------------------------------------------------

def test_contract_constants() -> None:
    assert sr.N_FFT == 4096
    assert sr.HOP == 1024
    assert sr.CONTRACT_FREQS == 2048
    assert sr.OUTER_PAD == 1536
    assert sr.EDGE_INVALID_SAMPLES == 4096


def test_periodic_hann_is_torch_convention() -> None:
    """Periodic Hann: w[0] == 0, w[N/2] == 1, and w has no symmetric peak
    duplication (w[1] != w[N-1] would fail for the symmetric variant of
    even length; periodic satisfies w[k] == w[N-k] for k>=1)."""
    w = sr.periodic_hann()
    assert w[0] == 0.0
    assert w[sr.N_FFT // 2] == pytest.approx(1.0)
    assert w[1] == pytest.approx(w[sr.N_FFT - 1])
    # symmetric np.hanning(4096) differs: its max is not exactly at N/2
    assert w[sr.N_FFT // 2] > np.hanning(sr.N_FFT).max() - 1e-12


# ---------------------------------------------------------------------------
# Shapes and layout inverses
# ---------------------------------------------------------------------------

def test_spec_shape_hop_aligned_and_unaligned() -> None:
    z = sr.spec(np.zeros((1, 2, 10240), dtype=np.float32))
    assert z.shape == (1, 2, 2, 2048, 10)
    z = sr.spec(np.zeros((1, 2, 10000), dtype=np.float32))
    assert z.shape == (1, 2, 2, 2048, 10)  # ceil(10000/1024) = 10


def test_magnitude_mask_are_inverse_views() -> None:
    rng = np.random.default_rng(3)
    z = rng.standard_normal((1, 2, 2, 64, 7)).astype(np.float32)
    m = sr.magnitude(z)
    assert m.shape == (1, 4, 64, 7)
    # channel-major interleave: [L_re, L_im, R_re, R_im]
    np.testing.assert_array_equal(m[0, 0], z[0, 0, 0])
    np.testing.assert_array_equal(m[0, 1], z[0, 0, 1])
    np.testing.assert_array_equal(m[0, 2], z[0, 1, 0])
    core_out = np.stack([m] * 4, axis=1)  # [B, S, 4, F, T]
    back = sr.mask(core_out)
    assert back.shape == (1, 4, 2, 2, 64, 7)
    np.testing.assert_array_equal(back[0, 0], z[0])


# ---------------------------------------------------------------------------
# Known answers
# ---------------------------------------------------------------------------

def test_tone_concentrates_at_expected_bin() -> None:
    """A 440 Hz tone peaks at bin round(440*4096/44100) = 41."""
    t = np.arange(10240) / 44100
    x = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)[None, None].repeat(2, axis=1)
    z = sr.spec(x)
    mag = np.sqrt(z[0, 0, 0] ** 2 + z[0, 0, 1] ** 2)  # [F, T]
    assert int(np.argmax(mag[:, 5])) == 41


def test_silence_maps_to_zero_spectrum_and_back() -> None:
    x = np.zeros((1, 2, 10240), dtype=np.float32)
    z = sr.spec(x)
    assert float(np.abs(z).max()) == 0.0
    y = sr.ispec(z[:, np.newaxis], 10240)[:, 0]
    assert float(np.abs(y).max()) == 0.0


def test_interior_roundtrip_is_exact_for_bandlimited() -> None:
    x = sr._band_limited_noise((1, 2, 20480), seed=5)
    y = sr.ispec(sr.spec(x)[:, np.newaxis], 20480)[:, 0]
    err = np.abs(y - x)
    e = sr.EDGE_INVALID_SAMPLES
    assert float(err[..., e:-e].max()) < 1e-4   # Nyquist-leakage bound
    assert float(err.max()) < 1.0               # edges bounded


def test_ispec_linearity_supports_karaoke_summation() -> None:
    """ispec(a+b) == ispec(a) + ispec(b): the app may sum accompaniment
    spectra before a single ISTFT (contract's karaoke optimization)."""
    rng = np.random.default_rng(9)
    a = rng.standard_normal((1, 1, 2, 2, 2048, 6)) * 0.1
    b = rng.standard_normal((1, 1, 2, 2, 2048, 6)) * 0.1
    lhs = sr.ispec(a + b, 4096)
    rhs = sr.ispec(a, 4096) + sr.ispec(b, 4096)
    np.testing.assert_allclose(lhs, rhs, atol=1e-12)


def test_self_check_passes() -> None:
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "spectral_reference.py"), "--self-check"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr


# ---------------------------------------------------------------------------
# Golden vectors
# ---------------------------------------------------------------------------

def test_golden_manifest_is_fresh() -> None:
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "generate_spectral_golden_vectors.py"), "--verify"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_golden_generation_is_deterministic() -> None:
    a = golden.build_manifest(golden.build_vectors())
    b = golden.build_manifest(golden.build_vectors())
    assert a == b


def test_golden_manifest_covers_all_stages() -> None:
    manifest = golden.build_manifest(golden.build_vectors())
    assert manifest["contract_version"] == "openkara.spectral-contract/v1"
    for name, arrays in manifest["fixtures"].items():
        assert set(arrays) == {"input", "spectral", "magnitude", "roundtrip"}, name
        for meta in arrays.values():
            assert len(meta["sha256"]) == 64
            assert meta["dtype"] == "float32"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
