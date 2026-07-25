"""Known-answer tests for the separation-quality metrics (issue #21, step 3).

Every metric is verified against synthetic cases with analytically known
results, independent of any model output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import separation_metrics as m  # noqa: E402

RNG = np.random.default_rng(1234)
SR = 44100


def _tone(freq: float, seconds: float = 1.0, sr: int = SR, channels: int = 2) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    x = np.sin(2 * np.pi * freq * t)
    return np.tile(x, (channels, 1))


# ---------------------------------------------------------------------------
# SI-SDR
# ---------------------------------------------------------------------------

def test_si_sdr_identical_is_very_high() -> None:
    ref = RNG.standard_normal((2, SR))
    assert m.si_sdr(ref, ref) > 100.0


def test_si_sdr_scale_invariant() -> None:
    """Scaling the estimate must not change SI-SDR (its defining property)."""
    ref = RNG.standard_normal((2, SR))
    noisy = ref + 0.05 * RNG.standard_normal((2, SR))
    a = m.si_sdr(ref, noisy)
    b = m.si_sdr(ref, 7.3 * noisy)
    assert a == pytest.approx(b, abs=1e-9)


def test_si_sdr_known_noise_ratio() -> None:
    """For zero-mean orthogonal noise, SI-SDR = 10*log10(P_ref / P_noise).

    Adding noise at 1/100 the reference power gives 20 dB.
    """
    n = SR * 4
    ref = RNG.standard_normal((1, n))
    ref -= ref.mean()
    noise = RNG.standard_normal((1, n))
    noise -= noise.mean()
    # Orthogonalize the noise against the reference.
    noise -= (np.sum(noise * ref) / np.sum(ref ** 2)) * ref
    noise *= np.sqrt(np.sum(ref ** 2) / (100.0 * np.sum(noise ** 2)))
    got = m.si_sdr(ref, ref + noise)
    assert got == pytest.approx(20.0, abs=0.01)


def test_si_sdr_anti_signal_is_scale_invariant_alias() -> None:
    """-ref is a scaled copy: SI-SDR treats it as perfect."""
    ref = RNG.standard_normal((2, SR))
    assert m.si_sdr(ref, -ref) > 100.0


def test_si_sdr_silent_reference_raises() -> None:
    with pytest.raises(ValueError):
        m.si_sdr(np.zeros((2, SR)), RNG.standard_normal((2, SR)))


def test_si_sdr_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        m.si_sdr(np.zeros((2, 100)) + 1.0, np.zeros((2, 200)))


# ---------------------------------------------------------------------------
# SDR
# ---------------------------------------------------------------------------

def test_sdr_known_residual_ratio() -> None:
    """SDR = 10*log10(P_ref / P_residual): residual at 1/1000 power = 30 dB."""
    n = SR * 4
    ref = RNG.standard_normal((1, n))
    residual = RNG.standard_normal((1, n))
    residual *= np.sqrt(np.sum(ref ** 2) / (1000.0 * np.sum(residual ** 2)))
    got = m.sdr(ref, ref - residual)
    assert got == pytest.approx(30.0, abs=0.01)


def test_sdr_not_scale_invariant() -> None:
    """Unlike SI-SDR, halving the estimate costs SDR ~6 dB of residual."""
    ref = RNG.standard_normal((2, SR))
    assert m.sdr(ref, 0.5 * ref) == pytest.approx(
        10.0 * np.log10(np.sum(ref ** 2) / np.sum((0.5 * ref) ** 2)), abs=1e-6
    )


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------

def test_leakage_none_for_orthogonal_estimate() -> None:
    """Disjoint sinusoids are (near-)orthogonal: essentially no leakage."""
    vocals = _tone(440.0)
    accompaniment = _tone(100.0)
    assert m.leakage_db(vocals, accompaniment) < -60.0


def test_leakage_known_mixture_level() -> None:
    """est = other + 0.1*source (orthogonal, equal power) leaks at
    10*log10(0.01*P/(P + 0.01*P)) ≈ -20.04 dB."""
    n = SR * 4
    source = RNG.standard_normal((1, n))
    other = RNG.standard_normal((1, n))
    other -= (np.sum(other * source) / np.sum(source ** 2)) * source
    other *= np.sqrt(np.sum(source ** 2) / np.sum(other ** 2))
    est = other + 0.1 * source
    expected = 10.0 * np.log10(0.01 / 1.01)
    assert m.leakage_db(source, est) == pytest.approx(expected, abs=0.05)


def test_leakage_full_source_is_zero_db() -> None:
    source = RNG.standard_normal((2, SR))
    assert m.leakage_db(source, source.copy()) == pytest.approx(0.0, abs=1e-6)


def test_leakage_silent_estimate_is_neg_inf() -> None:
    source = RNG.standard_normal((2, SR))
    assert m.leakage_db(source, np.zeros_like(source)) == float("-inf")


# ---------------------------------------------------------------------------
# Multi-resolution STFT loss
# ---------------------------------------------------------------------------

def test_mrstft_identical_is_zero() -> None:
    ref = RNG.standard_normal((2, SR))
    assert m.multi_resolution_stft_loss(ref, ref) < 1e-9


def test_mrstft_orders_by_divergence() -> None:
    """More noise must produce a strictly larger loss."""
    ref = _tone(440.0)
    noise = RNG.standard_normal(ref.shape)
    small = m.multi_resolution_stft_loss(ref, ref + 0.01 * noise)
    large = m.multi_resolution_stft_loss(ref, ref + 0.2 * noise)
    assert 0.0 < small < large


def test_mrstft_detects_missing_band() -> None:
    """Dropping one component of a two-tone signal is a visible loss."""
    two_tone = _tone(440.0) + _tone(2000.0)
    one_tone = _tone(440.0)
    assert m.multi_resolution_stft_loss(two_tone, one_tone) > 0.5


# ---------------------------------------------------------------------------
# Transient error
# ---------------------------------------------------------------------------

def _impulse_train(n: int, period: int, channels: int = 1) -> np.ndarray:
    x = np.zeros((channels, n))
    x[:, ::period] = 1.0
    return x


def test_transient_error_zero_when_identical() -> None:
    ref = _impulse_train(SR, 4410)
    assert m.transient_error(ref, ref) == pytest.approx(0.0, abs=1e-15)


def test_transient_error_penalizes_smoothed_attacks() -> None:
    """Removing the impulses entirely is a worse transient error than
    adding low-level noise everywhere."""
    ref = _impulse_train(SR, 4410)
    smoothed = np.zeros_like(ref)  # all attacks lost
    noisy = ref + 0.001 * RNG.standard_normal(ref.shape)
    assert m.transient_error(ref, smoothed) > m.transient_error(ref, noisy) * 10.0


# ---------------------------------------------------------------------------
# Low-frequency error
# ---------------------------------------------------------------------------

def test_lf_error_ignores_high_frequency_residual() -> None:
    """An error confined to 5 kHz leaves the <=150 Hz band untouched."""
    ref = _tone(60.0)
    est = ref + 0.5 * _tone(5000.0)
    assert m.low_frequency_error_db(ref, est) < -80.0


def test_lf_error_full_low_band_loss_is_zero_db() -> None:
    """Dropping the whole 60 Hz reference leaves a low-band residual equal
    to the low-band reference energy: 0 dB."""
    ref = _tone(60.0)
    est = np.zeros_like(ref)
    assert m.low_frequency_error_db(ref, est) == pytest.approx(0.0, abs=0.01)


def test_lf_error_no_low_energy_raises() -> None:
    ref = _tone(5000.0)
    with pytest.raises(ValueError):
        m.low_frequency_error_db(ref, ref)


# ---------------------------------------------------------------------------
# Chunk-boundary error
# ---------------------------------------------------------------------------

def test_boundary_error_zero_for_identical() -> None:
    x = RNG.standard_normal((2, SR))
    r = m.chunk_boundary_error(x, x.copy(), boundaries=[SR // 2], window=1024)
    assert r["boundary_max_abs_error"] == 0.0
    assert r["boundary_mse"] == 0.0


def test_boundary_error_detects_seam_discontinuity() -> None:
    """A step discontinuity exactly at the boundary shows up in the
    boundary window and not elsewhere."""
    n = SR
    b = n // 2
    full = RNG.standard_normal((1, n))
    chunked = full.copy()
    chunked[:, b:b + 64] += 0.5  # bad OLA seam
    r = m.chunk_boundary_error(full, chunked, boundaries=[b], window=1024)
    assert r["boundary_max_abs_error"] == pytest.approx(0.5, abs=1e-9)
    assert r["boundary_mse"] > 100.0 * r["elsewhere_mse"] + 1e-12
    assert r["elsewhere_mse"] == 0.0


def test_boundary_error_requires_proper_subset() -> None:
    x = np.ones((1, 100))
    with pytest.raises(ValueError):
        m.chunk_boundary_error(x, x, boundaries=[50], window=1000)  # covers all


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
