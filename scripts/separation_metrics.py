#!/usr/bin/env python3
"""Separation-quality metric implementations (issue #21, step 3).

Pure-numpy reference implementations of the release-gate separation
metrics, independently tested with synthetic known-answer cases in
``tests/test_separation_metrics.py``:

- ``si_sdr``                        — scale-invariant SDR per stem (dB)
- ``sdr``                           — plain SDR (dB)
- ``leakage_db``                    — source leaking into another estimate (dB)
- ``multi_resolution_stft_loss``    — spectral convergence + log-magnitude L1
- ``transient_error``               — error concentrated at reference transients
- ``low_frequency_error_db``        — residual energy below a cutoff (dB)
- ``chunk_boundary_error``          — reconstruction error near chunk boundaries

All functions take float arrays shaped ``[channels, samples]`` (mono
``[samples]`` is accepted and broadcast to one channel) and are
deterministic. They are consumed by the release tier of
``run_quality_suite.py`` once the licensed reference-stem corpus
(issue #21, step 2) provides ground-truth stems; the PR tier has no
ground truth and keeps its correctness metrics.

Self-check::

    python scripts/separation_metrics.py --self-check
"""

from __future__ import annotations

import sys

import numpy as np

EPS = 1e-10


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return x[np.newaxis, :]
    if x.ndim != 2:
        raise ValueError(f"expected [channels, samples] or [samples], got shape {x.shape}")
    return x


def _check_pair(reference: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref, est = _as_2d(reference), _as_2d(estimate)
    if ref.shape != est.shape:
        raise ValueError(f"shape mismatch: reference {ref.shape} vs estimate {est.shape}")
    return ref, est


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-invariant signal-to-distortion ratio in dB (Le Roux et al. 2019).

    The estimate is projected onto the reference; gain differences do not
    change the score. Identical (or exactly scaled) signals score +inf.
    """
    ref, est = _check_pair(reference, estimate)
    ref = ref - ref.mean(axis=1, keepdims=True)
    est = est - est.mean(axis=1, keepdims=True)
    ref_energy = np.sum(ref ** 2)
    if ref_energy < EPS:
        raise ValueError("reference is silent; SI-SDR is undefined")
    alpha = np.sum(est * ref) / ref_energy
    target = alpha * ref
    noise = est - target
    return float(10.0 * np.log10((np.sum(target ** 2) + EPS) / (np.sum(noise ** 2) + EPS)))


def sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Plain (scale-variant) signal-to-distortion ratio in dB."""
    ref, est = _check_pair(reference, estimate)
    ref_energy = np.sum(ref ** 2)
    if ref_energy < EPS:
        raise ValueError("reference is silent; SDR is undefined")
    noise_energy = np.sum((ref - est) ** 2)
    return float(10.0 * np.log10((ref_energy + EPS) / (noise_energy + EPS)))


def leakage_db(source_reference: np.ndarray, other_estimate: np.ndarray) -> float:
    """How much of ``source_reference`` leaks into ``other_estimate``, in dB.

    Projects the estimate onto the reference source and reports the
    projected energy relative to the estimate's total energy. An estimate
    containing none of the source scores -inf-like (very negative); an
    estimate that IS the source scores 0 dB.

    Example: ``leakage_db(ref_vocals, est_accompaniment)`` measures vocals
    leaking into the accompaniment estimate.
    """
    ref, est = _check_pair(source_reference, other_estimate)
    ref_energy = np.sum(ref ** 2)
    est_energy = np.sum(est ** 2)
    if ref_energy < EPS:
        raise ValueError("source reference is silent; leakage is undefined")
    if est_energy < EPS:
        return float("-inf")
    alpha = np.sum(est * ref) / ref_energy
    leaked_energy = (alpha ** 2) * ref_energy
    return float(10.0 * np.log10((leaked_energy + EPS) / (est_energy + EPS)))


def _stft_mag(x: np.ndarray, fft_size: int, hop: int) -> np.ndarray:
    """Magnitude STFT of a mono signal (Hann window, no padding)."""
    if len(x) < fft_size:
        raise ValueError(f"signal shorter than fft_size {fft_size}")
    window = np.hanning(fft_size)
    n_frames = 1 + (len(x) - fft_size) // hop
    frames = np.lib.stride_tricks.sliding_window_view(x, fft_size)[::hop][:n_frames]
    return np.abs(np.fft.rfft(frames * window, axis=1))


def multi_resolution_stft_loss(
    reference: np.ndarray,
    estimate: np.ndarray,
    fft_sizes: tuple[int, ...] = (512, 1024, 2048),
) -> float:
    """Multi-resolution STFT loss: mean over resolutions of spectral
    convergence plus log-magnitude L1 (Yamamoto et al. 2020). 0 for
    identical signals; grows with spectral divergence.
    """
    ref, est = _check_pair(reference, estimate)
    losses = []
    for fft_size in fft_sizes:
        hop = fft_size // 4
        for ch in range(ref.shape[0]):
            r = _stft_mag(ref[ch], fft_size, hop)
            e = _stft_mag(est[ch], fft_size, hop)
            sc = np.linalg.norm(r - e) / (np.linalg.norm(r) + EPS)
            log_l1 = np.mean(np.abs(np.log(r + EPS) - np.log(e + EPS)))
            losses.append(sc + log_l1)
    return float(np.mean(losses))


def transient_error(
    reference: np.ndarray,
    estimate: np.ndarray,
    frame: int = 512,
    top_fraction: float = 0.05,
) -> float:
    """MSE restricted to the reference's most transient frames.

    Transience is ranked by the positive first difference of per-frame RMS
    energy of the reference (attack detection). The error is the MSE of
    (reference - estimate) over the ``top_fraction`` most transient frames.
    """
    ref, est = _check_pair(reference, estimate)
    n = (ref.shape[1] // frame) * frame
    if n == 0:
        raise ValueError("signal shorter than one frame")
    ref_frames = ref[:, :n].reshape(ref.shape[0], -1, frame)
    est_frames = est[:, :n].reshape(est.shape[0], -1, frame)
    rms = np.sqrt(np.mean(ref_frames ** 2, axis=(0, 2)) + EPS)
    flux = np.diff(rms, prepend=rms[0])
    flux = np.maximum(flux, 0.0)
    n_frames = len(flux)
    k = max(1, int(np.ceil(n_frames * top_fraction)))
    transient_idx = np.argsort(flux)[-k:]
    diff = ref_frames[:, transient_idx, :] - est_frames[:, transient_idx, :]
    return float(np.mean(diff ** 2))


def low_frequency_error_db(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int = 44100,
    cutoff_hz: float = 150.0,
) -> float:
    """Residual energy below ``cutoff_hz`` relative to the reference's
    low-frequency energy, in dB. Near -inf when the low band is perfectly
    reconstructed; 0 dB when the low-band residual equals the low-band
    reference energy.
    """
    ref, est = _check_pair(reference, estimate)
    n = ref.shape[1]
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    low = freqs <= cutoff_hz
    ref_spec = np.fft.rfft(ref, axis=1)
    res_spec = np.fft.rfft(ref - est, axis=1)
    ref_low = np.sum(np.abs(ref_spec[:, low]) ** 2)
    res_low = np.sum(np.abs(res_spec[:, low]) ** 2)
    if ref_low < EPS:
        raise ValueError(f"reference has no energy below {cutoff_hz} Hz")
    return float(10.0 * np.log10((res_low + EPS) / (ref_low + EPS)))


def chunk_boundary_error(
    full_output: np.ndarray,
    chunked_output: np.ndarray,
    boundaries: list[int],
    window: int = 2048,
) -> dict[str, float]:
    """Reconstruction error near chunk boundaries versus elsewhere.

    ``boundaries`` are sample offsets where chunks were stitched. Returns
    the max absolute error and MSE within ``window`` samples of any
    boundary, plus the MSE elsewhere for comparison. A correct OLA
    reconstruction keeps ``boundary_mse`` in line with ``elsewhere_mse``.
    """
    full, chunked = _check_pair(full_output, chunked_output)
    n = full.shape[1]
    near = np.zeros(n, dtype=bool)
    for b in boundaries:
        lo, hi = max(0, b - window), min(n, b + window)
        near[lo:hi] = True
    if not near.any() or near.all():
        raise ValueError("boundaries must select a proper subset of samples")
    diff = full - chunked
    return {
        "boundary_max_abs_error": float(np.max(np.abs(diff[:, near]))),
        "boundary_mse": float(np.mean(diff[:, near] ** 2)),
        "elsewhere_mse": float(np.mean(diff[:, ~near] ** 2)),
    }


def _self_check() -> int:
    """Cheap analytic sanity checks (full known-answer coverage lives in
    tests/test_separation_metrics.py)."""
    rng = np.random.default_rng(42)
    ref = rng.standard_normal((2, 44100))

    assert si_sdr(ref, ref * 3.0) > 100.0, "SI-SDR must be scale-invariant"
    assert abs(sdr(ref, ref)) > 100.0 or sdr(ref, ref) > 100.0
    noise = rng.standard_normal((2, 44100))
    noisy = ref + 0.1 * noise
    assert 15.0 < si_sdr(ref, noisy) < 25.0
    assert multi_resolution_stft_loss(ref, ref) < 1e-9
    print("OK: separation metrics self-check passed")
    return 0


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        raise SystemExit(_self_check())
    print(__doc__)
