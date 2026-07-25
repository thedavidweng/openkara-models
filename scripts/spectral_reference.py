#!/usr/bin/env python3
"""Pure-numpy reference implementation of the OpenKara spectral contract v1.

These functions define the EXACT waveform<->spectral transform semantics of
the OpenKara spectral contract v1, extracted per issue #23 PR 1 so an
independent native implementation (OpenKara's Rust FFT frontend/backend) can
be validated against golden vectors without PyTorch. The earlier waveform
HTDemucs ONNX graphs implemented these same transforms in-graph with dense
conv1d/conv_transpose1d DFT matrices; that conv-DFT export path was removed
in issue #23 PR 4.

Contract: docs/spectral-core-contract.md (openkara.spectral-contract/v1).

Semantics reproduced here, in order:

``spec(x)``      — Demucs outer reflect padding, normalized centered STFT
                   (periodic Hann, one-sided), Nyquist-bin drop, frame crop,
                   layout permutation.
``magnitude(z)`` — complex-as-channels view (real/imag interleaved into the
                   channel axis) fed to the neural core.
``mask(m)``      — inverse view: core output back to explicit real/imag.
``ispec(z, n)``  — Nyquist re-append, frame re-pad, one-sided inverse DFT,
                   windowed overlap-add with clamped envelope division,
                   center + outer crops.

Self-check::

    python scripts/spectral_reference.py --self-check
"""

from __future__ import annotations

import sys

import numpy as np

N_FFT = 4096
HOP = 1024
N_BINS = N_FFT // 2 + 1          # 2049 one-sided bins before the Nyquist drop
CONTRACT_FREQS = N_FFT // 2      # 2048 bins carried by the contract tensor
ENVELOPE_CLAMP = 1e-8
OUTER_PAD = HOP // 2 * 3         # 1536


def periodic_hann(n_fft: int = N_FFT) -> np.ndarray:
    """torch.hann_window default: periodic Hann, sin^2(pi*n/N)."""
    n = np.arange(n_fft, dtype=np.float64)
    return (np.sin(np.pi * n / n_fft) ** 2).astype(np.float64)


def _dft_filters(n_fft: int = N_FFT) -> tuple[np.ndarray, np.ndarray]:
    """Forward one-sided DFT filters: cos and -sin (e^{-i 2 pi k n / N})."""
    k = np.arange(n_fft // 2 + 1, dtype=np.float64)[:, None]
    n = np.arange(n_fft, dtype=np.float64)[None, :]
    ang = 2.0 * np.pi * k * n / n_fft
    return np.cos(ang), -np.sin(ang)


def _idft_filters(n_fft: int = N_FFT) -> tuple[np.ndarray, np.ndarray]:
    """Inverse one-sided DFT filters with hermitian doubling on interior bins."""
    k = np.arange(n_fft // 2 + 1, dtype=np.float64)[:, None]
    n = np.arange(n_fft, dtype=np.float64)[None, :]
    ang = 2.0 * np.pi * k * n / n_fft
    scale = np.ones((n_fft // 2 + 1, 1))
    scale[1:-1] = 2.0
    return np.cos(ang) * scale, np.sin(ang) * scale


def _reflect_pad_1d(x: np.ndarray, left: int, right: int) -> np.ndarray:
    """Reflect padding on the last axis (torch 'reflect' semantics)."""
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(left, right)], mode="reflect")


def stft(x: np.ndarray) -> np.ndarray:
    """Centered, normalized, one-sided STFT of ``[..., samples]``.

    Matches torch.stft(center=True, pad_mode='reflect', normalized=True,
    window=hann_window(n_fft)) as implemented by OnnxSTFT: returns
    ``[..., N_BINS, frames, 2]`` (last axis = real, imag) where
    frames = 1 + samples // HOP.
    """
    window = periodic_hann()
    cosf, sinf = _dft_filters()
    norm = 1.0 / np.sqrt(N_FFT)
    xp = _reflect_pad_1d(x.astype(np.float64), N_FFT // 2, N_FFT // 2)
    n_frames = 1 + (xp.shape[-1] - N_FFT) // HOP
    frames = np.lib.stride_tricks.sliding_window_view(xp, N_FFT, axis=-1)[..., ::HOP, :]
    frames = frames[..., :n_frames, :] * window
    real = frames @ cosf.T * norm
    imag = frames @ sinf.T * norm
    return np.stack([np.moveaxis(real, -1, -2), np.moveaxis(imag, -1, -2)], axis=-1)


def istft(spec: np.ndarray, length: int) -> np.ndarray:
    """Inverse of :func:`stft` for ``[..., N_BINS, frames, 2]`` input.

    One-sided inverse DFT (hermitian doubling on interior bins), windowed
    overlap-add, envelope division clamped at ENVELOPE_CLAMP, center crop of
    N_FFT/2, then crop to ``length`` samples.
    """
    window = periodic_hann()
    cosf, sinf = _idft_filters()
    norm = 1.0 / np.sqrt(N_FFT)
    real = np.moveaxis(spec[..., 0], -1, -2).astype(np.float64)  # [..., frames, bins]
    imag = np.moveaxis(spec[..., 1], -1, -2).astype(np.float64)
    # Per-frame time signal: (cos^T real - sin^T imag), windowed + normalized.
    frames_td = (real @ cosf - imag @ sinf) * (window * norm)
    n_frames = frames_td.shape[-2]
    out_len = N_FFT + HOP * (n_frames - 1)
    lead = frames_td.shape[:-2]
    signal = np.zeros(lead + (out_len,), dtype=np.float64)
    envelope = np.zeros(out_len, dtype=np.float64)
    wsq = window * window
    for t in range(n_frames):
        signal[..., t * HOP: t * HOP + N_FFT] += frames_td[..., t, :]
        envelope[t * HOP: t * HOP + N_FFT] += wsq
    signal = signal / np.maximum(envelope, ENVELOPE_CLAMP)
    signal = signal[..., N_FFT // 2:]
    return signal[..., :length]


def spec(x: np.ndarray) -> np.ndarray:
    """Demucs ``_spec``: ``[B, C, samples]`` -> ``[B, C, 2, CONTRACT_FREQS, le]``.

    le = ceil(samples / HOP). Outer reflect padding of OUTER_PAD on the left
    and OUTER_PAD + le*HOP - samples on the right, STFT, Nyquist-bin drop,
    frame crop [2 : 2+le], then layout (batch, channel, real/imag, freq,
    frame).
    """
    if x.ndim != 3:
        raise ValueError(f"expected [B, C, samples], got {x.shape}")
    samples = x.shape[-1]
    le = -(-samples // HOP)  # ceil
    pad_r = OUTER_PAD + le * HOP - samples
    xp = _reflect_pad_1d(x.astype(np.float64), OUTER_PAD, pad_r)
    z = stft(xp)                       # [B, C, N_BINS, frames, 2]
    z = z[:, :, :-1, :, :]             # drop Nyquist bin
    z = z[:, :, :, 2: 2 + le, :]       # frame crop
    return np.transpose(z, (0, 1, 4, 2, 3))  # [B, C, 2, F, T]


def ispec(z: np.ndarray, length: int) -> np.ndarray:
    """Demucs ``_ispec``: ``[B, S, C, 2, CONTRACT_FREQS, T]`` -> ``[B, S, C, length]``.

    Re-appends a zero Nyquist bin, re-pads two frames on each side, runs the
    one-sided ISTFT to le = HOP*ceil(length/HOP) + 2*OUTER_PAD samples, and
    crops [OUTER_PAD : OUTER_PAD+length].
    """
    if z.ndim != 6:
        raise ValueError(f"expected [B, S, C, 2, F, T], got {z.shape}")
    z = np.pad(z, [(0, 0), (0, 0), (0, 0), (0, 0), (0, 1), (0, 0)])  # Nyquist
    z = np.pad(z, [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (2, 2)])  # frames
    le = HOP * (-(-length // HOP)) + 2 * OUTER_PAD
    b, s, c, _, freqs, frames = z.shape
    zi = np.transpose(z, (0, 1, 2, 4, 5, 3))         # [B,S,C,F,T,2]
    x = istft(zi, length=le)                          # [B,S,C,le]
    return x[..., OUTER_PAD: OUTER_PAD + length]


def magnitude(z: np.ndarray) -> np.ndarray:
    """Demucs ``_magnitude`` (cac=True): ``[B, C, 2, F, T]`` -> ``[B, C*2, F, T]``."""
    b, c, two, f, t = z.shape
    return z.reshape(b, c * two, f, t)


def mask(m: np.ndarray) -> np.ndarray:
    """Demucs ``_mask`` (cac=True): ``[B, S, C*2, F, T]`` -> ``[B, S, C, 2, F, T]``."""
    b, s, c2, f, t = m.shape
    return m.reshape(b, s, c2 // 2, 2, f, t)


# Contract facts verified by the self-check and documented in
# docs/spectral-core-contract.md:
#  - the Nyquist bin (22050 Hz) is discarded by the forward transform and
#    reconstructed as zero, so only band-limited reconstruction is exact;
#  - the first/last EDGE_INVALID_SAMPLES of a reconstructed window lose
#    overlap-add contributions from the cropped frames and are not
#    reconstruction-exact (measured: interior error 1e-10, transition band
#    up to ~3e-6 within one FFT window of each edge). The application's
#    segment overlap must cover this region.
EDGE_INVALID_SAMPLES = N_FFT  # 4096 per side


def _band_limited_noise(shape: tuple[int, ...], seed: int = 7) -> np.ndarray:
    """White noise with the top 5% of the band removed (no Nyquist energy)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape) * 0.2
    xf = np.fft.rfft(x, axis=-1)
    cut = int(xf.shape[-1] * 0.95)
    xf[..., cut:] = 0.0
    return np.fft.irfft(xf, n=shape[-1], axis=-1).astype(np.float32)


def _self_check() -> int:
    x = _band_limited_noise((1, 2, 44100))

    z = spec(x)
    assert z.shape == (1, 2, 2, CONTRACT_FREQS, 44), z.shape

    m = magnitude(z)
    assert m.shape == (1, 4, CONTRACT_FREQS, 44)
    z2 = mask(m[:, np.newaxis].repeat(4, axis=1))
    assert z2.shape == (1, 4, 2, 2, CONTRACT_FREQS, 44)

    # Round trip through the identity "mask": interior reconstruction must be
    # exact for band-limited content; edges are documented as invalid.
    zr = z[:, np.newaxis]  # [B, 1, C, 2, F, T] (single source)
    y = ispec(zr, 44100)[:, 0]
    err = np.abs(y - x)
    interior = float(err[..., EDGE_INVALID_SAMPLES:-EDGE_INVALID_SAMPLES].max())
    print(f"interior round-trip max abs error: {interior:.3e}")
    # Broadband content leaks energy into the discarded Nyquist bin through
    # the Hann window (~-80 dB), bounding broadband round-trips near 1e-4.
    # The tone check below proves exactness when no leakage is present.
    assert interior < 1e-4, interior
    assert float(err.max()) < 1.0  # edge error bounded, never explosive

    tone = np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)[None, None, :].astype(np.float32)
    tone2 = np.repeat(tone, 2, axis=1)
    yt = ispec(spec(tone2)[:, np.newaxis], 44100)[:, 0]
    err_t = float(np.abs(yt - tone2)[..., EDGE_INVALID_SAMPLES:-EDGE_INVALID_SAMPLES].max())
    print(f"tone interior round-trip max abs error: {err_t:.3e}")
    assert err_t < 1e-8, err_t

    print("OK: spectral reference self-check passed")
    return 0


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        raise SystemExit(_self_check())
    print(__doc__)
