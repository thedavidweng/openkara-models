"""Deterministic synthetic audio fixtures for the quality and performance gates.

Every fixture is generated from a fixed seed and has no external dependencies.
The output is a numpy array of shape ``(channels, frames)`` at the given sample
rate. Fixtures are used by ``run_quality_suite.py`` and ``run_runtime_benchmarks.py``
to provide reproducible inputs that exercise specific audio characteristics.

All functions accept ``channels``, ``sample_rate``, and ``frames`` as keyword
arguments (passed by the fixture runner from the corpus manifest) plus any
fixture-specific ``params``.
"""

from __future__ import annotations

import numpy as np


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def silence(*, channels: int = 2, sample_rate: int = 44100, frames: int = 343980,
            **params) -> np.ndarray:
    return np.zeros((channels, frames), dtype=np.float32)


def white_noise(*, channels: int = 2, sample_rate: int = 44100, frames: int = 343980,
                seed: int = 42, **params) -> np.ndarray:
    rng = _rng(seed)
    return rng.standard_normal((channels, frames)).astype(np.float32) * 0.1


def sine_sweep(*, channels: int = 2, sample_rate: int = 44100, frames: int = 343980,
               f0: float = 20, f1: float = 20000, seed: int = 42, **params) -> np.ndarray:
    """Logarithmic sine sweep from f0 to f1 Hz over the full duration."""
    t = np.arange(frames, dtype=np.float32) / sample_rate
    # Logarithmic frequency progression.
    log_f0 = np.log(f0)
    log_f1 = np.log(f1)
    freqs = np.exp(log_f0 + (log_f1 - log_f0) * t / t[-1])
    # Instantaneous phase.
    phase = 2 * np.pi * np.cumsum(freqs) / sample_rate
    mono = np.sin(phase).astype(np.float32) * 0.3
    return np.stack([mono, mono], axis=0)[:channels]


def multitone(*, channels: int = 2, sample_rate: int = 44100, frames: int = 343980,
              freqs: list[float] | None = None, seed: int = 42, **params) -> np.ndarray:
    """Sum of sine waves at the given frequencies."""
    if freqs is None:
        freqs = [100, 440, 1000, 4000, 12000]
    t = np.arange(frames, dtype=np.float32) / sample_rate
    mono = np.zeros(frames, dtype=np.float32)
    for f in freqs:
        mono += np.sin(2 * np.pi * f * t).astype(np.float32) * 0.1
    return np.stack([mono, mono], axis=0)[:channels]


def impulse(*, channels: int = 2, sample_rate: int = 44100, frames: int = 343980,
            position: int = 44100, **params) -> np.ndarray:
    """Single impulse at the given sample position."""
    out = np.zeros((channels, frames), dtype=np.float32)
    out[:, position] = 1.0
    return out


def chunk_boundary(*, channels: int = 2, sample_rate: int = 44100, frames: int = 687960,
                   chunk_frames: int = 343980, seed: int = 42, **params) -> np.ndarray:
    """Two chunks with a discontinuity at the boundary.

    First chunk: low-frequency sine. Second chunk: high-frequency sine.
    The discontinuity tests overlap-add reconstruction.
    """
    t1 = np.arange(chunk_frames, dtype=np.float32) / sample_rate
    t2 = np.arange(frames - chunk_frames, dtype=np.float32) / sample_rate
    chunk1 = np.sin(2 * np.pi * 100 * t1).astype(np.float32) * 0.3
    chunk2 = np.sin(2 * np.pi * 5000 * t2).astype(np.float32) * 0.3
    mono = np.concatenate([chunk1, chunk2])
    return np.stack([mono, mono], axis=0)[:channels]


def clipped(*, channels: int = 2, sample_rate: int = 44100, frames: int = 343980,
            seed: int = 42, clip_threshold: float = 0.8, **params) -> np.ndarray:
    """Clipped white noise."""
    rng = _rng(seed)
    sig = rng.standard_normal((channels, frames)).astype(np.float32) * 0.5
    return np.clip(sig, -clip_threshold, clip_threshold)


def low_level(*, channels: int = 2, sample_rate: int = 44100, frames: int = 343980,
              seed: int = 42, amplitude: float = 0.01, **params) -> np.ndarray:
    """Low-amplitude noise."""
    rng = _rng(seed)
    return rng.standard_normal((channels, frames)).astype(np.float32) * amplitude


def channel_imbalance(*, channels: int = 2, sample_rate: int = 44100, frames: int = 343980,
                      seed: int = 42, left_gain: float = 1.0, right_gain: float = 0.3,
                      **params) -> np.ndarray:
    """Stereo signal with gain imbalance between channels."""
    rng = _rng(seed)
    mono = rng.standard_normal(frames).astype(np.float32) * 0.2
    return np.stack([mono * left_gain, mono * right_gain], axis=0)[:channels]


def full_song(*, channels: int = 2, sample_rate: int = 44100, frames: int = 3439800,
              seed: int = 42, **params) -> np.ndarray:
    """Synthetic full song: 10 segments with varying spectral content."""
    rng = _rng(seed)
    segment_frames = frames // 10
    segments: list[np.ndarray] = []
    freqs = [80, 200, 440, 880, 1600, 3000, 6000, 10000, 15000, 20000]
    for i in range(10):
        t = np.arange(segment_frames, dtype=np.float32) / sample_rate
        f = freqs[i]
        sine = np.sin(2 * np.pi * f * t).astype(np.float32) * 0.2
        noise = rng.standard_normal(segment_frames).astype(np.float32) * 0.05
        segments.append(sine + noise)
    mono = np.concatenate(segments)
    # Pad to exact frames.
    if len(mono) < frames:
        mono = np.pad(mono, (0, frames - len(mono)))
    return np.stack([mono, mono], axis=0)[:channels]


FIXTURE_FUNCTIONS = {
    "silence": silence,
    "white_noise": white_noise,
    "sine_sweep": sine_sweep,
    "multitone": multitone,
    "impulse": impulse,
    "chunk_boundary": chunk_boundary,
    "clipped": clipped,
    "low_level": low_level,
    "channel_imbalance": channel_imbalance,
    "full_song": full_song,
}


def generate_fixture(fixture: dict) -> np.ndarray:
    """Generate a fixture from a corpus manifest entry."""
    gen = fixture["generator"]
    func = FIXTURE_FUNCTIONS[gen["function"]]
    return func(
        channels=fixture["channels"],
        sample_rate=fixture["sample_rate"],
        frames=fixture["frames"],
        **gen.get("params", {}),
    )
