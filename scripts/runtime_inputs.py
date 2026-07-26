#!/usr/bin/env python3
"""Shared input synthesis for the native runtime benchmark/comparison harnesses.

The runtime benchmark (``run_runtime_benchmarks.py``) and the full-vs-reduced
comparison (``compare_runtime_builds.py``) both drive the spectral-core ONNX
model through a built ORT runtime. The spectral-core interface has two named
inputs, ``spectral`` ``[1, 2, 2, 2048, T]`` and ``mix`` ``[1, 2, frames]``,
producing ``spectral_out`` ``[1, 4, 2, 2, 2048, T]`` and ``time_out``
``[1, 4, 2, frames]``.

The ``mix`` is a deterministic stereo waveform and
``spectral = spectral_reference.spec(mix)``, so the two inputs are
contract-consistent (contract v1) and repeated runs feed byte-identical
tensors. numpy and ``spectral_reference`` are imported lazily so importing this
module (e.g. for ``--help`` on a runner without numpy) stays cheap.
"""

from __future__ import annotations

from typing import Any

# The spectral-core interface is identified by exactly this input-name set.
SPECTRAL_INPUT_NAMES = frozenset({"spectral", "mix"})

# Contract-fixed dimensions of the spectral separation heads.
STEMS = 4      # drums / bass / other / vocals
CHANNELS = 2   # stereo


def deterministic_waveform(frames: int) -> "Any":
    """Return a deterministic stereo waveform ``[1, 2, frames]`` (float32).

    A fixed sum of sinusoids (no RNG) so repeated runs and the full-vs-reduced
    comparison feed byte-identical inputs to every session.
    """
    import numpy as np

    t = np.arange(frames, dtype=np.float64) / 44100.0
    left = (0.10 * np.sin(2.0 * np.pi * 220.0 * t)
            + 0.05 * np.sin(2.0 * np.pi * 440.0 * t)
            + 0.02 * np.sin(2.0 * np.pi * 1750.0 * t))
    right = (0.10 * np.sin(2.0 * np.pi * 221.0 * t)
             + 0.04 * np.sin(2.0 * np.pi * 660.0 * t)
             + 0.02 * np.sin(2.0 * np.pi * 1500.0 * t))
    return np.stack([left, right], axis=0)[np.newaxis].astype(np.float32)


def build_feed(session: "Any", frames: int) -> dict[str, "Any"]:
    """Build the spectral-core ORT feed dict for one inference window.

    The feed is keyed by the two contract input names ``spectral`` and ``mix``:
    ``mix`` is a deterministic stereo waveform and ``spectral`` is exactly
    ``spectral_reference.spec(mix)``, so the two inputs are contract-consistent.
    Raises ``ValueError`` if the session is not the spectral-core interface.
    """
    import numpy as np
    import spectral_reference

    names = {i.name for i in session.get_inputs()}
    if names != SPECTRAL_INPUT_NAMES:
        raise ValueError(
            f"session inputs {sorted(names)} are not the spectral-core interface "
            f"{sorted(SPECTRAL_INPUT_NAMES)}"
        )

    mix = deterministic_waveform(frames)                    # [1, 2, frames]
    spectral = spectral_reference.spec(mix.astype(np.float64)).astype(np.float32)
    return {"spectral": spectral, "mix": mix}


def expected_output_shapes(frames: int, spectral: "Any") -> dict[str, list[int]]:
    """Return the expected spectral-core output shapes keyed by output name.

    ``spectral`` is the synthesized spectral input tensor (used to read back the
    ``F`` and ``T`` dims).
    """
    freq = int(spectral.shape[-2])
    frames_t = int(spectral.shape[-1])
    return {
        "spectral_out": [1, STEMS, CHANNELS, 2, freq, frames_t],
        "time_out": [1, STEMS, CHANNELS, frames],
    }
