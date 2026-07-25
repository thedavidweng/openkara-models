#!/usr/bin/env python3
"""Shared input synthesis for the native runtime benchmark/comparison harnesses.

The runtime benchmark (``run_runtime_benchmarks.py``) and the full-vs-reduced
comparison (``compare_runtime_builds.py``) both drive an ONNX model through a
built ORT runtime. Two model interfaces are supported:

  waveform : a single input ``[1, 2, frames]`` (the gen-1-pinned waveform
             manifest still consumed elsewhere).
  spectral : the generation-8 spectral-core interface with two named inputs,
             ``spectral`` ``[1, 2, 2, 2048, T]`` and ``mix`` ``[1, 2, frames]``,
             producing ``spectral_out`` ``[1, 4, 2, 2, 2048, T]`` and
             ``time_out`` ``[1, 4, 2, frames]``.

The interface is detected from the session's input-name set so the two
harnesses stay in lockstep. For the spectral interface the ``mix`` is a
deterministic stereo waveform and ``spectral = spectral_reference.spec(mix)``,
so the two inputs are contract-consistent (contract v1). numpy and
``spectral_reference`` are imported lazily so importing this module (e.g. for
``--help`` on a runner without numpy) stays cheap.
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


def detect_interface(session: "Any") -> str:
    """Return ``"spectral"`` or ``"waveform"`` from the session's input names."""
    names = {i.name for i in session.get_inputs()}
    return "spectral" if names == SPECTRAL_INPUT_NAMES else "waveform"


def build_feed(session: "Any", frames: int) -> tuple[dict[str, "Any"], str]:
    """Build the ORT feed dict for one inference window.

    Returns ``(feed, interface)`` where ``interface`` is ``"spectral"`` or
    ``"waveform"``. The feed is keyed by the session's declared input names so
    it is fed positionally-independent (by name).
    """
    import numpy as np

    interface = detect_interface(session)
    if interface == "spectral":
        import spectral_reference

        mix = deterministic_waveform(frames)                    # [1, 2, frames]
        spectral = spectral_reference.spec(mix.astype(np.float64)).astype(np.float32)
        return {"spectral": spectral, "mix": mix}, "spectral"

    # Waveform: single input, deterministic zeros (unchanged legacy behaviour).
    name = session.get_inputs()[0].name
    return {name: np.zeros((1, CHANNELS, frames), dtype=np.float32)}, "waveform"


def expected_output_shapes(interface: str, frames: int, spectral: "Any" | None,
                           waveform_shape: list[int] | None) -> dict[str, list[int]] | None:
    """Return expected output shapes keyed by output name for the spectral
    interface, or ``None`` for the waveform interface (which uses a single
    flat expected shape supplied by the caller).

    ``spectral`` is the synthesized spectral input tensor (used to read back
    the ``F`` and ``T`` dims); ``waveform_shape`` is unused for spectral.
    """
    if interface != "spectral":
        return None
    if spectral is None:
        raise ValueError("spectral interface requires the synthesized spectral tensor")
    freq = int(spectral.shape[-2])
    frames_t = int(spectral.shape[-1])
    return {
        "spectral_out": [1, STEMS, CHANNELS, 2, freq, frames_t],
        "time_out": [1, STEMS, CHANNELS, frames],
    }
