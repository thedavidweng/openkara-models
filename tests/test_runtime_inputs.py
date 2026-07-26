"""Tests for the shared runtime input synthesis (scripts/runtime_inputs.py).

Covers the spectral-core feed construction used by both
run_runtime_benchmarks.py and compare_runtime_builds.py. numpy is required
for the synthesis path, so the whole module is skipped when it is absent
(e.g. the numpy-less source-lock CI job).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import runtime_inputs  # noqa: E402


class _NodeArg:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    """Minimal stand-in for an ORT InferenceSession exposing input names."""

    def __init__(self, input_names: list[str]) -> None:
        self._inputs = [_NodeArg(n) for n in input_names]

    def get_inputs(self):
        return self._inputs


def test_deterministic_waveform_shape_and_determinism() -> None:
    a = runtime_inputs.deterministic_waveform(4096)
    b = runtime_inputs.deterministic_waveform(4096)
    assert a.shape == (1, 2, 4096)
    assert a.dtype == np.float32
    assert np.array_equal(a, b)  # deterministic, no RNG
    assert np.all(np.isfinite(a))


def test_build_feed_spectral_is_contract_consistent() -> None:
    frames = 4096  # le = ceil(4096/1024) = 4 frames
    feed = runtime_inputs.build_feed(_FakeSession(["spectral", "mix"]), frames)
    assert set(feed) == {"spectral", "mix"}
    assert feed["mix"].shape == (1, 2, frames)
    assert feed["mix"].dtype == np.float32
    # spectral = spec(mix): [1, 2, 2, CONTRACT_FREQS, T]
    import spectral_reference
    assert feed["spectral"].shape == (1, 2, 2, spectral_reference.CONTRACT_FREQS, 4)
    assert feed["spectral"].dtype == np.float32
    assert np.all(np.isfinite(feed["spectral"]))
    # Contract consistency: spectral is exactly spec(mix).
    expected = spectral_reference.spec(feed["mix"].astype(np.float64)).astype(np.float32)
    assert np.array_equal(feed["spectral"], expected)


def test_build_feed_rejects_non_spectral_session() -> None:
    """A session whose inputs are not the spectral-core set is rejected — the
    spectral interface is the only supported path."""
    for names in (["mix"], ["audio"], ["spectral", "mix", "extra"]):
        with pytest.raises(ValueError):
            runtime_inputs.build_feed(_FakeSession(names), 4096)


def test_expected_output_shapes_spectral() -> None:
    frames = 4096
    feed = runtime_inputs.build_feed(_FakeSession(["spectral", "mix"]), frames)
    exp = runtime_inputs.expected_output_shapes(frames, feed["spectral"])
    import spectral_reference
    assert exp == {
        "spectral_out": [1, 4, 2, 2, spectral_reference.CONTRACT_FREQS, 4],
        "time_out": [1, 4, 2, frames],
    }
