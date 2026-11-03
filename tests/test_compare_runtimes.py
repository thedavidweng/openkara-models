import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_runtimes import FRAMES, fixture_audio, max_error, qualifies, verify_archive, windows


def test_real_fixture_crosses_boundary_and_windows_keep_samples():
    path = Path(__file__).parent / "fixtures" / "piano-cc0.wav"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "9f5bdcaf036f6f2453048d7d809b5db39d5edf846ac5a792474049721549c2e1"
    )
    audio = fixture_audio(path)
    chunks = list(windows(audio))
    assert audio.shape == (2, 441000)
    assert len(chunks) == 3
    np.testing.assert_array_equal(chunks[0][0], audio[:, :FRAMES])
    remainder = audio.shape[1] - FRAMES // 2
    np.testing.assert_array_equal(chunks[1][0, :, :remainder], audio[:, FRAMES // 2:])
    assert not chunks[1][0, :, remainder:].any()
    np.testing.assert_array_equal(chunks[2][0, :, :audio.shape[1] - 2 * (FRAMES // 2)],
                                  audio[:, 2 * (FRAMES // 2):])


def test_comparison_checks_every_tensor_and_rejects_invalid_output():
    original = [np.array([0, 1], dtype=np.float32), np.array([2], dtype=np.float32)]
    changed = [original[0].copy(), np.array([2.001], dtype=np.float32)]
    assert max_error(original, changed)[1] > 1e-4
    with pytest.raises(ValueError):
        max_error(original, [original[0]])
    with pytest.raises(ValueError):
        max_error(original, [original[0], np.array([np.nan], dtype=np.float32)])


def test_gate_enforces_error_and_linux_performance_limits():
    row = {
        "output_max_abs": [0, 1e-4],
        "stable_cold_load_s": 1,
        "candidate_cold_load_s": 1.5,
        "stable_warm_rtf": 1,
        "candidate_warm_rtf": 1.5,
    }
    linux = "x86_64-unknown-linux-gnu"
    assert qualifies([row], linux)
    assert not qualifies([], linux)
    assert not qualifies([{**row, "output_max_abs": [0, 1.01e-4]}], linux)
    for metric in ("candidate_cold_load_s", "candidate_warm_rtf"):
        assert not qualifies([{**row, metric: 1.51}], linux)
        assert qualifies([{**row, metric: 1.51}], "aarch64-apple-darwin")


def test_archive_requires_both_size_and_digest(tmp_path):
    archive = tmp_path / "runtime.zip"
    archive.write_bytes(b"runtime")
    digest = hashlib.sha256(b"runtime").hexdigest()
    assert verify_archive(archive, 7, digest) == digest
    with pytest.raises(ValueError, match="size"):
        verify_archive(archive, 8, digest)
    with pytest.raises(ValueError, match="SHA-256"):
        verify_archive(archive, 7, "0" * 64)
