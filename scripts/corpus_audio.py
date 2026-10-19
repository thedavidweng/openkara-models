"""Load reviewed reference bundles from an explicitly provisioned local directory.

No implicit network/download path: <sha256>.npz contains float32 mix [C,T],
stems [4,C,T], integer sample_rate and stem_names in Demucs contract order.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np

STEM_NAMES = ["drums", "bass", "other", "vocals"]


def load_fixture(fixture: dict, corpus_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    if fixture["kind"] == "synthetic":
        from synthetic_fixtures import generate_fixture
        return generate_fixture(fixture), None
    if corpus_dir is None:
        raise ValueError("real-audio requires --corpus-dir with reviewed reference bundles")
    source = fixture["generator"]
    raw = (corpus_dir / f"{source['sha256']}.npz").read_bytes()
    if len(raw) != source["size"] or hashlib.sha256(raw).hexdigest() != source["sha256"]:
        raise ValueError("reference bundle digest/size mismatch")
    with np.load(io.BytesIO(raw), allow_pickle=False) as bundle:
        if set(bundle.files) != {"mix", "stems", "sample_rate", "stem_names"}:
            raise ValueError("reference bundle must contain mix, stems, sample_rate and stem_names")
        mix, stems = bundle["mix"], bundle["stems"]
        rate, names = bundle["sample_rate"], bundle["stem_names"]
    if rate.shape != () or rate.dtype.kind not in "iu" or rate.item() != fixture["sample_rate"]:
        raise ValueError("reference bundle sample rate mismatch; no implicit resampling")
    if names.tolist() != STEM_NAMES or fixture["reference_stems"]["order"] != STEM_NAMES:
        raise ValueError("reference stem order mismatch")
    shape = (fixture["channels"], fixture["frames"])
    if mix.shape != shape or stems.shape != (4, *shape):
        raise ValueError("reference bundle shape mismatch")
    if any(a.dtype != np.dtype("float32") or not np.isfinite(a).all() for a in (mix, stems)):
        raise ValueError("reference audio must be finite float32")
    if not np.allclose(mix, stems.sum(axis=0), rtol=1e-5, atol=1e-6):
        raise ValueError("reference stems do not sum to mixture")
    return mix, stems


def separation_result(fixture: dict, reference: np.ndarray, estimate: np.ndarray,
                      segment_frames: int) -> dict:
    """Worst stem/pair values; chunk error compares stitched estimates to truth."""
    import separation_metrics as m
    if estimate.shape != (1, *reference.shape) or not np.isfinite(estimate).all():
        raise ValueError("separation output must be finite [1,4,C,T]")
    estimate = estimate[0]
    boundaries = list(range(segment_frames // 2, fixture["frames"], segment_frames // 2))
    if fixture["frames"] <= segment_frames:
        raise ValueError("real reference fixture must cross an actual inference chunk boundary")
    result = {"fixture_id": fixture["fixture_id"], "reference_bundle": {
        "sha256": fixture["generator"]["sha256"], "size": fixture["generator"]["size"]}}
    result.update({
        "si_sdr_db": min(m.si_sdr(r, e) for r, e in zip(reference, estimate)),
        "sdr_db": min(m.sdr(r, e) for r, e in zip(reference, estimate)),
        "leakage_db": max(m.leakage_db(reference[i], estimate[j]) for i in range(4) for j in range(4) if i != j),
        "mrstft_loss": max(m.multi_resolution_stft_loss(r, e) for r, e in zip(reference, estimate)),
        "transient_error": max(m.transient_error(r, e) for r, e in zip(reference, estimate)),
        "low_frequency_error_db": max(m.low_frequency_error_db(r, e, fixture["sample_rate"]) for r, e in zip(reference, estimate)),
        "chunk_boundary_error": max(m.chunk_boundary_error(r, e, boundaries,
            window=min(2048, segment_frames // 8))["boundary_mse"] for r, e in zip(reference, estimate)),
    })
    if not all(np.isfinite(v) for k, v in result.items() if k not in {"fixture_id", "reference_bundle"}):
        raise ValueError("undefined/nonfinite separation metric; curate a non-silent reference")
    return result
