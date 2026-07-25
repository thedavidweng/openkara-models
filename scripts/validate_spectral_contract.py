#!/usr/bin/env python3
"""
Validate a spectral-core ONNX artifact against the spectral contract
(issue #23 PR 2; docs/spectral-core-contract.md).

Structural validation (default, no torch required):

- official-ORT operator-domain gate;
- exact contract tensor interface, forbidden transform ops, dense DFT
  filter-bank initializer signatures (scripts/spectral_core_graph.py);
- required artifact metadata (cache key, optimized-by, contract version);
- an ORT CPU inference run on a deterministic fixture with shape and
  finiteness checks.

Numeric validation (``--torch-reference``, conversion-workflow only):

- the ONNX core outputs must match the PyTorch model traced at the same
  boundary (stage-level equivalence);
- the contract composition ``ispec(spectral_out) + time_out`` (float64
  numpy reference transforms) must match the ORIGINAL unpatched PyTorch
  model's waveform output (end-to-end equivalence), using the same
  MSE < 1e-4 gate as the waveform conversion pipeline.

This is the conversion-pipeline check for spectral-core artifacts — the
release quality gate remains scripts/run_quality_suite.py (issue #21).

Usage::

    python scripts/validate_spectral_contract.py models/htdemucs.spectral.onnx
    python scripts/validate_spectral_contract.py models/htdemucs.spectral.onnx \
        --torch-reference --model htdemucs
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import onnx

SCRIPTS_DIR = Path(__file__).parent
ROOT_DIR = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import spectral_reference as sr
from onnx_runtime_contract import (
    MODEL_CACHE_KEY_METADATA,
    MODEL_OPTIMIZED_BY_METADATA,
    assert_release_onnx_compatible_with_official_ort,
    make_contract_compliant_session,
)
from spectral_core_graph import (
    CONTRACT_OUTPUTS,
    SEGMENT_FRAMES,
    assert_spectral_core_graph,
    assert_spectral_core_metadata,
)

MSE_THRESHOLD = 1e-4


def deterministic_mix(seed=42):
    """Deterministic full-band stereo fixture at the contract segment size."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((1, 2, SEGMENT_FRAMES)).astype(np.float32)


def run_core(session, spectral, mix):
    outputs = session.run(None, {"spectral": spectral, "mix": mix})
    return outputs[0], outputs[1]


def validate_structure(onnx_path):
    print("--- Structural validation ---")
    assert_release_onnx_compatible_with_official_ort(onnx_path)
    print("Runtime contract (operator domains): OK")

    model = onnx.load(str(onnx_path), load_external_data=True)
    assert_spectral_core_graph(model)
    assert_spectral_core_metadata(model)
    print("Spectral-core contract gate: OK")

    metadata = {p.key: p.value for p in model.metadata_props}
    missing = [
        key
        for key in (MODEL_CACHE_KEY_METADATA, MODEL_OPTIMIZED_BY_METADATA)
        if key not in metadata
    ]
    if missing:
        raise AssertionError(f"missing metadata keys: {', '.join(missing)}")
    if metadata[MODEL_OPTIMIZED_BY_METADATA] != "onnxruntime":
        raise AssertionError("optimized-by metadata must be 'onnxruntime'")
    print(f"Metadata: {MODEL_CACHE_KEY_METADATA}="
          f"{metadata[MODEL_CACHE_KEY_METADATA]}")
    del model
    gc.collect()

    print("Loading ORT CPU session...")
    session = make_contract_compliant_session(onnx_path)
    mix = deterministic_mix()
    spectral = sr.spec(mix).astype(np.float32)
    print("Running deterministic fixture...")
    spectral_out, time_out = run_core(session, spectral, mix)

    for (name, shape), arr in zip(CONTRACT_OUTPUTS, (spectral_out, time_out)):
        if tuple(arr.shape) != tuple(shape):
            raise AssertionError(
                f"output {name!r} shape {arr.shape} != contract {shape}"
            )
        if not np.isfinite(arr).all():
            raise AssertionError(f"output {name!r} contains non-finite values")
        print(f"  {name}: shape OK, finite, rms={np.sqrt(np.mean(arr ** 2)):.4e}")

    print("Structural validation: PASS")
    return session


def _compare(label, reference, actual):
    mse = float(np.mean((reference - actual) ** 2))
    max_abs = float(np.max(np.abs(reference - actual)))
    passed = mse < MSE_THRESHOLD
    print(f"  {label:>14s}: MSE {mse:.2e}  max-abs {max_abs:.2e}  "
          f"{'PASS' if passed else 'FAIL'}")
    return passed


def torch_references(model_name, mix_np, spectral_np):
    """Reference outputs: (waveform y_ref, core zout_ref, core time_ref).

    The waveform reference comes from the ORIGINAL unpatched model; the core
    references come from the same weights traced at the spectral boundary.
    For htdemucs_ft each sub-model is loaded once and both forwards run
    before it is released (ensemble = plain average, matching the shipped
    waveform ensemble graph).
    """
    import torch
    from demucs_loader import iter_sub_models, load
    from spectral_core_patch import SpectralCoreModule, SpectralCoreTracePatch

    mix_t = torch.from_numpy(mix_np)
    spectral_t = torch.from_numpy(spectral_np)

    def both_forwards(model):
        with torch.no_grad():
            y = model(mix_t)
        patch = SpectralCoreTracePatch.from_model(model)
        core = SpectralCoreModule(model, patch)
        with patch, torch.no_grad():
            zout, time = core(spectral_t, mix_t)
        return y.numpy(), zout.numpy(), time.numpy()

    if model_name == "htdemucs_ft":
        n_models, _, sub_models = iter_sub_models(model_name)
        acc = None
        for sub_model, _, i in sub_models:
            print(f"  torch reference: sub-model {i}...")
            outs = both_forwards(sub_model)
            if acc is None:
                acc = list(outs)
            else:
                for j, o in enumerate(outs):
                    acc[j] += o
            del sub_model, outs
            gc.collect()
        return tuple(a / n_models for a in acc)

    model, _ = load(model_name)
    outs = both_forwards(model)
    del model
    gc.collect()
    return outs


def validate_torch_reference(session, model_name):
    print("\n--- Numeric validation against PyTorch reference ---")
    import torch

    torch.manual_seed(42)
    mix = deterministic_mix()
    spectral = sr.spec(mix).astype(np.float32)

    y_ref, zout_ref, time_ref = torch_references(model_name, mix, spectral)

    print("Running ONNX core...")
    spectral_out, time_out = run_core(session, spectral, mix)

    print("Stage-level core equivalence (ONNX vs traced-boundary torch):")
    ok = _compare("spectral_out", zout_ref, spectral_out)
    ok &= _compare("time_out", time_ref, time_out)

    print("End-to-end contract composition (ispec(spectral_out) + time_out "
          "vs original torch waveform):")
    composed = (
        sr.ispec(spectral_out.astype(np.float64), SEGMENT_FRAMES)
        + time_out.astype(np.float64)
    ).astype(np.float32)
    ok &= _compare("composed", y_ref, composed)

    if not ok:
        raise AssertionError(
            f"numeric validation failed (MSE threshold {MSE_THRESHOLD:.0e})"
        )
    print("Numeric validation: PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="Spectral-core ONNX file")
    parser.add_argument(
        "--torch-reference",
        action="store_true",
        help="Also validate numerically against the PyTorch reference "
        "(requires torch + demucs; conversion workflow only).",
    )
    parser.add_argument(
        "--model",
        default="htdemucs",
        help="Source model for --torch-reference (default: htdemucs)",
    )
    args = parser.parse_args()

    if not args.artifact.is_file():
        print(f"ERROR: artifact not found: {args.artifact}", file=sys.stderr)
        return 1

    print("=" * 60)
    print(f"Spectral-core contract validation: {args.artifact}")
    print("=" * 60)

    session = validate_structure(args.artifact)

    if args.torch_reference:
        validate_torch_reference(session, args.model)

    print("\nVALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
