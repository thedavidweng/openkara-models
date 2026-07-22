#!/usr/bin/env python3
"""
Validate the converted ONNX model against PyTorch reference output.

Runs the same input through both the PyTorch model and the ONNX model,
then compares outputs to ensure numerical equivalence (MSE < 1e-4).

This is the **conversion pipeline smoke test** — a fast single-input check
that runs immediately after ONNX export in the convert.yml workflow. It is
NOT the release quality gate. The release gate is scripts/run_quality_suite.py,
which runs the full corpus fixture set (see quality/corpus-manifest.json) and
enforces tiered budgets (see quality/budgets.json). A stable catalog release
must not rely solely on this single-input MSE check; it must also pass the
full quality suite + runtime quality suite + enforce_quality_gates.py.
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import onnx
import torch

ROOT_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

from onnx_runtime_contract import (
    assert_release_onnx_compatible_with_official_ort,
    make_contract_compliant_session,
    MODEL_CACHE_KEY_METADATA,
    MODEL_OPTIMIZED_BY_METADATA,
)
from demucs_loader import SUPPORTED_MODELS, load, iter_sub_models

MSE_THRESHOLD = 1e-4


def compute_pytorch_ensemble_output(model_name, test_input):
    """Compute ensemble output by running sub-models one at a time.

    Loads each sub-model sequentially and averages their outputs,
    keeping memory usage low. Returns (averaged_output, segment_frames).
    """
    n_models, segment_frames, sub_models = iter_sub_models(model_name)

    print(f"Computing ensemble output from {n_models} sub-models...")
    accumulated = None

    for sub_model, seg, i in sub_models:
        with torch.no_grad():
            out = sub_model(test_input)
        print(f"  Sub-model {i} output shape: {out.shape}")
        if accumulated is None:
            accumulated = out.clone()
        else:
            accumulated += out
        del out, sub_model
        gc.collect()

    return accumulated / n_models, segment_frames


def load_onnx_session(onnx_path):
    """Load the ONNX model for inference."""
    if not onnx_path.exists():
        print(f"ERROR: ONNX model not found at {onnx_path}")
        print("Run convert_htdemucs_to_onnx.py first.")
        sys.exit(1)

    session = make_contract_compliant_session(onnx_path)

    print(f"ONNX model loaded: {onnx_path}")
    print(f"  Inputs:  {[inp.name for inp in session.get_inputs()]}")
    print(f"  Outputs: {[out.name for out in session.get_outputs()]}")

    return session


def validate_model_metadata(onnx_path):
    onnx_model = onnx.load(str(onnx_path))
    metadata = {prop.key: prop.value for prop in onnx_model.metadata_props}

    missing = [
        key
        for key in (MODEL_CACHE_KEY_METADATA, MODEL_OPTIMIZED_BY_METADATA)
        if key not in metadata
    ]
    if missing:
        print(f"  ERROR: missing metadata keys: {', '.join(missing)}")
        return False

    print("\n--- Validating optimized artifact metadata ---")
    print(f"  {MODEL_CACHE_KEY_METADATA}: {metadata[MODEL_CACHE_KEY_METADATA]}")
    print(f"  {MODEL_OPTIMIZED_BY_METADATA}: {metadata[MODEL_OPTIMIZED_BY_METADATA]}")

    if metadata[MODEL_OPTIMIZED_BY_METADATA] != "onnxruntime":
        print("  ERROR: optimized-by metadata must be 'onnxruntime'")
        return False

    return True


def validate_single(pytorch_np, onnx_session, test_input_np, label):
    """Compare PyTorch output against ONNX output.

    Returns (mse, max_abs_diff, passed).
    """
    print(f"\n--- Testing {label} ---")
    print(f"  PyTorch output shape: {pytorch_np.shape}")

    input_name = onnx_session.get_inputs()[0].name
    onnx_output = onnx_session.run(None, {input_name: test_input_np})
    onnx_np = onnx_output[0]
    print(f"  ONNX output shape:    {onnx_np.shape}")

    if pytorch_np.shape != onnx_np.shape:
        raise AssertionError(
            f"PyTorch {pytorch_np.shape} != ONNX {onnx_np.shape}: "
            "shape mismatch indicates a conversion bug"
        )

    mse = np.mean((pytorch_np - onnx_np) ** 2)
    max_abs = np.max(np.abs(pytorch_np - onnx_np))
    mean_abs = np.mean(np.abs(pytorch_np - onnx_np))

    passed = mse < MSE_THRESHOLD

    print(f"  MSE:              {mse:.2e}")
    print(f"  Max abs diff:     {max_abs:.2e}")
    print(f"  Mean abs diff:    {mean_abs:.2e}")
    print(f"  Result:           {'PASS' if passed else 'FAIL'}")

    return mse, max_abs, passed


def validate_output_shape(onnx_session, segment_frames):
    """Verify output has expected stem structure."""
    print("\n--- Validating output structure ---")

    torch.manual_seed(42)
    test_input = torch.randn(1, 2, segment_frames).numpy()

    input_name = onnx_session.get_inputs()[0].name
    outputs = onnx_session.run(None, {input_name: test_input})
    output = outputs[0]

    if output.ndim == 4:
        n_stems = output.shape[1]
        n_channels = output.shape[2]
    elif output.ndim == 3:
        n_stems = output.shape[0]
        n_channels = output.shape[1]
    else:
        print(f"  ERROR: unexpected output ndim={output.ndim}, shape={output.shape}")
        return False

    print(f"  Output shape: {output.shape}")
    print(f"  Stems:    {n_stems} (expected 4)")
    print(f"  Channels: {n_channels} (expected 2)")

    if n_stems != 4:
        print(f"  ERROR: expected 4 stems, got {n_stems}")
        return False
    if n_channels != 2:
        print(f"  ERROR: expected 2 channels, got {n_channels}")
        return False

    print("  Structure: PASS")
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate ONNX model against PyTorch reference."
    )
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default="htdemucs",
        help="Model to validate (default: htdemucs)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model
    onnx_path = ROOT_DIR / "models" / f"{model_name}.onnx"

    print("=" * 60)
    print(f"Demucs {model_name} ONNX Validation")
    print("=" * 60)

    if not onnx_path.exists():
        print(f"ERROR: ONNX model not found at {onnx_path}")
        print("Run convert_htdemucs_to_onnx.py first.")
        sys.exit(1)

    try:
        assert_release_onnx_compatible_with_official_ort(onnx_path)
    except RuntimeError as e:
        print(e)
        sys.exit(1)
    print("Runtime contract (operator domains): OK")

    torch.manual_seed(42)

    if model_name == "htdemucs_ft":
        from demucs_loader import load_sub_model
        m0, segment_frames, _ = load_sub_model(model_name, 0)
        del m0
        gc.collect()

        torch.manual_seed(42)
        test_input = torch.randn(1, 2, segment_frames)

        pytorch_output, _ = compute_pytorch_ensemble_output(model_name, test_input)
        pytorch_np = pytorch_output.numpy()
    else:
        pytorch_model, segment_frames = load(model_name)
        torch.manual_seed(42)
        test_input = torch.randn(1, 2, segment_frames)
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input)
        pytorch_np = pytorch_output.numpy()

    onnx_session = load_onnx_session(onnx_path)

    if not validate_model_metadata(onnx_path):
        print("\nVALIDATION FAILED: optimized artifact metadata missing or invalid")
        sys.exit(1)

    if not validate_output_shape(onnx_session, segment_frames):
        print("\nVALIDATION FAILED: output structure mismatch")
        sys.exit(1)

    label = f"full segment ({segment_frames / 44100:.1f}s)"
    torch.manual_seed(42)
    test_input_np = torch.randn(1, 2, segment_frames).numpy()
    mse, max_abs, passed = validate_single(pytorch_np, onnx_session, test_input_np, label)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Test':>30s} {'MSE':>12s} {'Max Abs':>12s} {'Status':>8s}")
    print("-" * 66)
    status = "PASS" if passed else "FAIL"
    print(f"{label:>30s} {mse:>12.2e} {max_abs:>12.2e} {status:>8s}")

    print()
    if passed:
        print("VALIDATION PASSED - all tests within MSE threshold")
    else:
        print(f"VALIDATION FAILED - MSE threshold: {MSE_THRESHOLD:.0e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
