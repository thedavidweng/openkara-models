#!/usr/bin/env python3
"""
Validate the converted ONNX model against PyTorch reference output.

Runs the same input through both the PyTorch model and the ONNX model,
then compares outputs to ensure numerical equivalence (MSE < 1e-4).
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort

ROOT_DIR = Path(__file__).parent.parent
SAMPLE_RATE = 44100

MSE_THRESHOLD = 1e-4

SUPPORTED_MODELS = ("htdemucs", "htdemucs_ft")


def load_pytorch_model(model_name):
    """Load the pretrained PyTorch model (single model, unwrapped)."""
    from demucs.pretrained import get_model

    bag = get_model(model_name)

    if hasattr(bag, "models"):
        model = bag.models[0]
        print(f"Unwrapped BagOfModels → {type(model).__name__}")
    else:
        model = bag

    model.eval()
    model.cpu()

    segment_frames = int(model.segment * model.samplerate)
    print(f"Model segment: {model.segment}s = {segment_frames} frames")

    return model, segment_frames


def compute_pytorch_ensemble_output(model_name, test_input):
    """Compute ensemble output by running sub-models one at a time.

    Loads each sub-model sequentially and averages their outputs,
    keeping memory usage low.
    """
    from demucs.pretrained import get_model

    bag = get_model(model_name)
    if not hasattr(bag, "models"):
        raise RuntimeError(f"{model_name} is not a BagOfModels")

    n_models = len(bag.models)
    segment_frames = int(bag.models[0].segment * bag.models[0].samplerate)
    # Keep models list but process one at a time
    models = list(bag.models)
    del bag
    gc.collect()

    print(f"Computing ensemble output from {n_models} sub-models...")
    accumulated = None

    for i, sub_model in enumerate(models):
        sub_model.eval()
        sub_model.cpu()
        with torch.no_grad():
            out = sub_model(test_input)
        print(f"  Sub-model {i} output shape: {out.shape}")
        if accumulated is None:
            accumulated = out.clone()
        else:
            accumulated += out
        del out

    result = accumulated / n_models
    return result, segment_frames


def load_onnx_session(onnx_path):
    """Load the ONNX model for inference."""
    if not onnx_path.exists():
        print(f"ERROR: ONNX model not found at {onnx_path}")
        print("Run convert_htdemucs_to_onnx.py first.")
        sys.exit(1)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_options)

    print(f"ONNX model loaded: {onnx_path}")
    print(f"  Inputs:  {[inp.name for inp in session.get_inputs()]}")
    print(f"  Outputs: {[out.name for out in session.get_outputs()]}")

    return session


def validate_single(pytorch_np, onnx_session, test_input_np, label):
    """Compare PyTorch output against ONNX output.

    Returns (mse, max_abs_diff, passed).
    """
    print(f"\n--- Testing {label} ---")
    print(f"  PyTorch output shape: {pytorch_np.shape}")

    # ONNX inference
    input_name = onnx_session.get_inputs()[0].name
    onnx_output = onnx_session.run(None, {input_name: test_input_np})
    onnx_np = onnx_output[0]
    print(f"  ONNX output shape:    {onnx_np.shape}")

    # Handle potential shape differences
    pt = pytorch_np
    ox = onnx_np
    if pt.shape != ox.shape:
        if pt.ndim == 4 and ox.ndim == 3:
            pt = pt.squeeze(0)
        elif pt.ndim == 3 and ox.ndim == 4:
            ox = ox.squeeze(0)

        if pt.shape != ox.shape:
            min_frames = min(pt.shape[-1], ox.shape[-1])
            pt = pt[..., :min_frames]
            ox = ox[..., :min_frames]

    # Compute metrics
    mse = np.mean((pt - ox) ** 2)
    max_abs = np.max(np.abs(pt - ox))
    mean_abs = np.mean(np.abs(pt - ox))

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

    # Fixed seed for reproducibility
    torch.manual_seed(42)

    if model_name == "htdemucs_ft":
        # For ensemble: compute reference by averaging sub-model outputs
        # Use a small test input first to get segment_frames
        from demucs.pretrained import get_model
        bag = get_model(model_name)
        segment_frames = int(bag.models[0].segment * bag.models[0].samplerate)
        del bag
        gc.collect()

        torch.manual_seed(42)
        test_input = torch.randn(1, 2, segment_frames)

        pytorch_output, _ = compute_pytorch_ensemble_output(model_name, test_input)
        pytorch_np = pytorch_output.numpy()
    else:
        pytorch_model, segment_frames = load_pytorch_model(model_name)
        torch.manual_seed(42)
        test_input = torch.randn(1, 2, segment_frames)
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input)
        pytorch_np = pytorch_output.numpy()

    onnx_session = load_onnx_session(onnx_path)

    # Validate structure
    if not validate_output_shape(onnx_session, segment_frames):
        print("\nVALIDATION FAILED: output structure mismatch")
        sys.exit(1)

    # Validate numerical accuracy
    label = f"full segment ({segment_frames / SAMPLE_RATE:.1f}s)"
    torch.manual_seed(42)
    test_input_np = torch.randn(1, 2, segment_frames).numpy()
    mse, max_abs, passed = validate_single(pytorch_np, onnx_session, test_input_np, label)

    # Summary
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
