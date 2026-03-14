#!/usr/bin/env python3
"""
Validate the converted ONNX model against PyTorch reference output.

Runs the same input through both the PyTorch htdemucs model and the ONNX
model, then compares outputs to ensure numerical equivalence (MSE < 1e-4).

Tests multiple input lengths to verify dynamic axes work correctly.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort

ROOT_DIR = Path(__file__).parent.parent
ONNX_PATH = ROOT_DIR / "models" / "htdemucs.onnx"
SAMPLE_RATE = 44100

# Test durations in seconds
TEST_DURATIONS = [1, 5, 10]
MSE_THRESHOLD = 1e-4


def load_pytorch_model():
    """Load the pretrained htdemucs PyTorch model."""
    from demucs.pretrained import get_model

    model = get_model("htdemucs")
    model.eval()
    model.cpu()
    return model


def load_onnx_session():
    """Load the ONNX model for inference."""
    if not ONNX_PATH.exists():
        print(f"ERROR: ONNX model not found at {ONNX_PATH}")
        print("Run convert_htdemucs_to_onnx.py first.")
        sys.exit(1)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(ONNX_PATH), sess_options)

    print(f"ONNX model loaded: {ONNX_PATH}")
    print(f"  Inputs:  {[inp.name for inp in session.get_inputs()]}")
    print(f"  Outputs: {[out.name for out in session.get_outputs()]}")

    return session


def validate_single(pytorch_model, onnx_session, duration_seconds: float):
    """Run validation for a single input duration.

    Returns (mse, max_abs_diff, passed).
    """
    frames = int(SAMPLE_RATE * duration_seconds)
    print(f"\n--- Testing {duration_seconds}s ({frames} frames) ---")

    # Fixed seed for reproducibility
    torch.manual_seed(42)
    test_input = torch.randn(1, 2, frames)

    # PyTorch inference
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input)
    pytorch_np = pytorch_output.numpy()
    print(f"  PyTorch output shape: {pytorch_np.shape}")

    # ONNX inference
    input_name = onnx_session.get_inputs()[0].name
    onnx_output = onnx_session.run(None, {input_name: test_input.numpy()})
    onnx_np = onnx_output[0]
    print(f"  ONNX output shape:    {onnx_np.shape}")

    # Handle potential shape differences
    # PyTorch: (1, 4, 2, frames), ONNX might be (4, 2, frames) or (1, 4, 2, frames)
    if pytorch_np.shape != onnx_np.shape:
        # Try squeezing batch dim
        if pytorch_np.ndim == 4 and onnx_np.ndim == 3:
            pytorch_np = pytorch_np.squeeze(0)
        elif pytorch_np.ndim == 3 and onnx_np.ndim == 4:
            onnx_np = onnx_np.squeeze(0)

        if pytorch_np.shape != onnx_np.shape:
            # Trim to matching frame count (minor padding differences)
            min_frames = min(pytorch_np.shape[-1], onnx_np.shape[-1])
            pytorch_np = pytorch_np[..., :min_frames]
            onnx_np = onnx_np[..., :min_frames]

    # Compute metrics
    mse = np.mean((pytorch_np - onnx_np) ** 2)
    max_abs = np.max(np.abs(pytorch_np - onnx_np))
    mean_abs = np.mean(np.abs(pytorch_np - onnx_np))

    passed = mse < MSE_THRESHOLD

    print(f"  MSE:              {mse:.2e}")
    print(f"  Max abs diff:     {max_abs:.2e}")
    print(f"  Mean abs diff:    {mean_abs:.2e}")
    print(f"  Result:           {'PASS' if passed else 'FAIL'}")

    return mse, max_abs, passed


def validate_output_shape(onnx_session):
    """Verify output has expected stem structure."""
    print("\n--- Validating output structure ---")

    torch.manual_seed(42)
    test_input = torch.randn(1, 2, SAMPLE_RATE * 5).numpy()

    input_name = onnx_session.get_inputs()[0].name
    outputs = onnx_session.run(None, {input_name: test_input})
    output = outputs[0]

    # Expected: (1, 4, 2, frames) or (4, 2, frames)
    # 4 stems: drums, bass, other, vocals
    # 2 channels: stereo
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


def main():
    print("=" * 60)
    print("Demucs htdemucs ONNX Validation")
    print("=" * 60)

    pytorch_model = load_pytorch_model()
    onnx_session = load_onnx_session()

    # Validate structure
    if not validate_output_shape(onnx_session):
        print("\nVALIDATION FAILED: output structure mismatch")
        sys.exit(1)

    # Validate numerical accuracy at multiple durations
    all_passed = True
    results = []

    for duration in TEST_DURATIONS:
        mse, max_abs, passed = validate_single(pytorch_model, onnx_session, duration)
        results.append((duration, mse, max_abs, passed))
        if not passed:
            all_passed = False

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Duration':>10s} {'MSE':>12s} {'Max Abs':>12s} {'Status':>8s}")
    print("-" * 46)
    for duration, mse, max_abs, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{duration:>9.0f}s {mse:>12.2e} {max_abs:>12.2e} {status:>8s}")

    print()
    if all_passed:
        print("VALIDATION PASSED - all tests within MSE threshold")
    else:
        print(f"VALIDATION FAILED - MSE threshold: {MSE_THRESHOLD:.0e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
