#!/usr/bin/env python3
"""
Validate the converted ONNX model against PyTorch reference output.

Runs the same input through both the PyTorch model and the ONNX model,
then compares outputs to ensure numerical equivalence (MSE < 1e-4).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

ROOT_DIR = Path(__file__).parent.parent
SAMPLE_RATE = 44100

MSE_THRESHOLD = 1e-4

SUPPORTED_MODELS = ("htdemucs", "htdemucs_ft")


class EnsembleHTDemucs(nn.Module):
    """Wrapper that runs multiple HTDemucs sub-models and averages their outputs."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        first = models[0]
        self.segment = first.segment
        self.samplerate = first.samplerate

    def forward(self, x):
        outputs = [m(x) for m in self.models]
        return torch.stack(outputs).mean(dim=0)


def load_pytorch_model(model_name):
    """Load the pretrained PyTorch model.

    For htdemucs: unwraps BagOfModels to get the single HDemucs instance.
    For htdemucs_ft: wraps all sub-models in EnsembleHTDemucs for averaging.
    """
    from demucs.pretrained import get_model

    bag = get_model(model_name)

    if model_name == "htdemucs_ft":
        if hasattr(bag, "models"):
            models = list(bag.models)
            print(f"Loaded {model_name}: BagOfModels with {len(models)} sub-models")
            model = EnsembleHTDemucs(models)
        else:
            model = bag
    else:
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


def validate_single(pytorch_model, onnx_session, frames: int, label: str = ""):
    """Run validation for a single input size.

    Returns (mse, max_abs_diff, passed).
    """
    print(f"\n--- Testing {label} ({frames} frames) ---")

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
    if pytorch_np.shape != onnx_np.shape:
        if pytorch_np.ndim == 4 and onnx_np.ndim == 3:
            pytorch_np = pytorch_np.squeeze(0)
        elif pytorch_np.ndim == 3 and onnx_np.ndim == 4:
            onnx_np = onnx_np.squeeze(0)

        if pytorch_np.shape != onnx_np.shape:
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


def validate_output_shape(onnx_session, segment_frames: int):
    """Verify output has expected stem structure."""
    print("\n--- Validating output structure ---")

    torch.manual_seed(42)
    test_input = torch.randn(1, 2, segment_frames).numpy()

    input_name = onnx_session.get_inputs()[0].name
    outputs = onnx_session.run(None, {input_name: test_input})
    output = outputs[0]

    # Expected: (1, 4, 2, frames) or (4, 2, frames)
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

    pytorch_model, segment_frames = load_pytorch_model(model_name)
    onnx_session = load_onnx_session(onnx_path)

    # Validate structure
    if not validate_output_shape(onnx_session, segment_frames):
        print("\nVALIDATION FAILED: output structure mismatch")
        sys.exit(1)

    # Validate numerical accuracy
    all_passed = True
    results = []

    test_cases = [
        (segment_frames, f"full segment ({segment_frames / SAMPLE_RATE:.1f}s)"),
    ]

    for frames, label in test_cases:
        mse, max_abs, passed = validate_single(pytorch_model, onnx_session, frames, label)
        results.append((label, mse, max_abs, passed))
        if not passed:
            all_passed = False

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Test':>30s} {'MSE':>12s} {'Max Abs':>12s} {'Status':>8s}")
    print("-" * 66)
    for label, mse, max_abs, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{label:>30s} {mse:>12.2e} {max_abs:>12.2e} {status:>8s}")

    print()
    if all_passed:
        print("VALIDATION PASSED - all tests within MSE threshold")
    else:
        print(f"VALIDATION FAILED - MSE threshold: {MSE_THRESHOLD:.0e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
