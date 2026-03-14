#!/usr/bin/env python3
"""
Convert pretrained Demucs htdemucs model from PyTorch to ONNX format.

The main challenge is that ONNX does not natively support complex-valued
STFT/ISTFT operations. This script uses two strategies:

1. Direct export (PyTorch 2.1+ / opset 17): torch.stft maps to ONNX STFT op,
   and complex tensors are decomposed automatically.
2. Fallback: monkey-patch torch.stft/istft with conv1d-based real-valued
   implementations from onnx_stft.py, so the tracer captures exportable ops.

References:
- https://github.com/sevagh/demucs.onnx
- https://mixxx.org/news/2025-10-27-gsoc2025-demucs-to-onnx-dhunstack/
- https://github.com/adefossez/demucs/pull/10
"""

import os
import sys
import hashlib
from pathlib import Path

import torch
import numpy as np

# Add scripts directory to path for local imports
SCRIPTS_DIR = Path(__file__).parent
ROOT_DIR = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

OUTPUT_PATH = ROOT_DIR / "models" / "htdemucs.onnx"

# htdemucs default STFT parameters
N_FFT = 4096
HOP_LENGTH = 1024
SAMPLE_RATE = 44100
DUMMY_SECONDS = 10


def load_htdemucs():
    """Load the pretrained htdemucs model."""
    from demucs.pretrained import get_model

    model = get_model("htdemucs")
    model.eval()
    model.cpu()
    print(f"Loaded htdemucs model: {sum(p.numel() for p in model.parameters())} parameters")
    return model


def try_direct_export(model, dummy_input):
    """Attempt direct ONNX export using PyTorch's built-in STFT/complex support.

    Works with PyTorch 2.1+ and opset 17 which has native STFT op and
    automatic complex tensor decomposition.
    """
    print("Attempting direct ONNX export (opset 17)...")

    # HDemucs.forward() returns (batch, sources, channels, frames)
    # = (1, 4, 2, frames)
    torch.onnx.export(
        model,
        (dummy_input,),
        str(OUTPUT_PATH),
        input_names=["audio"],
        output_names=["stems"],
        dynamic_axes={
            "audio": {2: "frame_count"},
            "stems": {3: "frame_count"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print("Direct export succeeded.")


def try_patched_export(model, dummy_input):
    """Export with monkey-patched STFT/ISTFT using conv1d-based implementations.

    Replaces torch.stft and torch.istft at the module level during export
    so the ONNX tracer captures conv1d operations instead of unsupported
    complex STFT ops.
    """
    from onnx_stft import OnnxSTFT, OnnxISTFT

    print("Attempting patched export with conv1d-based STFT/ISTFT...")

    stft_module = OnnxSTFT(n_fft=N_FFT, hop_length=HOP_LENGTH)
    istft_module = OnnxISTFT(n_fft=N_FFT, hop_length=HOP_LENGTH)

    # Save originals
    orig_stft = torch.stft
    orig_istft = torch.istft

    def patched_stft(input, n_fft, hop_length=None, win_length=None,
                     window=None, center=True, pad_mode='reflect',
                     normalized=False, onesided=None, return_complex=None):
        """Replacement for torch.stft that uses conv1d internally."""
        # Our OnnxSTFT handles center padding and windowing
        if input.dim() == 1:
            input = input.unsqueeze(0)
        if input.dim() == 2:
            input = input.unsqueeze(0)
        # Returns (batch, channels, freq, time, 2) with last dim = [real, imag]
        spec = stft_module(input)
        # Flatten batch/channel dims back to match torch.stft output
        batch, ch, freq, time, ri = spec.shape
        spec = spec.reshape(batch * ch, freq, time, ri)
        if batch * ch == 1:
            spec = spec.squeeze(0)
        # Convert to complex if requested (for compatibility with downstream code)
        if return_complex:
            return torch.view_as_complex(spec.contiguous())
        return spec

    def patched_istft(input, n_fft, hop_length=None, win_length=None,
                      window=None, center=True, normalized=False,
                      onesided=None, length=None, return_complex=False):
        """Replacement for torch.istft that uses conv_transpose1d internally."""
        if torch.is_complex(input):
            input = torch.view_as_real(input)
        if input.dim() == 3:
            # (freq, time, 2) -> (1, 1, freq, time, 2)
            input = input.unsqueeze(0).unsqueeze(0)
        elif input.dim() == 4:
            # (batch, freq, time, 2) -> (batch, 1, freq, time, 2)
            input = input.unsqueeze(1)
        signal = istft_module(input, length=length or 0)
        # Flatten back
        batch, ch, frames = signal.shape
        signal = signal.reshape(batch * ch, frames)
        if batch * ch == 1:
            signal = signal.squeeze(0)
        return signal

    try:
        torch.stft = patched_stft
        torch.istft = patched_istft

        torch.onnx.export(
            model,
            (dummy_input,),
            str(OUTPUT_PATH),
            input_names=["audio"],
            output_names=["stems"],
            dynamic_axes={
                "audio": {2: "frame_count"},
                "stems": {3: "frame_count"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        print("Patched export succeeded.")
    finally:
        # Restore originals
        torch.stft = orig_stft
        torch.istft = orig_istft


def verify_onnx():
    """Run basic ONNX model validation."""
    import onnx

    print(f"Verifying {OUTPUT_PATH}...")
    onnx_model = onnx.load(str(OUTPUT_PATH))
    onnx.checker.check_model(onnx_model, full_check=True)
    print("ONNX checker passed.")

    # Print model info
    print(f"  Inputs:  {[inp.name for inp in onnx_model.graph.input]}")
    print(f"  Outputs: {[out.name for out in onnx_model.graph.output]}")
    print(f"  Opset:   {onnx_model.opset_import[0].version}")


def compute_sha256():
    """Compute and save SHA-256 checksum."""
    sha256 = hashlib.sha256()
    with open(OUTPUT_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    digest = sha256.hexdigest()
    print(f"SHA-256: {digest}")
    return digest


def main():
    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)

    # Load model
    model = load_htdemucs()

    # Create dummy input: stereo audio at 44.1kHz
    dummy_input = torch.randn(1, 2, SAMPLE_RATE * DUMMY_SECONDS)
    print(f"Dummy input shape: {dummy_input.shape}")

    # Quick forward pass to verify model works
    with torch.no_grad():
        output = model(dummy_input)
    print(f"PyTorch output shape: {output.shape}")
    # Expected: (1, 4, 2, 441000)

    # Try export strategies
    try:
        try_direct_export(model, dummy_input)
    except Exception as e:
        print(f"Direct export failed: {e}")
        print()
        try:
            try_patched_export(model, dummy_input)
        except Exception as e2:
            print(f"Patched export also failed: {e2}")
            print()
            print("Both export strategies failed.")
            print("Please check PyTorch version (need 2.1+) and report this issue.")
            sys.exit(1)

    # Verify
    verify_onnx()

    # File size
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Model size: {size_mb:.1f} MB")

    # SHA-256
    compute_sha256()

    print(f"\nConversion complete: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
