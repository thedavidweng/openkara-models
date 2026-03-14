#!/usr/bin/env python3
"""
Convert pretrained Demucs htdemucs model from PyTorch to ONNX format.

The main challenge is that ONNX does not natively support complex-valued
STFT/ISTFT operations, and newer PyTorch ONNX export paths choke on Demucs'
data-dependent asserts. This script uses two strategies:

1. Direct export with the legacy TorchScript-based ONNX exporter.
2. Fallback: replace Demucs' complex spectrogram path with a fully
   real-valued conv1d/conv_transpose1d implementation from onnx_stft.py.

References:
- https://github.com/sevagh/demucs.onnx
- https://mixxx.org/news/2025-10-27-gsoc2025-demucs-to-onnx-dhunstack/
- https://github.com/adefossez/demucs/pull/10
"""

import os
import sys
import hashlib
import math
import types
from pathlib import Path

import torch

# Add scripts directory to path for local imports
SCRIPTS_DIR = Path(__file__).parent
ROOT_DIR = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

OUTPUT_PATH = ROOT_DIR / "models" / "htdemucs.onnx"

# htdemucs default STFT parameters
N_FFT = 4096
HOP_LENGTH = 1024
SAMPLE_RATE = 44100


def load_htdemucs():
    """Load the pretrained htdemucs model.

    get_model() may return a BagOfModels wrapper whose forward() raises
    NotImplementedError. We unwrap it to get the raw HDemucs instance
    with a working forward() method.
    """
    from demucs.pretrained import get_model

    bag = get_model("htdemucs")

    # Unwrap BagOfModels if needed
    if hasattr(bag, "models"):
        model = bag.models[0]
        print(f"Unwrapped BagOfModels → {type(model).__name__}")
    else:
        model = bag

    model.eval()
    model.cpu()
    print(f"Loaded htdemucs model: {sum(p.numel() for p in model.parameters())} parameters")

    # Read the model's fixed segment length
    # HTDemucs.forward() requires input of EXACTLY this many frames.
    # Longer inputs cause a reshape failure at the time-branch output.
    # Shorter inputs are padded internally, but the ONNX graph is traced
    # with a fixed size so we must use the exact training segment.
    segment_frames = int(model.segment * model.samplerate)
    print(f"Model segment: {model.segment}s = {segment_frames} frames")

    return model, segment_frames


def try_direct_export(model, dummy_input):
    """Attempt direct ONNX export using the legacy TorchScript exporter.

    PyTorch 2.9+ defaults torch.onnx.export() to the newer torch.export-based
    exporter, which currently trips over data-dependent asserts inside Demucs.
    We explicitly keep the legacy exporter here for reproducibility.
    """
    print("Attempting direct ONNX export with legacy exporter (opset 17)...")

    # HDemucs.forward() returns (batch, sources, channels, frames)
    # = (1, 4, 2, segment_frames)
    # No dynamic_axes: HTDemucs requires fixed segment length internally
    # (reshape at time-branch output hardcodes training_length).
    # The consumer (OpenKara) handles chunking for longer audio.
    torch.onnx.export(
        model,
        (dummy_input,),
        str(OUTPUT_PATH),
        input_names=["audio"],
        output_names=["stems"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print("Direct export succeeded.")


def bind_real_valued_export_patch(model):
    """Replace Demucs' complex spectrogram path with real-valued equivalents.

    HTDemucs with `cac=True` packs real/imag bins into the channel dimension.
    For export we keep that representation explicit and avoid creating complex
    tensors entirely, because the legacy ONNX exporter cannot lower
    `view_as_complex`.
    """
    import torch.nn.functional as F
    from demucs.hdemucs import pad1d
    from onnx_stft import OnnxSTFT, OnnxISTFT

    if not model.cac:
        raise RuntimeError("Real-valued export patch expects an HTDemucs model with cac=True")

    stft_module = OnnxSTFT(n_fft=model.nfft, hop_length=model.hop_length)
    istft_module = OnnxISTFT(n_fft=model.nfft, hop_length=model.hop_length)

    originals = {
        "_spec": model._spec,
        "_ispec": model._ispec,
        "_magnitude": model._magnitude,
        "_mask": model._mask,
    }

    def exportable_spec(self, x):
        hl = self.hop_length
        nfft = self.nfft
        assert hl == nfft // 4
        le = int(math.ceil(x.shape[-1] / hl))
        pad = hl // 2 * 3
        x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")
        z = stft_module(x)[:, :, :-1, :, :]
        assert z.shape[-2] == le + 4, (z.shape, x.shape, le)
        z = z[:, :, :, 2: 2 + le, :]
        return z.permute(0, 1, 4, 2, 3).contiguous()

    def exportable_ispec(self, z, length=None, scale=0):
        assert scale == 0, "Scaled ISTFT export is not implemented"
        hl = self.hop_length
        z = F.pad(z, (0, 0, 0, 1))
        z = F.pad(z, (2, 2))
        pad = hl // 2 * 3
        le = hl * int(math.ceil(length / hl)) + 2 * pad

        batch, sources, channels, _, freqs, frames = z.shape
        z = z.permute(0, 1, 2, 4, 5, 3).contiguous()
        z = z.view(batch * sources, channels, freqs, frames, 2)
        x = istft_module(z, length=le)
        x = x.view(batch, sources, channels, le)
        return x[..., pad: pad + length]

    def exportable_magnitude(self, z):
        batch, channels, _, freqs, frames = z.shape
        return z.reshape(batch, channels * 2, freqs, frames)

    def exportable_mask(self, z, m):
        batch, sources, _, freqs, frames = m.shape
        return m.view(batch, sources, -1, 2, freqs, frames).contiguous()

    model._spec = types.MethodType(exportable_spec, model)
    model._ispec = types.MethodType(exportable_ispec, model)
    model._magnitude = types.MethodType(exportable_magnitude, model)
    model._mask = types.MethodType(exportable_mask, model)

    def restore():
        for name, method in originals.items():
            setattr(model, name, method)

    return restore


def try_patched_export(model, dummy_input):
    """Export with a real-valued conv1d/conv_transpose1d STFT fallback."""
    print("Attempting patched export with real-valued STFT/ISTFT...")

    restore = None
    try:
        restore = bind_real_valued_export_patch(model)

        torch.onnx.export(
            model,
            (dummy_input,),
            str(OUTPUT_PATH),
            input_names=["audio"],
            output_names=["stems"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
        print("Patched export succeeded.")
    finally:
        if restore is not None:
            restore()


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

    # Load model and get its fixed segment length
    model, segment_frames = load_htdemucs()

    # Create dummy input matching the model's exact segment length.
    # HTDemucs.forward() requires input of exactly segment_frames;
    # it cannot handle longer audio (reshape failure in time branch).
    dummy_input = torch.randn(1, 2, segment_frames)
    print(f"Dummy input shape: {dummy_input.shape}")

    # Quick forward pass to verify model works
    with torch.no_grad():
        output = model(dummy_input)
    print(f"PyTorch output shape: {output.shape}")
    # Expected: (1, 4, 2, segment_frames)

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
            print("Please inspect the exporter logs above and update the fallback path.")
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
