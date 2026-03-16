#!/usr/bin/env python3
"""
Convert pretrained Demucs htdemucs/htdemucs_ft model from PyTorch to ONNX format.

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

import argparse
import os
import sys
import hashlib
import math
import types
from pathlib import Path

import torch
import torch.nn as nn

# Add scripts directory to path for local imports
SCRIPTS_DIR = Path(__file__).parent
ROOT_DIR = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

SUPPORTED_MODELS = ("htdemucs", "htdemucs_ft")

# htdemucs default STFT parameters
N_FFT = 4096
HOP_LENGTH = 1024
SAMPLE_RATE = 44100


class EnsembleHTDemucs(nn.Module):
    """Wrapper that runs multiple HTDemucs sub-models and averages their outputs.

    Used for htdemucs_ft which is a BagOfModels containing 4 fine-tuned
    HTDemucs instances. This wrapper makes the ensemble exportable as a
    single ONNX graph with the same interface as a single HTDemucs model.
    """

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        # Copy attributes from the first sub-model for compatibility
        first = models[0]
        self.segment = first.segment
        self.samplerate = first.samplerate
        self.nfft = first.nfft
        self.hop_length = first.hop_length
        self.cac = first.cac

    def forward(self, x):
        outputs = [m(x) for m in self.models]
        return torch.stack(outputs).mean(dim=0)


def load_model(model_name):
    """Load a pretrained Demucs model.

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
            # Unexpected: htdemucs_ft should always be a BagOfModels
            model = bag
            print(f"Warning: {model_name} is not a BagOfModels, using as-is")
    else:
        # htdemucs: unwrap BagOfModels to get single model
        if hasattr(bag, "models"):
            model = bag.models[0]
            print(f"Unwrapped BagOfModels → {type(model).__name__}")
        else:
            model = bag

    model.eval()
    model.cpu()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {model_name} model: {total_params:,} parameters")

    # Read the model's fixed segment length
    segment_frames = int(model.segment * model.samplerate)
    print(f"Model segment: {model.segment}s = {segment_frames} frames")

    return model, segment_frames


def try_direct_export(model, dummy_input, output_path):
    """Attempt direct ONNX export using the legacy TorchScript exporter."""
    print("Attempting direct ONNX export with legacy exporter (opset 17)...")

    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        input_names=["audio"],
        output_names=["stems"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print("Direct export succeeded.")


def _get_sub_models(model):
    """Get the list of HDemucs instances to patch.

    For EnsembleHTDemucs, returns all sub-models.
    For a single HDemucs, returns [model].
    """
    if isinstance(model, EnsembleHTDemucs):
        return list(model.models)
    return [model]


def bind_real_valued_export_patch(model):
    """Replace Demucs' complex spectrogram path with real-valued equivalents.

    Patches all sub-models if the model is an EnsembleHTDemucs.
    """
    import torch.nn.functional as F
    from demucs.hdemucs import pad1d
    from onnx_stft import OnnxSTFT, OnnxISTFT

    sub_models = _get_sub_models(model)
    all_originals = []

    for sub_model in sub_models:
        if not sub_model.cac:
            raise RuntimeError("Real-valued export patch expects an HTDemucs model with cac=True")

        stft_module = OnnxSTFT(n_fft=sub_model.nfft, hop_length=sub_model.hop_length)
        istft_module = OnnxISTFT(n_fft=sub_model.nfft, hop_length=sub_model.hop_length)

        originals = {
            "_spec": sub_model._spec,
            "_ispec": sub_model._ispec,
            "_magnitude": sub_model._magnitude,
            "_mask": sub_model._mask,
        }
        all_originals.append((sub_model, originals))

        def make_exportable_spec(stft_mod):
            def exportable_spec(self, x):
                hl = self.hop_length
                nfft = self.nfft
                assert hl == nfft // 4
                le = int(math.ceil(x.shape[-1] / hl))
                pad = hl // 2 * 3
                x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")
                z = stft_mod(x)[:, :, :-1, :, :]
                assert z.shape[-2] == le + 4, (z.shape, x.shape, le)
                z = z[:, :, :, 2: 2 + le, :]
                return z.permute(0, 1, 4, 2, 3).contiguous()
            return exportable_spec

        def make_exportable_ispec(istft_mod):
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
                x = istft_mod(z, length=le)
                x = x.view(batch, sources, channels, le)
                return x[..., pad: pad + length]
            return exportable_ispec

        def exportable_magnitude(self, z):
            batch, channels, _, freqs, frames = z.shape
            return z.reshape(batch, channels * 2, freqs, frames)

        def exportable_mask(self, z, m):
            batch, sources, _, freqs, frames = m.shape
            return m.view(batch, sources, -1, 2, freqs, frames).contiguous()

        sub_model._spec = types.MethodType(make_exportable_spec(stft_module), sub_model)
        sub_model._ispec = types.MethodType(make_exportable_ispec(istft_module), sub_model)
        sub_model._magnitude = types.MethodType(exportable_magnitude, sub_model)
        sub_model._mask = types.MethodType(exportable_mask, sub_model)

    def restore():
        for sub_model, originals in all_originals:
            for name, method in originals.items():
                setattr(sub_model, name, method)

    return restore


def try_patched_export(model, dummy_input, output_path):
    """Export with a real-valued conv1d/conv_transpose1d STFT fallback."""
    print("Attempting patched export with real-valued STFT/ISTFT...")

    restore = None
    try:
        restore = bind_real_valued_export_patch(model)

        torch.onnx.export(
            model,
            (dummy_input,),
            str(output_path),
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


def verify_onnx(output_path):
    """Run basic ONNX model validation."""
    import onnx

    print(f"Verifying {output_path}...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model, full_check=True)
    print("ONNX checker passed.")

    print(f"  Inputs:  {[inp.name for inp in onnx_model.graph.input]}")
    print(f"  Outputs: {[out.name for out in onnx_model.graph.output]}")
    print(f"  Opset:   {onnx_model.opset_import[0].version}")


def compute_sha256(output_path):
    """Compute and save SHA-256 checksum."""
    sha256 = hashlib.sha256()
    with open(output_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    digest = sha256.hexdigest()
    print(f"SHA-256: {digest}")
    return digest


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Demucs model to ONNX format."
    )
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default="htdemucs",
        help="Model to convert (default: htdemucs)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model
    output_path = ROOT_DIR / "models" / f"{model_name}.onnx"

    os.makedirs(output_path.parent, exist_ok=True)

    # Load model and get its fixed segment length
    model, segment_frames = load_model(model_name)

    # Create dummy input matching the model's exact segment length.
    dummy_input = torch.randn(1, 2, segment_frames)
    print(f"Dummy input shape: {dummy_input.shape}")

    # Quick forward pass to verify model works
    with torch.no_grad():
        output = model(dummy_input)
    print(f"PyTorch output shape: {output.shape}")

    # Try export strategies
    try:
        try_direct_export(model, dummy_input, output_path)
    except Exception as e:
        print(f"Direct export failed: {e}")
        print()
        try:
            try_patched_export(model, dummy_input, output_path)
        except Exception as e2:
            print(f"Patched export also failed: {e2}")
            print()
            print("Both export strategies failed.")
            print("Please inspect the exporter logs above and update the fallback path.")
            sys.exit(1)

    # Verify
    verify_onnx(output_path)

    # File size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Model size: {size_mb:.1f} MB")

    # SHA-256
    compute_sha256(output_path)

    print(f"\nConversion complete: {output_path}")


if __name__ == "__main__":
    main()
