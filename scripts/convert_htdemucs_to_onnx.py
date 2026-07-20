#!/usr/bin/env python3
"""
Convert pretrained Demucs htdemucs/htdemucs_ft model from PyTorch to ONNX format.

ONNX does not natively support complex-valued STFT/ISTFT operations used by
Demucs. This script replaces Demucs' complex spectrogram path with a fully
real-valued conv1d/conv_transpose1d implementation from onnx_stft.py before
exporting.

For htdemucs_ft (4-model ensemble), sub-models are exported one at a time
to stay within CI runner memory limits, then merged into a single ONNX
graph with an averaging output.

References:
- https://github.com/sevagh/demucs.onnx
- https://mixxx.org/news/2025-10-27-gsoc2025-demucs-to-onnx-dhunstack/
- https://github.com/adefossez/demucs/pull/10
"""

import argparse
import collections
import gc
import os
import sys
import hashlib
import tempfile
from pathlib import Path

import torch

SCRIPTS_DIR = Path(__file__).parent
ROOT_DIR = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from onnx_runtime_contract import (
    assert_release_onnx_compatible_with_official_ort,
    make_contract_compliant_session,
    MODEL_CACHE_KEY_METADATA,
    MODEL_OPTIMIZED_BY_METADATA,
)
from onnx_stft import RealValuedSpectrogramPatch
from demucs_loader import SUPPORTED_MODELS, load, load_sub_model
from ensemble_merge import build_ensemble_graph


def export_model_with_real_stft(model, dummy_input, output_path):
    """Export with real-valued conv1d/conv_transpose1d STFT/ISTFT."""
    print("Exporting ONNX with real-valued STFT/ISTFT...")

    with RealValuedSpectrogramPatch.from_model(model):
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
    print("ONNX export succeeded.")


def merge_ensemble_onnx(sub_model_paths, output_path, n_models):
    """Load sub-model ONNX files, merge into an ensemble graph, and save.

    Thin I/O wrapper around build_ensemble_graph.
    """
    import onnx

    print(f"\nMerging {n_models} sub-model ONNX files into ensemble...")

    sub_models = []
    for i, path in enumerate(sub_model_paths):
        m = onnx.load(str(path))
        onnx.checker.check_model(m)
        sub_models.append(m)
        print(f"  Loaded sub-model {i}: {path}")

    print("Running shape inference on ensemble model...")
    ensemble_model = build_ensemble_graph(sub_models, n_models)

    print(f"Saving ensemble model to {output_path}...")
    onnx.save(ensemble_model, str(output_path))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Ensemble model saved: {output_path} ({size_mb:.1f} MB)")

    print("Verifying ensemble model...")
    loaded = onnx.load(str(output_path))
    onnx.checker.check_model(loaded)
    print("Ensemble ONNX checker passed.")


def convert_ensemble(model_name, output_path):
    """Convert htdemucs_ft by exporting sub-models one at a time then merging.

    This approach keeps peak memory usage comparable to a single htdemucs export,
    allowing it to run on standard CI runners with ~7GB RAM.
    """
    model_0, segment_frames, n_models = load_sub_model(model_name, 0)
    del model_0
    gc.collect()
    print(f"\nEnsemble has {n_models} sub-models, segment_frames={segment_frames}")

    dummy_input = torch.randn(1, 2, segment_frames)

    sub_model_paths = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(n_models):
            print(f"\n{'='*60}")
            print(f"Exporting sub-model {i+1}/{n_models}")
            print(f"{'='*60}")

            sub_model, _, _ = load_sub_model(model_name, i)

            with torch.no_grad():
                out = sub_model(dummy_input)
            print(f"  PyTorch output shape: {out.shape}")
            del out

            tmp_path = Path(tmpdir) / f"sub_{i}.onnx"
            export_model_with_real_stft(sub_model, dummy_input, tmp_path)

            sub_model_paths.append(tmp_path)

            del sub_model
            gc.collect()

        merge_ensemble_onnx(sub_model_paths, output_path, n_models)

    return segment_frames


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

    counts = collections.Counter(node.op_type for node in onnx_model.graph.node)
    print("  Top ops:")
    for op_type, count in counts.most_common(12):
        print(f"    - {op_type}: {count}")


def upsert_metadata_prop(onnx_model, key, value):
    for prop in onnx_model.metadata_props:
        if prop.key == key:
            prop.value = value
            return
    prop = onnx_model.metadata_props.add()
    prop.key = key
    prop.value = value


def annotate_optimized_model(output_path):
    """Attach deterministic metadata to the final optimized ONNX artifact.

    The cache key is derived from the optimized model bytes before metadata is
    injected so it changes whenever graph structure or weights change.
    """
    import onnx

    cache_key = hashlib.sha256(output_path.read_bytes()).hexdigest()
    onnx_model = onnx.load(str(output_path))
    upsert_metadata_prop(onnx_model, MODEL_CACHE_KEY_METADATA, cache_key)
    upsert_metadata_prop(onnx_model, MODEL_OPTIMIZED_BY_METADATA, "onnxruntime")
    onnx.save(onnx_model, str(output_path))
    print(f"Metadata: {MODEL_CACHE_KEY_METADATA}={cache_key}")
    return cache_key


def optimize_onnx_with_ort(input_path, output_path):
    """Run ONNX Runtime offline graph optimization and emit the final model.

    Uses the contract-compliant session factory with optimized_model_filepath
    set so ORT writes the optimized graph to output_path. See
    docs/runtime-contract.md for why ORT_ENABLE_EXTENDED (not ALL).
    """
    make_contract_compliant_session(input_path, optimized_model_filepath=output_path)

    if not output_path.exists():
        raise RuntimeError(f"ORT optimization did not write {output_path}")

    assert_release_onnx_compatible_with_official_ort(output_path)
    print(f"Optimized ONNX written to {output_path}")


def compute_sha256(output_path):
    """Compute and save SHA-256 checksum."""
    with open(output_path, "rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
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

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_output_path = Path(tmpdir) / f"{model_name}.raw.onnx"

        if model_name == "htdemucs_ft":
            convert_ensemble(model_name, raw_output_path)
        else:
            model, segment_frames = load(model_name)

            dummy_input = torch.randn(1, 2, segment_frames)
            print(f"Dummy input shape: {dummy_input.shape}")

            with torch.no_grad():
                output = model(dummy_input)
            print(f"PyTorch output shape: {output.shape}")

            export_model_with_real_stft(model, dummy_input, raw_output_path)

        print("\nVerifying raw ONNX export...")
        verify_onnx(raw_output_path)
        optimize_onnx_with_ort(raw_output_path, output_path)
        annotate_optimized_model(output_path)

    verify_onnx(output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Model size: {size_mb:.1f} MB")

    compute_sha256(output_path)

    print(f"\nConversion complete: {output_path}")


if __name__ == "__main__":
    main()
