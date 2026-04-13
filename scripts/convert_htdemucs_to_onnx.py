#!/usr/bin/env python3
"""
Convert pretrained Demucs htdemucs/htdemucs_ft model from PyTorch to ONNX format.

The main challenge is that ONNX does not natively support complex-valued
STFT/ISTFT operations, and newer PyTorch ONNX export paths choke on Demucs'
data-dependent asserts. This script uses two strategies:

1. Direct export with the legacy TorchScript-based ONNX exporter.
2. Fallback: replace Demucs' complex spectrogram path with a fully
   real-valued conv1d/conv_transpose1d implementation from onnx_stft.py.

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
import math
import tempfile
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
MODEL_CACHE_KEY_METADATA = "openkara.model_cache_key"
MODEL_OPTIMIZED_BY_METADATA = "openkara.optimized_by"


def load_single_model(model_name):
    """Load a single pretrained Demucs model (unwrapped from BagOfModels)."""
    from demucs.pretrained import get_model

    bag = get_model(model_name)

    if hasattr(bag, "models"):
        model = bag.models[0]
        print(f"Unwrapped BagOfModels → {type(model).__name__}")
    else:
        model = bag

    model.eval()
    model.cpu()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {model_name} model: {total_params:,} parameters")

    segment_frames = int(model.segment * model.samplerate)
    print(f"Model segment: {model.segment}s = {segment_frames} frames")

    return model, segment_frames


def load_sub_model(model_name, index):
    """Load a single sub-model from a BagOfModels by index.

    Loads the full BagOfModels, extracts the requested sub-model,
    and discards the rest to conserve memory.
    """
    from demucs.pretrained import get_model

    bag = get_model(model_name)

    if not hasattr(bag, "models"):
        raise RuntimeError(f"{model_name} is not a BagOfModels")

    n_models = len(bag.models)
    if index >= n_models:
        raise RuntimeError(f"Sub-model index {index} >= {n_models}")

    model = bag.models[index]
    # Discard the bag to free memory
    del bag
    gc.collect()

    model.eval()
    model.cpu()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded sub-model {index}: {total_params:,} parameters")

    segment_frames = int(model.segment * model.samplerate)
    return model, segment_frames, n_models


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


def bind_real_valued_export_patch(model):
    """Replace Demucs' complex spectrogram path with real-valued equivalents."""
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


def export_single_model(model, dummy_input, output_path):
    """Export a single HTDemucs model to ONNX, trying direct then patched."""
    try:
        try_direct_export(model, dummy_input, output_path)
    except Exception as e:
        print(f"Direct export failed: {e}")
        print()
        try_patched_export(model, dummy_input, output_path)


def merge_ensemble_onnx(sub_model_paths, output_path, n_models):
    """Merge multiple ONNX sub-model graphs into a single ensemble graph.

    Creates a combined ONNX model that:
    1. Feeds the same input to all sub-models
    2. Averages their outputs
    3. Produces a single output with the same shape as any individual model

    Each sub-model graph is namespaced (prefixed) to avoid node name collisions.
    """
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    import numpy as np

    print(f"\nMerging {n_models} sub-model ONNX files into ensemble...")

    # Load all sub-models
    sub_models = []
    for i, path in enumerate(sub_model_paths):
        m = onnx.load(str(path))
        onnx.checker.check_model(m)
        sub_models.append(m)
        print(f"  Loaded sub-model {i}: {path}")

    # Use the first model as reference for input/output shapes
    ref = sub_models[0]
    ref_input = ref.graph.input[0]
    ref_output = ref.graph.output[0]

    # Collect all initializers and nodes from each sub-model, prefixed
    all_initializers = []
    all_nodes = []
    sub_output_names = []

    for i, m in enumerate(sub_models):
        prefix = f"sub{i}_"

        # Build a rename map for this sub-model
        # All internal names get prefixed, but the input name maps to shared input
        internal_names = set()
        for node in m.graph.node:
            for name in list(node.input) + list(node.output):
                internal_names.add(name)
        for init in m.graph.initializer:
            internal_names.add(init.name)
        for inp in m.graph.input:
            internal_names.add(inp.name)
        for out in m.graph.output:
            internal_names.add(out.name)

        input_name = m.graph.input[0].name
        output_name = m.graph.output[0].name
        prefixed_output = prefix + output_name
        sub_output_names.append(prefixed_output)

        def rename(name):
            if not name:
                return ""  # Empty string = optional input not provided
            if name == input_name:
                return "audio"  # Shared input
            return prefix + name

        # Prefix initializers
        for init in m.graph.initializer:
            new_init = onnx.TensorProto()
            new_init.CopyFrom(init)
            new_init.name = rename(init.name)
            all_initializers.append(new_init)

        # Prefix nodes
        for node in m.graph.node:
            new_node = helper.make_node(
                node.op_type,
                inputs=[rename(n) for n in node.input],
                outputs=[rename(n) for n in node.output],
                name=prefix + node.name if node.name else "",
                domain=node.domain,
            )
            # Copy attributes
            for attr in node.attribute:
                new_node.attribute.append(attr)
            all_nodes.append(new_node)

    # Add averaging nodes:
    # 1. Concat all sub-model outputs along a new dim 0: (n_models, 1, 4, 2, frames)
    concat_output = "ensemble_concat"
    all_nodes.append(helper.make_node(
        "Unsqueeze",
        inputs=[sub_output_names[0], "unsqueeze_axes"],
        outputs=["unsqueeze_0"],
    ))
    for i in range(1, n_models):
        all_nodes.append(helper.make_node(
            "Unsqueeze",
            inputs=[sub_output_names[i], "unsqueeze_axes"],
            outputs=[f"unsqueeze_{i}"],
        ))

    all_nodes.append(helper.make_node(
        "Concat",
        inputs=[f"unsqueeze_{i}" for i in range(n_models)],
        outputs=[concat_output],
        axis=0,
    ))

    # 2. ReduceMean along axis 0 (use attribute form for opset < 18)
    all_nodes.append(helper.make_node(
        "ReduceMean",
        inputs=[concat_output],
        outputs=["stems"],
        axes=[0],
        keepdims=0,
    ))

    # Add constant tensor for Unsqueeze axes (opset 13+ takes axes as input)
    axes_0 = numpy_helper.from_array(np.array([0], dtype=np.int64), name="unsqueeze_axes")
    all_initializers.append(axes_0)

    # Build the ensemble graph
    ensemble_input = helper.make_tensor_value_info(
        "audio",
        TensorProto.FLOAT,
        [d.dim_value if d.dim_value else d.dim_param
         for d in ref_input.type.tensor_type.shape.dim],
    )
    ensemble_output = helper.make_tensor_value_info(
        "stems",
        TensorProto.FLOAT,
        [d.dim_value if d.dim_value else d.dim_param
         for d in ref_output.type.tensor_type.shape.dim],
    )

    graph = helper.make_graph(
        all_nodes,
        "htdemucs_ft_ensemble",
        [ensemble_input],
        [ensemble_output],
        initializer=all_initializers,
    )

    # Copy opset from reference model
    opset_imports = [helper.make_opsetid(op.domain, op.version) for op in ref.opset_import]

    ensemble_model = helper.make_model(graph, opset_imports=opset_imports)
    ensemble_model.ir_version = ref.ir_version

    # Run shape inference and optimization before final check.
    # Shape inference resolves missing type info that the checker requires.
    from onnx import shape_inference
    print("Running shape inference on ensemble model...")
    ensemble_model = shape_inference.infer_shapes(ensemble_model)

    # Save with external data if model is large (>2GB protobuf limit)
    print(f"Saving ensemble model to {output_path}...")
    onnx.save(ensemble_model, str(output_path))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Ensemble model saved: {output_path} ({size_mb:.1f} MB)")

    # Basic verification (skip full_check to avoid re-running expensive
    # shape inference; we already ran it above)
    print("Verifying ensemble model...")
    loaded = onnx.load(str(output_path))
    onnx.checker.check_model(loaded)
    print("Ensemble ONNX checker passed.")


def convert_ensemble(model_name, output_path):
    """Convert htdemucs_ft by exporting sub-models one at a time then merging.

    This approach keeps peak memory usage comparable to a single htdemucs export,
    allowing it to run on standard CI runners with ~7GB RAM.
    """
    # First, discover how many sub-models and get segment_frames
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

            # Load only this sub-model
            sub_model, _, _ = load_sub_model(model_name, i)

            # Quick forward pass
            with torch.no_grad():
                out = sub_model(dummy_input)
            print(f"  PyTorch output shape: {out.shape}")
            del out

            # Export to temp file
            tmp_path = Path(tmpdir) / f"sub_{i}.onnx"
            export_single_model(sub_model, dummy_input, tmp_path)

            sub_model_paths.append(tmp_path)

            # Free memory before next sub-model
            del sub_model
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Merge all sub-models into one ensemble ONNX
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
    print_operator_inventory(onnx_model)


def print_operator_inventory(onnx_model):
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


def get_graph_stats(model_path):
    """Return (file_size_bytes, node_count, op_counter) for an ONNX model."""
    import onnx

    size_bytes = model_path.stat().st_size
    onnx_model = onnx.load(str(model_path))
    node_count = len(onnx_model.graph.node)
    op_counter = collections.Counter(n.op_type for n in onnx_model.graph.node)
    return size_bytes, node_count, op_counter


def print_raw_vs_optimized(raw_path, opt_path):
    """Print a structured comparison of raw export vs ORT-optimized artifact."""
    raw_size, raw_nodes, raw_ops = get_graph_stats(raw_path)
    opt_size, opt_nodes, opt_ops = get_graph_stats(opt_path)

    print("\n" + "=" * 60)
    print("Raw vs Optimized Comparison")
    print("=" * 60)
    print(f"  {'':30s} {'Raw':>12s} {'Optimized':>12s} {'Delta':>12s}")
    print(f"  {'-'*66}")
    print(f"  {'File size (MB)':30s} {raw_size/1048576:>12.1f} {opt_size/1048576:>12.1f} {(opt_size-raw_size)/1048576:>+12.1f}")
    print(f"  {'Node count':30s} {raw_nodes:>12d} {opt_nodes:>12d} {opt_nodes-raw_nodes:>+12d}")

    all_ops = sorted(set(raw_ops) | set(opt_ops), key=lambda o: -(raw_ops.get(o, 0) + opt_ops.get(o, 0)))
    print(f"\n  {'Operator':30s} {'Raw':>6s} {'Opt':>6s} {'Delta':>6s}")
    print(f"  {'-'*48}")
    for op in all_ops[:15]:
        r = raw_ops.get(op, 0)
        o = opt_ops.get(op, 0)
        delta = o - r
        marker = "" if delta == 0 else f"{delta:+d}"
        print(f"  {op:30s} {r:>6d} {o:>6d} {marker:>6s}")
    print()


def optimize_onnx_with_ort(input_path, output_path):
    """Run ONNX Runtime offline graph optimization and emit the final model."""
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath = str(output_path)

    ort.InferenceSession(
        str(input_path),
        sess_options,
        providers=["CPUExecutionProvider"],
    )

    if not output_path.exists():
        raise RuntimeError(f"ORT optimization did not write {output_path}")

    print(f"Optimized ONNX written to {output_path}")


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

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_output_path = Path(tmpdir) / f"{model_name}.raw.onnx"

        if model_name == "htdemucs_ft":
            # Ensemble: export sub-models one-by-one then merge
            convert_ensemble(model_name, raw_output_path)
        else:
            # Single model
            model, segment_frames = load_single_model(model_name)

            dummy_input = torch.randn(1, 2, segment_frames)
            print(f"Dummy input shape: {dummy_input.shape}")

            with torch.no_grad():
                output = model(dummy_input)
            print(f"PyTorch output shape: {output.shape}")

            export_single_model(model, dummy_input, raw_output_path)

        print("\nVerifying raw ONNX export...")
        verify_onnx(raw_output_path)
        optimize_onnx_with_ort(raw_output_path, output_path)
        print_raw_vs_optimized(raw_output_path, output_path)
        annotate_optimized_model(output_path)

    # Verify final optimized artifact
    verify_onnx(output_path)

    # File size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Model size: {size_mb:.1f} MB")

    # SHA-256
    compute_sha256(output_path)

    print(f"\nConversion complete: {output_path}")


if __name__ == "__main__":
    main()
