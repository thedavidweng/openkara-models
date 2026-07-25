#!/usr/bin/env python3
"""
Export Demucs spectral-core ONNX models (issue #23 PR 2).

The spectral-core boundary moves the waveform<->spectral transforms out of
the shipped graph: the public input is the contract spectral tensor plus the
raw mix (time branch), and the public output is the pre-ISTFT tensor plus the
time-branch waveform. Compared to the conv-DFT export
(convert_htdemucs_to_onnx.py) this removes ~134 MB of dense DFT filter-bank
constants per model and every transform node; architecture and weights are
otherwise identical.

Contract: docs/spectral-core-contract.md (openkara.spectral-contract/v1).

    stems[s] = ispec(spectral_out[:, s], 343980) + time_out[:, s]

CLI mirrors convert_htdemucs_to_onnx.py so convert.yml can drive both
interfaces with the same job topology:

    python scripts/export_spectral_core.py --model htdemucs
    python scripts/export_spectral_core.py --model htdemucs_ft --sub-model-index 0
    python scripts/export_spectral_core.py --model htdemucs_ft --merge-from sub-models/
"""

import argparse
import collections
import gc
import os
import sys
import tempfile
from pathlib import Path

import torch

SCRIPTS_DIR = Path(__file__).parent
ROOT_DIR = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from onnx_runtime_contract import (
    annotate_optimized_model,
    assert_release_onnx_compatible_with_official_ort,
    compute_sha256,
    make_contract_compliant_session,
    upsert_metadata_prop,
)
from demucs_loader import SUPPORTED_MODELS, load, load_sub_model
from spectral_core_patch import (
    SpectralCoreModule,
    SpectralCoreTracePatch,
    dummy_inputs,
)
from spectral_core_graph import (
    CONTRACT_INPUTS,
    CONTRACT_OUTPUTS,
    SPECTRAL_CONTRACT_METADATA,
    SPECTRAL_CONTRACT_VERSION,
    assert_spectral_core_graph,
    assert_spectral_core_metadata,
    build_spectral_core_ensemble,
    strip_time_zero_add,
)

INPUT_NAMES = [name for name, _ in CONTRACT_INPUTS]
OUTPUT_NAMES = [name for name, _ in CONTRACT_OUTPUTS]

OUTPUT_SEMANTICS = (
    "spectral_out=[1,4,2,2,2048,336] pre-ISTFT (de-normalized) "
    "drums/bass/other/vocals; time_out=[1,4,2,343980] time-branch waveform; "
    "stems[s] = ispec(spectral_out[:,s], 343980) + time_out[:,s]"
)
INPUT_SEMANTICS = (
    "spectral=[1,2,2,2048,336] contract-v1 spec(mix); "
    "mix=[1,2,343980] stereo 44.1 kHz waveform"
)


def export_spectral_core_onnx(model, output_path):
    """Trace the patched model at the spectral-core boundary and export."""
    import onnx

    print("Exporting spectral-core ONNX...")
    patch = SpectralCoreTracePatch.from_model(model)
    core = SpectralCoreModule(model, patch)
    spectral, mix = dummy_inputs()

    with patch, torch.no_grad():
        spectral_out, time_out = core.forward(spectral, mix)
    print(f"  Eager check: spectral_out {tuple(spectral_out.shape)}, "
          f"time_out {tuple(time_out.shape)}")
    del spectral_out, time_out

    # Suppress TracerWarnings from Demucs' shape-based asserts: the model is
    # traced with the fixed contract shapes, so those Python-level checks are
    # correctly constant-folded (same rationale as the conv-DFT exporter).
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        with patch:
            torch.onnx.export(
                core,
                (spectral, mix),
                str(output_path),
                input_names=INPUT_NAMES,
                output_names=OUTPUT_NAMES,
                opset_version=17,
                do_constant_folding=True,
                dynamo=False,
            )

    onnx_model = onnx.load(str(output_path))
    strip_time_zero_add(onnx_model)
    onnx.save(onnx_model, str(output_path))
    print("Spectral-core ONNX export succeeded (time-branch zero Add stripped).")


def verify_onnx(output_path):
    """Structural verification: checker, interface echo, op histogram."""
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


def optimize_and_annotate(raw_path, output_path):
    """ORT offline optimization + metadata + contract gate on the artifact."""
    import onnx

    make_contract_compliant_session(raw_path, optimized_model_filepath=output_path)
    if not output_path.exists():
        raise RuntimeError(f"ORT optimization did not write {output_path}")
    assert_release_onnx_compatible_with_official_ort(output_path)
    print(f"Optimized ONNX written to {output_path}")

    cache_key = annotate_optimized_model(output_path)

    onnx_model = onnx.load(str(output_path))
    upsert_metadata_prop(onnx_model, SPECTRAL_CONTRACT_METADATA,
                         SPECTRAL_CONTRACT_VERSION)
    upsert_metadata_prop(onnx_model, "openkara.tensor_interface", "spectral-core")
    upsert_metadata_prop(onnx_model, "openkara.stem_profile", "four-stem")
    upsert_metadata_prop(onnx_model, "openkara.input_semantics", INPUT_SEMANTICS)
    upsert_metadata_prop(onnx_model, "openkara.output_semantics", OUTPUT_SEMANTICS)
    onnx.save(onnx_model, str(output_path))
    print(f"Metadata: {SPECTRAL_CONTRACT_METADATA}={SPECTRAL_CONTRACT_VERSION}")

    onnx_model = onnx.load(str(output_path))
    assert_spectral_core_graph(onnx_model)
    assert_spectral_core_metadata(onnx_model)
    print("Spectral-core contract gate passed (interface, transform ops, "
          "DFT signatures, metadata).")
    return cache_key


def export_single_sub_model(model_name, index, output_path):
    """Export one spectral-core sub-model to raw ONNX (for parallel CI)."""
    sub_model, segment_frames, n_models = load_sub_model(model_name, index)
    print(f"\nSub-model {index}/{n_models - 1}, segment_frames={segment_frames}")

    os.makedirs(output_path.parent or Path("."), exist_ok=True)
    export_spectral_core_onnx(sub_model, output_path)
    verify_onnx(output_path)
    print(f"\nSpectral-core sub-model {index} export complete: {output_path}")


def merge_and_finalize(sub_model_dir, output_path):
    """Merge spectral-core sub-models, optimize, annotate, gate."""
    import onnx

    sub_paths = sorted(
        sub_model_dir.glob("sub_*.onnx"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not sub_paths:
        raise RuntimeError(f"No sub_*.onnx files found in {sub_model_dir}")

    n_models = len(sub_paths)
    print(f"Found {n_models} spectral-core sub-model files in {sub_model_dir}")

    sub_models = []
    for p in sub_paths:
        m = onnx.load(str(p))
        onnx.checker.check_model(m)
        sub_models.append(m)
        print(f"  Loaded: {p}")

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_output_path = Path(tmpdir) / "spectral_core.raw.onnx"
        ensemble = build_spectral_core_ensemble(sub_models, n_models)
        del sub_models
        gc.collect()
        onnx.save(ensemble, str(raw_output_path))
        del ensemble
        gc.collect()

        print("\nVerifying raw merged ONNX...")
        verify_onnx(raw_output_path)
        optimize_and_annotate(raw_output_path, output_path)

    verify_onnx(output_path)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Model size: {size_mb:.1f} MB")
    compute_sha256(output_path)
    print(f"\nSpectral-core merge + finalize complete: {output_path}")


def convert_ensemble(model_name, output_path):
    """Single-job ensemble path: export sub-cores one at a time, then merge."""
    model_0, segment_frames, n_models = load_sub_model(model_name, 0)
    del model_0
    gc.collect()
    print(f"\nEnsemble has {n_models} sub-models, segment_frames={segment_frames}")

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(n_models):
            print(f"\n{'=' * 60}")
            print(f"Exporting spectral-core sub-model {i + 1}/{n_models}")
            print(f"{'=' * 60}")

            sub_model, _, _ = load_sub_model(model_name, i)
            export_spectral_core_onnx(sub_model, Path(tmpdir) / f"sub_{i}.onnx")
            del sub_model
            gc.collect()

        merge_and_finalize(Path(tmpdir), output_path)


def default_output(model_name):
    return ROOT_DIR / "models" / f"{model_name}.spectral.onnx"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Demucs spectral-core ONNX models."
    )
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default="htdemucs",
        help="Model to export (default: htdemucs)",
    )
    parser.add_argument(
        "--sub-model-index",
        type=int,
        default=None,
        help="Export only this sub-model from a htdemucs_ft ensemble. "
        "Outputs a raw ONNX file (no ORT optimization). "
        "Used by CI parallel ensemble export.",
    )
    parser.add_argument(
        "--merge-from",
        type=Path,
        default=None,
        help="Merge sub-model ONNX files from this directory, then run ORT "
        "optimization and annotation. Expects sub_0.onnx, sub_1.onnx, ... "
        "Used by CI parallel ensemble export.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path. Defaults to models/<model>.spectral.onnx, or "
        "sub_<i>.onnx when --sub-model-index is set.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model

    if args.sub_model_index is not None:
        if model_name != "htdemucs_ft":
            print("--sub-model-index requires --model htdemucs_ft", file=sys.stderr)
            sys.exit(1)
        output = args.output or Path(f"sub_{args.sub_model_index}.onnx")
        export_single_sub_model(model_name, args.sub_model_index, output)
        return

    if args.merge_from is not None:
        if model_name != "htdemucs_ft":
            print("--merge-from requires --model htdemucs_ft", file=sys.stderr)
            sys.exit(1)
        output = args.output or default_output(model_name)
        os.makedirs(output.parent, exist_ok=True)
        merge_and_finalize(args.merge_from, output)
        return

    output_path = args.output or default_output(model_name)
    os.makedirs(output_path.parent, exist_ok=True)

    if model_name == "htdemucs_ft":
        convert_ensemble(model_name, output_path)
        return

    model, segment_frames = load(model_name)
    print(f"Contract segment: {segment_frames} frames")

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_output_path = Path(tmpdir) / f"{model_name}.spectral.raw.onnx"
        export_spectral_core_onnx(model, raw_output_path)
        del model
        gc.collect()

        print("\nVerifying raw ONNX export...")
        verify_onnx(raw_output_path)
        optimize_and_annotate(raw_output_path, output_path)

    verify_onnx(output_path)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Model size: {size_mb:.1f} MB")
    compute_sha256(output_path)
    print(f"\nSpectral-core export complete: {output_path}")


if __name__ == "__main__":
    main()
