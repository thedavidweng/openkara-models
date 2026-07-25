#!/usr/bin/env python3
"""Derive the deterministic ``karaoke_2stem`` projection (issue #22).

Takes an approved four-output HTDemucs graph (output ``stems``
``[1, 4, 2, F]`` with source order drums/bass/other/vocals per the catalog
output contract) and appends deterministic graph nodes:

    vocals        = stems[:, 3:4]
    accompaniment = stems[:, 0:1] + stems[:, 1:2] + stems[:, 2:3]

The projected graph exposes exactly ``[1, 2, 2, F]`` — vocals then
accompaniment. The shared neural-network computation is unchanged; the
projection reduces host-visible outputs and application-side mixing work.
It does NOT eliminate the four-source network computation.

Only operators already present in the stable models' reduced-operator set
are used (Slice, Sum, Concat), so the projection loads on every published
reduced-build ONNX Runtime without a runtime rebuild.

Deterministic: repeated derivation from the same source is byte-identical.

Usage::

    python scripts/derive_karaoke_2stem.py --input htdemucs.onnx --output htdemucs.karaoke2stem.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from onnx_runtime_contract import MODEL_CACHE_KEY_METADATA  # noqa: E402

PREFIX = "openkara_k2s_"
SOURCE_ORDER = ("drums", "bass", "other", "vocals")
PROJECTION_SPEC = "vocals=src[3]; accompaniment=src[0]+src[1]+src[2]"


def _int64_init(name: str, values: list[int]) -> onnx.TensorProto:
    return onnx.numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)


def derive_karaoke_2stem(model: onnx.ModelProto, source_sha256: str) -> onnx.ModelProto:
    """Append the projection to a four-output model in-place and return it."""
    graph = model.graph

    if len(graph.output) != 1:
        raise ValueError(f"expected exactly one graph output, got {len(graph.output)}")
    out = graph.output[0]
    dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
    if len(dims) != 4 or dims[1] != 4:
        raise ValueError(f"expected four-source output [N,4,C,F], got {dims}")
    frames = dims[3]
    src_name = out.name

    taken = {t.name for t in graph.initializer} | {n.name for n in graph.node}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        taken.add(vi.name)
    for node in graph.node:
        taken.update(node.output)
    for name in (f"{PREFIX}stems", f"{PREFIX}vocals", f"{PREFIX}accompaniment"):
        if name in taken:
            raise ValueError(f"name collision: {name} already exists in the graph")

    # Slice parameters (axis 1 = source axis).
    axes = _int64_init(f"{PREFIX}axes", [1])
    graph.initializer.extend([
        axes,
        _int64_init(f"{PREFIX}s0", [0]), _int64_init(f"{PREFIX}e0", [1]),
        _int64_init(f"{PREFIX}s1", [1]), _int64_init(f"{PREFIX}e1", [2]),
        _int64_init(f"{PREFIX}s2", [2]), _int64_init(f"{PREFIX}e2", [3]),
        _int64_init(f"{PREFIX}s3", [3]), _int64_init(f"{PREFIX}e3", [4]),
    ])

    def slice_node(idx: int, out_name: str) -> onnx.NodeProto:
        return helper.make_node(
            "Slice",
            [src_name, f"{PREFIX}s{idx}", f"{PREFIX}e{idx}", f"{PREFIX}axes"],
            [out_name],
            name=f"{PREFIX}slice_{SOURCE_ORDER[idx]}",
        )

    graph.node.extend([
        slice_node(3, f"{PREFIX}vocals"),
        slice_node(0, f"{PREFIX}drums"),
        slice_node(1, f"{PREFIX}bass"),
        slice_node(2, f"{PREFIX}other"),
        helper.make_node(
            "Sum",
            [f"{PREFIX}drums", f"{PREFIX}bass", f"{PREFIX}other"],
            [f"{PREFIX}accompaniment"],
            name=f"{PREFIX}sum_accompaniment",
        ),
        helper.make_node(
            "Concat",
            [f"{PREFIX}vocals", f"{PREFIX}accompaniment"],
            [f"{PREFIX}stems"],
            axis=1,
            name=f"{PREFIX}concat_stems",
        ),
    ])

    # The old output becomes an internal tensor; the projection is the only output.
    del graph.output[:]
    graph.output.append(
        helper.make_tensor_value_info(
            f"{PREFIX}stems", TensorProto.FLOAT, [dims[0], 2, dims[2], frames]
        )
    )

    # Metadata: derivation identity for the catalog and the app cache key.
    meta = {p.key: p.value for p in model.metadata_props}
    source_cache_key = meta.get(MODEL_CACHE_KEY_METADATA, "")
    new_meta = dict(meta)
    if source_cache_key:
        new_meta[MODEL_CACHE_KEY_METADATA] = f"{source_cache_key}-karaoke2stem"
    new_meta["openkara.stem_profile"] = "karaoke-2stem"
    new_meta["openkara.output_semantics"] = f"[1,2,2,{frames}] vocals/accompaniment"
    new_meta["openkara.projection"] = PROJECTION_SPEC
    new_meta["openkara.derived_from_sha256"] = source_sha256
    del model.metadata_props[:]
    for k in sorted(new_meta):
        model.metadata_props.add(key=k, value=new_meta[k])

    return model


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 1

    source_sha = _sha256(args.input)
    model = onnx.load(str(args.input), load_external_data=True)
    model = derive_karaoke_2stem(model, source_sha)
    onnx.save(model, str(args.output))

    out_dims = [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]
    print(f"OK: {args.input.name} -> {args.output.name}")
    print(f"  projection: {PROJECTION_SPEC}")
    print(f"  output: {model.graph.output[0].name} {out_dims}")
    print(f"  derived_from_sha256: {source_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
