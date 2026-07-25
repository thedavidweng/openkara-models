#!/usr/bin/env python3
"""Deduplicate byte-identical initializers in an ONNX graph (issue #22).

The htdemucs_ft ensemble merges four sub-models that each carry their own
copy of the constant DFT filter matrices (~33.6 MB each) and other identical
constants — ~322 MB of redundant bytes in the published 1.42 GB artifact.
Content-identical initializers (same dtype, shape, and bytes) are merged
into one, and every node input is rewired to the surviving name.

The transform is exactly lossless: initializer VALUES are unchanged, only
duplicate storage is removed, so runtime outputs are bit-identical.

Deterministic: the survivor of each duplicate group is the lexicographically
smallest name; repeated runs produce byte-identical output.

Usage::

    python scripts/dedupe_onnx_initializers.py --input model.onnx --output model.dedup.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import onnx


def _content_key(init: onnx.TensorProto) -> str:
    arr = onnx.numpy_helper.to_array(init)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def _rewire_graph(graph: onnx.GraphProto, rename: dict[str, str]) -> int:
    """Rewrite node inputs (recursing into subgraphs) per the rename map."""
    n_rewired = 0
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name in rename:
                node.input[i] = rename[name]
                n_rewired += 1
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                n_rewired += _rewire_graph(attr.g, rename)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for sub in attr.graphs:
                    n_rewired += _rewire_graph(sub, rename)
    return n_rewired


def dedupe_initializers(model: onnx.ModelProto) -> dict[str, int]:
    """Merge content-identical initializers in-place. Returns stats."""
    graph = model.graph
    groups: dict[str, list[onnx.TensorProto]] = {}
    for init in graph.initializer:
        groups.setdefault(_content_key(init), []).append(init)

    rename: dict[str, str] = {}
    bytes_saved = 0
    for inits in groups.values():
        if len(inits) < 2:
            continue
        survivor = min(inits, key=lambda t: t.name)
        arr_bytes = onnx.numpy_helper.to_array(survivor).nbytes
        for t in inits:
            if t.name != survivor.name:
                rename[t.name] = survivor.name
                bytes_saved += arr_bytes

    if rename:
        n_rewired = _rewire_graph(graph, rename)
        survivors = [t for t in graph.initializer if t.name not in rename]
        del graph.initializer[:]
        graph.initializer.extend(survivors)
        # Drop stale value_info / graph.input entries for removed names.
        for field in (graph.value_info, graph.input):
            keep = [vi for vi in field if vi.name not in rename]
            if len(keep) != len(field):
                del field[:]
                field.extend(keep)
    else:
        n_rewired = 0

    return {
        "duplicate_groups": sum(1 for v in groups.values() if len(v) > 1),
        "initializers_removed": len(rename),
        "inputs_rewired": n_rewired,
        "bytes_saved": bytes_saved,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 1

    model = onnx.load(str(args.input), load_external_data=True)
    stats = dedupe_initializers(model)
    onnx.save(model, str(args.output))

    in_size = args.input.stat().st_size
    out_size = args.output.stat().st_size
    print(f"OK: {args.input.name} -> {args.output.name}")
    print(f"  duplicate groups:     {stats['duplicate_groups']}")
    print(f"  initializers removed: {stats['initializers_removed']}")
    print(f"  inputs rewired:       {stats['inputs_rewired']}")
    print(f"  bytes: {in_size:,} -> {out_size:,} "
          f"({(in_size - out_size) / in_size * 100.0:.1f}% smaller)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
