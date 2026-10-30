#!/usr/bin/env python3
"""Catalog model interfaces for native runtime measurement.

The native C API harness accepts any fixed-shape float32 model: it reads one
``<input-name>.f32`` file per session input and writes ``output_<i>.f32`` in
session output order. This module turns a catalog model entry plus the ONNX
graph interface into the exact feed tensors and expected output shapes for the
three shipped interfaces:

- ``spectral-core``: contract inputs ``spectral`` + ``mix``, outputs
  ``spectral_out`` + ``time_out``;
- ``waveform``: one ``[1, 2, 343980]`` stereo input, one
  ``[1, 4, 2, 343980]`` four-stem output;
- ``waveform`` with ``stem_profile: dual-output``: the same input, outputs
  ``outputs[0]=[1,4,2,343980]`` and ``outputs[1]=[1,2,2,343980]``.

Only catalog-declared shapes are accepted. Nothing here executes a Python
runtime wheel; measurement happens through ``native_quality.NativeRuntime``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SPECTRAL_INPUT_NAMES = frozenset({"spectral", "mix"})
SAMPLE_RATE = 44100

_SHAPE_RE = re.compile(r"\[\s*[0-9]+\s*(?:,\s*[0-9]+\s*)*\]")


class InterfaceError(ValueError):
    """The catalog model entry and the ONNX graph do not describe a supported interface."""


@dataclass(frozen=True)
class MeasurementSpec:
    interface: str
    inputs: dict[str, Any]
    output_shapes: list[list[int]]

    @property
    def output_shape_string(self) -> str:
        return ";".join(
            "[" + ",".join(str(dim) for dim in shape) + "]" for shape in self.output_shapes
        )


def parse_shape_literal(raw: str) -> list[int]:
    """Parse ``[1,4,2,343980]`` (brackets optional) into positive dimensions."""
    text = raw.strip()
    if text.startswith("["):
        if not text.endswith("]"):
            raise InterfaceError(f"malformed shape: {raw!r}")
        text = text[1:-1]
    parts = [part.strip() for part in text.split(",")]
    if not parts or any(not part.isdigit() for part in parts):
        raise InterfaceError(f"malformed shape: {raw!r}")
    dims = [int(part) for part in parts]
    if any(dim <= 0 for dim in dims):
        raise InterfaceError(f"shape dimensions must be positive: {raw!r}")
    return dims


def _single_shape(output_semantics: str) -> list[int]:
    match = _SHAPE_RE.search(output_semantics)
    if match is None:
        raise InterfaceError(f"no output shape in model semantics: {output_semantics!r}")
    return parse_shape_literal(match.group(0))


def parse_output_shapes(output_semantics: str, interface: str) -> list[list[int]]:
    """Return expected output shapes in session output order."""
    if interface == "spectral-core":
        named = dict(
            (name, parse_shape_literal(shape))
            for name, shape in re.findall(
                r"\b(spectral_out|time_out)\s*=\s*(\[[^;]*\])", output_semantics
            )
        )
        if set(named) != {"spectral_out", "time_out"}:
            raise InterfaceError(
                f"spectral-core output semantics must name spectral_out and time_out: "
                f"{output_semantics!r}"
            )
        return [named["spectral_out"], named["time_out"]]

    named = [
        (int(index), parse_shape_literal(shape))
        for index, shape in re.findall(
            r"outputs\[([0-9]+)\]\s*=\s*(\[[^;]*\])", output_semantics
        )
    ]
    if named:
        order = [index for index, _ in named]
        if order != list(range(len(named))):
            raise InterfaceError(
                f"output semantics indexes must be 0..N-1 in order: {output_semantics!r}"
            )
        return [shape for _, shape in named]

    return [_single_shape(output_semantics)]


def onnx_interface(model_path: Path) -> tuple[dict[str, list[int]], list[str], list[list[int]]]:
    """Return ``(input shapes, output names, expected output shapes)`` from ONNX.

    Requires static positive dims for every input and, where the graph carries
    them, for every output. External data referenced by the graph is loaded
    from the model's directory.
    """
    import onnx
    from onnx import TensorProto

    model = onnx.load(str(model_path))
    initializers = {init.name for init in model.graph.initializer}
    inputs: dict[str, list[int]] = {}
    for value in model.graph.input:
        if value.name in initializers:
            continue
        tensor = value.type.tensor_type
        if tensor.elem_type != TensorProto.FLOAT:
            raise InterfaceError(f"input {value.name!r} is not float32")
        dims = []
        for dim in tensor.shape.dim:
            if dim.dim_param or not dim.HasField("dim_value") or dim.dim_value <= 0:
                raise InterfaceError(f"input {value.name!r} must have static positive dims")
            dims.append(int(dim.dim_value))
        inputs[value.name] = dims

    outputs: list[str] = []
    output_shapes: list[list[int]] = []
    for value in model.graph.output:
        outputs.append(value.name)
        tensor = value.type.tensor_type
        dims = []
        for dim in tensor.shape.dim:
            if dim.dim_param or not dim.HasField("dim_value") or dim.dim_value <= 0:
                dims = []
                break
            dims.append(int(dim.dim_value))
        output_shapes.append(dims)
    return inputs, outputs, output_shapes


def measurement_spec(model: dict, model_path: Path, frames: int) -> MeasurementSpec:
    """Build the deterministic feed and expected output shapes for one model.

    ``model`` is the catalog ``model`` block. ``model_path`` is the single
    extracted ONNX graph whose bytes are measured.
    """
    if model.get("sample_rate") != SAMPLE_RATE:
        raise InterfaceError(f"unsupported sample rate: {model.get('sample_rate')!r}")
    if model.get("segment_frames") != frames:
        raise InterfaceError(
            f"model segment_frames={model.get('segment_frames')!r} != measurement frames={frames}"
        )
    interface = model.get("tensor_interface")
    semantics = model.get("output_semantics")
    if not isinstance(semantics, str):
        raise InterfaceError("model output_semantics is required")
    expected = parse_output_shapes(semantics, interface)

    graph_inputs, graph_outputs, graph_output_shapes = onnx_interface(model_path)
    if len(graph_outputs) != len(expected):
        raise InterfaceError(
            f"graph has {len(graph_outputs)} outputs but semantics declares {len(expected)}"
        )
    for name, declared, actual in zip(graph_outputs, expected, graph_output_shapes):
        if actual and actual != declared:
            raise InterfaceError(
                f"output {name!r} shape {actual} != catalog semantics {declared}"
            )

    from runtime_inputs import deterministic_waveform

    wave = deterministic_waveform(frames)  # [1, 2, frames]

    if interface == "spectral-core":
        if set(graph_inputs) != SPECTRAL_INPUT_NAMES:
            raise InterfaceError(
                f"spectral-core graph inputs {sorted(graph_inputs)} != "
                f"{sorted(SPECTRAL_INPUT_NAMES)}"
            )
        if graph_inputs["mix"] != [1, 2, frames]:
            raise InterfaceError(f"spectral-core mix shape {graph_inputs['mix']} is not [1,2,{frames}]")
        import spectral_reference

        spectral = spectral_reference.spec(wave.astype("float64")).astype("float32")
        if list(spectral.shape) != [1, 2, 2, 2048, 336]:
            raise InterfaceError(f"unexpected synthesized spectral shape {spectral.shape}")
        return MeasurementSpec(
            interface=interface,
            inputs={"spectral": spectral, "mix": wave},
            output_shapes=expected,
        )

    if interface == "waveform":
        if len(graph_inputs) != 1:
            raise InterfaceError(f"waveform interface requires exactly one input, got {sorted(graph_inputs)}")
        name, shape = next(iter(graph_inputs.items()))
        if shape != [1, 2, frames]:
            raise InterfaceError(f"waveform input {name!r} shape {shape} is not [1,2,{frames}]")
        return MeasurementSpec(interface=interface, inputs={name: wave}, output_shapes=expected)

    raise InterfaceError(f"unsupported tensor_interface: {interface!r}")
