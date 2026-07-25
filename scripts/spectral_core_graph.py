"""Pure-ONNX helpers for spectral-core graphs (issue #23 PR 2) — no torch.

Separated from scripts/export_spectral_core.py so the graph surgery and the
contract gate can be unit-tested with only the onnx package (mirrors
scripts/ensemble_merge.py). Three responsibilities:

1. ``strip_time_zero_add`` — the trace patch makes ``_ispec`` return a scalar
   zero so the model's ``x = xt + x`` reduces to the time branch; this
   removes that residual Add-with-zero from the exported graph.
2. ``assert_spectral_core_graph`` / ``assert_spectral_core_metadata`` — the
   structural contract gate: exact tensor interface, no transform ops, no
   dense DFT filter-bank initializer signatures, contract version metadata.
3. ``build_spectral_core_ensemble`` — multi-input/multi-output ensemble
   merge for htdemucs_ft spectral cores. ISTFT is linear, so averaging the
   pre-ISTFT and time-branch outputs is exactly equivalent to averaging the
   sub-models' waveforms (what scripts/ensemble_merge.py ships today).

Contract: docs/spectral-core-contract.md (``openkara.spectral-contract/v1``).
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

from ensemble_merge import _deduplicate_initializers

SPECTRAL_CONTRACT_VERSION = "openkara.spectral-contract/v1"
SPECTRAL_CONTRACT_METADATA = "openkara.spectral_contract"

N_FFT = 4096
N_BINS = N_FFT // 2 + 1          # 2049
CONTRACT_FREQS = N_FFT // 2      # 2048
SEGMENT_FRAMES = 343980
SEGMENT_SPECTRAL_FRAMES = 336    # ceil(343980 / 1024)
SOURCES = 4                      # drums, bass, other, vocals

# Exact public tensor interface, in order (order is part of the contract:
# applications address outputs positionally).
CONTRACT_INPUTS = (
    ("spectral", (1, 2, 2, CONTRACT_FREQS, SEGMENT_SPECTRAL_FRAMES)),
    ("mix", (1, 2, SEGMENT_FRAMES)),
)
CONTRACT_OUTPUTS = (
    ("spectral_out", (1, SOURCES, 2, 2, CONTRACT_FREQS, SEGMENT_SPECTRAL_FRAMES)),
    ("time_out", (1, SOURCES, 2, SEGMENT_FRAMES)),
)

# ONNX signal/transform operators that must never appear in a spectral-core
# graph — the whole point is that the transform lives in the application.
FORBIDDEN_TRANSFORM_OPS = frozenset({
    "STFT",
    "DFT",
    "Col2Im",
    "HannWindow",
    "HammingWindow",
    "BlackmanWindow",
    "MelWeightMatrix",
})


class SpectralCoreContractError(AssertionError):
    """A graph violates the spectral-core tensor contract."""


def _iter_graphs(graph):
    """Yield graph and every nested subgraph (If/Loop/Scan bodies)."""
    yield graph
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                yield from _iter_graphs(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for g in attr.graphs:
                    yield from _iter_graphs(g)


def _tensor_dims(value_info):
    dims = []
    for d in value_info.type.tensor_type.shape.dim:
        if d.dim_param or not d.HasField("dim_value"):
            raise SpectralCoreContractError(
                f"tensor {value_info.name!r} has a symbolic dimension; the "
                "spectral-core interface is fixed-shape"
            )
        dims.append(d.dim_value)
    return tuple(dims)


def _operand_is_all_zero(graph, name):
    """True if ``name`` is a constant whose every element is zero.

    Covers the three constant encodings the TorchScript exporter emits:
    an initializer, a ``Constant`` node, or a ``ConstantOfShape`` node
    (``torch.Tensor.new_zeros`` traces to the latter; its output is all
    zero whenever the ``value`` attribute is absent or zero, regardless
    of the runtime shape input).
    """
    for init in graph.initializer:
        if init.name == name:
            arr = numpy_helper.to_array(init)
            return bool(arr.size) and not arr.any()
    for node in graph.node:
        if name not in node.output:
            continue
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    arr = numpy_helper.to_array(attr.t)
                    return bool(arr.size) and not arr.any()
                if attr.name == "value_float":
                    return attr.f == 0.0
                if attr.name == "value_int":
                    return attr.i == 0
            return False
        if node.op_type == "ConstantOfShape":
            for attr in node.attribute:
                if attr.name == "value":
                    arr = numpy_helper.to_array(attr.t)
                    return not arr.any()
            return True  # default fill value is float32 zero
        return False
    return False


def _prune_zero_cone(graph, seeds):
    """Remove the now-dead constant chain behind the stripped zero operand.

    Bottom-up fixpoint restricted to the ancestor cone of ``seeds``: a node
    or initializer is removed only if it is (transitively) behind a seed and
    no surviving node consumes it. Shared constants with remaining consumers
    are kept.
    """
    seeds = set(seeds)
    changed = True
    while changed:
        changed = False
        consumed = {name for node in graph.node for name in node.input if name}
        consumed |= {vi.name for vi in graph.output}
        for idx in reversed(range(len(graph.initializer))):
            init = graph.initializer[idx]
            if init.name in seeds and init.name not in consumed:
                del graph.initializer[idx]
                changed = True
        for idx in reversed(range(len(graph.node))):
            node = graph.node[idx]
            if any(o in seeds for o in node.output) and \
                    not any(o in consumed for o in node.output):
                seeds.update(n for n in node.input if n)
                del graph.node[idx]
                changed = True


def strip_time_zero_add(model: onnx.ModelProto) -> onnx.ModelProto:
    """Remove the ``time_out = Add(time_branch, 0)`` residual in place.

    The trace patch returns a scalar zero from ``_ispec`` so the traced
    ``x = xt + x`` is an Add with an all-zero constant operand. This function
    deletes that Add (and its zero constant), renames the surviving operand
    to ``time_out`` throughout the graph, and fails loudly if the expected
    pattern is absent — a missing pattern means the trace patch changed and
    the exporter must be revisited, not silently shipped.
    """
    graph = model.graph
    time_name = CONTRACT_OUTPUTS[1][0]

    producer = None
    producer_idx = None
    for idx, node in enumerate(graph.node):
        if time_name in node.output:
            producer = node
            producer_idx = idx
            break
    if producer is None:
        raise SpectralCoreContractError(
            f"no node produces graph output {time_name!r}"
        )
    if producer.op_type != "Add" or len(producer.input) != 2:
        raise SpectralCoreContractError(
            f"expected {time_name!r} to be produced by the traced "
            f"Add-with-zero, found {producer.op_type!r}"
        )

    survivor = None
    zero_name = None
    for candidate, other in ((producer.input[0], producer.input[1]),
                             (producer.input[1], producer.input[0])):
        if _operand_is_all_zero(graph, candidate):
            survivor, zero_name = other, candidate
            break
    if survivor is None:
        raise SpectralCoreContractError(
            f"neither operand of the {time_name!r} Add is an all-zero "
            "constant; refusing to strip a semantic Add"
        )

    del graph.node[producer_idx]
    _prune_zero_cone(graph, {zero_name})

    # The surviving tensor takes over the graph-output name everywhere.
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name == survivor:
                node.input[i] = time_name
        for i, name in enumerate(node.output):
            if name == survivor:
                node.output[i] = time_name
    for vi in graph.value_info:
        if vi.name == survivor:
            vi.name = time_name

    return model


def assert_spectral_core_graph(model: onnx.ModelProto) -> None:
    """Structural contract gate for a spectral-core graph.

    Rejects transform nodes and dense DFT filter-bank initializer signatures
    (issue #23 PR 2), and pins the exact public tensor interface.
    """
    graph = model.graph

    # 1. Exact tensor interface, positional.
    actual_inputs = [(vi.name, _tensor_dims(vi)) for vi in graph.input]
    actual_outputs = [(vi.name, _tensor_dims(vi)) for vi in graph.output]
    if actual_inputs != [(n, tuple(s)) for n, s in CONTRACT_INPUTS]:
        raise SpectralCoreContractError(
            f"graph inputs {actual_inputs} do not match the spectral-core "
            f"contract {list(CONTRACT_INPUTS)}"
        )
    if actual_outputs != [(n, tuple(s)) for n, s in CONTRACT_OUTPUTS]:
        raise SpectralCoreContractError(
            f"graph outputs {actual_outputs} do not match the spectral-core "
            f"contract {list(CONTRACT_OUTPUTS)}"
        )
    for vi in list(graph.input) + list(graph.output):
        elem = vi.type.tensor_type.elem_type
        if elem != TensorProto.FLOAT:
            raise SpectralCoreContractError(
                f"tensor {vi.name!r} must be float32, got elem_type={elem}"
            )

    # 2. No transform ops anywhere (including subgraphs).
    for g in _iter_graphs(graph):
        for node in g.node:
            if node.op_type in FORBIDDEN_TRANSFORM_OPS:
                raise SpectralCoreContractError(
                    f"forbidden transform op {node.op_type!r} "
                    f"(node {node.name!r}) in spectral-core graph"
                )

    # 3. No dense DFT filter-bank signatures. The conv-DFT banks are the
    # only tensors in this model family pairing the FFT size with a
    # one-sided bin count — e.g. (2049, 1, 4096) / (1, 2049, 4096).
    def check_tensor(t, where):
        dims = set(t.dims)
        if N_FFT in dims and (N_BINS in dims or CONTRACT_FREQS in dims):
            raise SpectralCoreContractError(
                f"dense DFT filter-bank signature {tuple(t.dims)} in "
                f"{where}: {t.name!r}"
            )

    for g in _iter_graphs(graph):
        for init in g.initializer:
            check_tensor(init, "initializer")
        for node in g.node:
            if node.op_type == "Constant":
                for attr in node.attribute:
                    if attr.name == "value":
                        check_tensor(attr.t, f"Constant node {node.name!r}")


def assert_spectral_core_metadata(model: onnx.ModelProto) -> None:
    """The artifact must carry the contract version in model metadata."""
    meta = {p.key: p.value for p in model.metadata_props}
    version = meta.get(SPECTRAL_CONTRACT_METADATA)
    if version != SPECTRAL_CONTRACT_VERSION:
        raise SpectralCoreContractError(
            f"model metadata {SPECTRAL_CONTRACT_METADATA!r} is {version!r}, "
            f"expected {SPECTRAL_CONTRACT_VERSION!r}"
        )


def build_spectral_core_ensemble(sub_models, n_models,
                                 graph_name="htdemucs_ft_spectral_core_ensemble"):
    """Merge spectral-core sub-models into one averaging ensemble graph.

    Multi-I/O counterpart of scripts/ensemble_merge.py::build_ensemble_graph:
    the contract inputs are fed to every sub-model unchanged, each contract
    output is averaged across sub-models via Sum + Mul (no stacked
    intermediate), all other tensors are namespaced per sub-model, and
    identical initializers are deduplicated. Averaging the pre-ISTFT and
    time-branch tensors is exactly the waveform average by linearity of the
    ISTFT and the final crop.
    """
    if len(sub_models) != n_models:
        raise ValueError(f"expected {n_models} sub-models, got {len(sub_models)}")

    input_names = {name for name, _ in CONTRACT_INPUTS}
    output_names = [name for name, _ in CONTRACT_OUTPUTS]

    ref = sub_models[0]
    for i, m in enumerate(sub_models):
        actual_in = [vi.name for vi in m.graph.input]
        actual_out = [vi.name for vi in m.graph.output]
        if actual_in != [n for n, _ in CONTRACT_INPUTS] or \
                actual_out != output_names:
            raise SpectralCoreContractError(
                f"sub-model {i} interface ({actual_in} -> {actual_out}) does "
                "not match the spectral-core contract"
            )

    all_initializers = []
    all_nodes = []
    # output name -> list of per-sub prefixed tensor names
    sub_outputs = {name: [] for name in output_names}

    for i, m in enumerate(sub_models):
        prefix = f"sub{i}_"

        def rename(name):
            if not name:
                return ""
            if name in input_names:
                return name
            return prefix + name

        for init in m.graph.initializer:
            new_init = TensorProto()
            new_init.CopyFrom(init)
            new_init.name = rename(init.name)
            all_initializers.append(new_init)

        for node in m.graph.node:
            new_node = helper.make_node(
                node.op_type,
                inputs=[rename(n) for n in node.input],
                outputs=[rename(n) for n in node.output],
                name=prefix + node.name if node.name else "",
                domain=node.domain,
            )
            for attr in node.attribute:
                new_node.attribute.append(attr)
            all_nodes.append(new_node)

        for name in output_names:
            sub_outputs[name].append(prefix + name)

    scale = numpy_helper.from_array(
        np.array([1.0 / n_models], dtype=np.float32), name="ensemble_scale"
    )
    all_initializers.append(scale)

    for name in output_names:
        all_nodes.append(helper.make_node(
            "Sum",
            inputs=sub_outputs[name],
            outputs=[f"ensemble_sum_{name}"],
        ))
        all_nodes.append(helper.make_node(
            "Mul",
            inputs=[f"ensemble_sum_{name}", "ensemble_scale"],
            outputs=[name],
        ))

    all_nodes, all_initializers = _deduplicate_initializers(
        all_nodes, all_initializers
    )

    def value_info(name, shape):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))

    graph = helper.make_graph(
        all_nodes,
        graph_name,
        [value_info(n, s) for n, s in CONTRACT_INPUTS],
        [value_info(n, s) for n, s in CONTRACT_OUTPUTS],
        initializer=all_initializers,
    )

    opset_imports = [
        helper.make_opsetid(op.domain, op.version) for op in ref.opset_import
    ]
    ensemble_model = helper.make_model(graph, opset_imports=opset_imports)
    ensemble_model.ir_version = ref.ir_version

    ensemble_model = shape_inference.infer_shapes(ensemble_model)
    return ensemble_model
