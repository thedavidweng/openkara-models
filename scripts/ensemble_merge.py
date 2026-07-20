"""Pure ONNX ensemble graph merge — no torch dependency.

build_ensemble_graph takes in-memory onnx.ModelProto sub-models and returns
an in-memory ensemble onnx.ModelProto. This module is separated from
convert_htdemucs_to_onnx.py so it can be unit-tested with only the onnx
package, without torch or demucs.

Two optimizations beyond naive concatenation:

1. **Initializer deduplication** — sub-models share identical STFT/ISTFT
   filter banks (same n_fft/hop_length). After prefixing, these become
   distinct initializers with identical content. We hash tensor content
   and collapse duplicates to a single shared initializer, reducing model
   size and improving runtime cache locality.

2. **Sum + Mul averaging** — instead of Unsqueeze→Concat→ReduceMean (which
   materializes a stacked [N, ...] intermediate tensor), we use a single
   Sum node followed by Mul by 1/N. Fewer nodes, no intermediate tensor,
   identical numerical result.
"""

import hashlib

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference


def _tensor_content_hash(init: TensorProto) -> str:
    """Stable hash of tensor data + shape + dtype, excluding the name field."""
    arr = numpy_helper.to_array(init)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(np.array(arr.shape, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _deduplicate_initializers(nodes, initializers):
    """Collapse identical initializers to a single shared copy.

    Returns (nodes_with_renamed_inputs, deduped_initializers).
    """
    canonical: dict[str, str] = {}   # content_hash -> canonical name
    rename: dict[str, str] = {}      # old name -> canonical name

    for init in initializers:
        ch = _tensor_content_hash(init)
        if ch in canonical:
            rename[init.name] = canonical[ch]
        else:
            canonical[ch] = init.name

    for node in nodes:
        for i, inp in enumerate(node.input):
            if inp in rename:
                node.input[i] = rename[inp]

    deduped = [init for init in initializers if init.name not in rename]
    return nodes, deduped


def build_ensemble_graph(sub_models, n_models):
    """Build a single ensemble ONNX graph from sub-model graphs.

    Pure function: takes in-memory onnx.ModelProto list, returns an in-memory
    ensemble onnx.ModelProto. Feeds the same input to all sub-models, averages
    their outputs via Sum+Mul, namespaces each sub-model's nodes/initializers
    with a prefix to avoid collisions, deduplicates identical initializers
    (shared STFT/ISTFT filter banks), and runs shape inference before returning.
    """
    ref = sub_models[0]
    ref_input = ref.graph.input[0]
    ref_output = ref.graph.output[0]

    all_initializers = []
    all_nodes = []
    sub_output_names = []

    for i, m in enumerate(sub_models):
        prefix = f"sub{i}_"

        input_name = m.graph.input[0].name
        output_name = m.graph.output[0].name
        prefixed_output = prefix + output_name
        sub_output_names.append(prefixed_output)

        def rename(name):
            if not name:
                return ""
            if name == input_name:
                return "audio"
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

    # Average via Sum + Mul — no intermediate stacked tensor.
    all_nodes.append(helper.make_node(
        "Sum",
        inputs=sub_output_names,
        outputs=["ensemble_sum"],
    ))
    all_nodes.append(helper.make_node(
        "Mul",
        inputs=["ensemble_sum", "ensemble_scale"],
        outputs=["stems"],
    ))

    scale = numpy_helper.from_array(
        np.array([1.0 / n_models], dtype=np.float32), name="ensemble_scale"
    )
    all_initializers.append(scale)

    # Deduplicate identical initializers (shared STFT/ISTFT filter banks).
    all_nodes, all_initializers = _deduplicate_initializers(all_nodes, all_initializers)

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

    opset_imports = [helper.make_opsetid(op.domain, op.version) for op in ref.opset_import]

    ensemble_model = helper.make_model(graph, opset_imports=opset_imports)
    ensemble_model.ir_version = ref.ir_version

    ensemble_model = shape_inference.infer_shapes(ensemble_model)
    return ensemble_model
