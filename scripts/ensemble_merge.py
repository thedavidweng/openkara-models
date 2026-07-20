"""Pure ONNX ensemble graph merge — no torch dependency.

build_ensemble_graph takes in-memory onnx.ModelProto sub-models and returns
an in-memory ensemble onnx.ModelProto. This module is separated from
convert_htdemucs_to_onnx.py so it can be unit-tested with only the onnx
package, without torch or demucs.
"""

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference


def build_ensemble_graph(sub_models, n_models):
    """Build a single ensemble ONNX graph from sub-model graphs.

    Pure function: takes in-memory onnx.ModelProto list, returns an in-memory
    ensemble onnx.ModelProto. Feeds the same input to all sub-models, averages
    their outputs, and namespaces each sub-model's nodes/initializers with a
    prefix to avoid collisions. Runs shape inference before returning.
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

    all_nodes.append(helper.make_node(
        "ReduceMean",
        inputs=[concat_output],
        outputs=["stems"],
        axes=[0],
        keepdims=0,
    ))

    axes_0 = numpy_helper.from_array(np.array([0], dtype=np.int64), name="unsqueeze_axes")
    all_initializers.append(axes_0)

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
