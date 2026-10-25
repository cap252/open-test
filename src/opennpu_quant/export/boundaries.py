import copy
import numpy as np
import onnx
from onnx import numpy_helper
from ..graph.partition import extract
from .._io import object_hash


def split_detector(model, mid):
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    nodes = list(model.graph.node)
    initializers = {x.name: numpy_helper.to_array(x) for x in model.graph.initializer}
    ignored = set()
    if mid in ("yolov9_t", "yolov9_s", "yolov8n"):
        for i, node in enumerate(nodes):
            if node.op_type == "Conv" and node.input[1] in initializers:
                w = initializers[node.input[1]]
                if w.shape == (1, 16, 1, 1) and np.array_equal(
                    w.ravel(), np.arange(16, dtype=w.dtype)
                ):
                    ignored.add(i)
        if len(ignored) != 1:
            raise ValueError("Expected exactly one original DFL integral Conv")
    consumers = {}
    for i, node in enumerate(nodes):
        for x in node.input:
            consumers.setdefault(x, []).append(i)

    def descendant_conv(i):
        pending = [j for x in nodes[i].output for j in consumers.get(x, [])]
        seen = set()
        while pending:
            j = pending.pop()
            if j in seen:
                continue
            seen.add(j)
            if nodes[j].op_type == "Conv" and j not in ignored:
                return True
            pending.extend(k for x in nodes[j].output for k in consumers.get(x, []))
        return False

    terminal = [
        n.output[0]
        for i, n in enumerate(nodes)
        if n.op_type == "Conv" and i not in ignored and not descendant_conv(i)
    ]
    if mid == "yolov5s_seg":
        producer = {x: i for i, n in enumerate(nodes) for x in n.output}
        ancestors = set()
        pending = ["prototypes"]
        while pending:
            x = pending.pop()
            i = producer.get(x)
            if i is None or i in ancestors:
                continue
            ancestors.add(i)
            pending.extend(nodes[i].input)
        proto = [x for x in terminal if producer[x] in ancestors]
        if len(proto) != 1:
            raise ValueError("Ambiguous prototype output boundary")
        terminal = [x for x in terminal if x not in proto] + ["prototypes"]
    expected = {
        "nanodet_plus_m_320": 4,
        "posenet_101": 4,
        "retinanet_resnet50_fpn": 10,
        "yolov9_t": 6,
        "yolov9_s": 6,
        "yolov8n": 6,
        "yolox": 9,
        "yolov5s_seg": 4,
        "yolov3_full": 3,
        "yolov4_full": 3,
        "yolov7_full": 3,
    }
    if mid in expected and len(terminal) != expected[mid]:
        raise ValueError("Ambiguous terminal prediction heads: " + str(terminal))
    if mid == "yolov7_face" and len(terminal) not in (3, 6):
        raise ValueError("Unexpected tiny face head structure")
    learned_scales = []
    if mid == "yolov7_face":
        replaced = []
        for output in terminal:
            candidates = []
            for i in consumers.get(output, []):
                node = nodes[i]
                if node.op_type != "Mul":
                    continue
                other = node.input[1 - list(node.input).index(output)]
                value = initializers.get(other)
                if (
                    value is not None
                    and value.ndim == 4
                    and value.shape[0] == 1
                    and value.shape[2:] == (1, 1)
                ):
                    candidates.append(node)
            if len(candidates) > 1:
                raise ValueError("Ambiguous learned output scale")
            if candidates:
                scale = candidates[0]
                replaced.append(scale.output[0])
                learned_scales.append(
                    dict(
                        node=scale.name,
                        conv_output=output,
                        network_output=scale.output[0],
                    )
                )
            else:
                replaced.append(output)
        if len(learned_scales) != 3:
            raise ValueError("Expected three learned prediction scales")
        terminal = replaced
    inputs = [v.name for v in model.graph.input]
    outputs = [v.name for v in model.graph.output]
    post = extract(model, terminal + inputs, outputs)
    # Shape-only dependencies can reach upstream features even after all prediction
    # convolutions are cut. Keep the original integer Shape values as extra outputs.
    post_producers = {
        output: node for node in post.graph.node for output in node.output
    }
    ignored_names = {nodes[index].name for index in ignored}
    roots = set(terminal + inputs)

    def has_uncut_feature_conv(value, seen=None):
        seen = set() if seen is None else seen
        if value in roots or value in seen or value not in post_producers:
            return False
        seen.add(value)
        node = post_producers[value]
        if node.op_type == "Conv" and node.name not in ignored_names:
            return True
        return any(has_uncut_feature_conv(x, seen) for x in node.input)

    shape_outputs = [
        node.output[0]
        for node in post.graph.node
        if node.op_type == "Shape"
        and (
            mid in ("yolov9_t", "yolov7_face") or has_uncut_feature_conv(node.input[0])
        )
    ]
    terminal += shape_outputs
    network = extract(model, inputs, terminal)
    if any(n.name in ignored_names for n in network.graph.node):
        raise ValueError("Decoder DFL integral leaked into network")
    post = extract(model, terminal + inputs, outputs)
    postconv = [n for n in post.graph.node if n.op_type == "Conv"]
    if len(postconv) != len(ignored):
        raise ValueError("Feature Conv leaked across decoder boundary")
    evidence = dict(
        kind="terminal_predictor_heads",
        network_inputs=inputs,
        network_outputs=terminal,
        original_outputs=outputs,
        shape_outputs=shape_outputs,
        learned_output_scales=learned_scales,
        postprocess_inputs=[v.name for v in post.graph.input],
        decoder_conv_count=len(postconv),
        reason="Feature prediction convolutions remain in network; original reshape/anchor/decode graph runs on CPU",
        node_evidence=[
            dict(name=n.name, outputs=list(n.output), weights=n.input[1])
            for n in nodes
            if n.op_type == "Conv" and n.output[0] in terminal
        ],
    )
    return network, post, evidence


def materialize_tensor_constants(model):
    result = copy.deepcopy(model)
    kept = []
    converted = []
    names = {t.name for t in result.graph.initializer}
    for node in result.graph.node:
        if node.op_type != "Constant" or node.domain not in ("", "ai.onnx"):
            kept.append(node)
            continue
        if (
            node.input
            or len(node.output) != 1
            or len(node.attribute) != 1
            or node.attribute[0].name != "value"
            or node.attribute[0].type != onnx.AttributeProto.TENSOR
        ):
            raise ValueError("Unsupported Constant representation")
        name = node.output[0]
        if name in names:
            raise ValueError("Duplicate Constant initializer")
        value = copy.deepcopy(node.attribute[0].t)
        value.name = name
        result.graph.initializer.append(value)
        names.add(name)
        converted.append(
            dict(
                node=node.name,
                tensor=name,
                dtype=value.data_type,
                shape=list(value.dims),
                value_identity=object_hash(
                    [
                        value.data_type,
                        list(value.dims),
                        numpy_helper.to_array(value).tobytes().hex(),
                    ]
                ),
            )
        )
    del result.graph.node[:]
    result.graph.node.extend(kept)
    onnx.checker.check_model(result, full_check=True)
    return result, dict(
        kind="tensor_constant_to_initializer",
        converted=converted,
        reason="ORT materializes tensor Constants even with optimization disabled; preserve tensor bytes before QDQ audit",
    )


def fold_constant_floors(model):
    """Evaluate only initializer/Constant Floor inputs; never alter feature Floor."""
    result = copy.deepcopy(model)
    values = {t.name: numpy_helper.to_array(t) for t in result.graph.initializer}
    changes = []
    for node in result.graph.node:
        if (
            node.op_type == "Constant"
            and len(node.attribute) == 1
            and node.attribute[0].name == "value"
        ):
            values[node.output[0]] = numpy_helper.to_array(node.attribute[0].t)
        if node.op_type != "Floor" or node.input[0] not in values:
            continue
        value = values[node.input[0]]
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite constant Floor")
        folded = np.floor(value)
        changes.append(dict(node=node.name, input=node.input[0], output=node.output[0]))
        node.CopyFrom(
            onnx.helper.make_node(
                "Constant",
                [],
                list(node.output),
                name=node.name,
                value=numpy_helper.from_array(folded),
            )
        )
        values[node.output[0]] = folded
    onnx.checker.check_model(result, full_check=True)
    return result, dict(kind="constant_only_Floor_evaluation", changes=changes)


def fold_resize_sizes(model, floors):
    """Keep the released constant Cast sizes after fixed-shape Floor evaluation."""
    result = copy.deepcopy(model)
    values = {v.name: numpy_helper.to_array(v) for v in result.graph.initializer}
    descendants = {v["output"] for v in floors}
    changes = []
    for node in result.graph.node:
        if node.op_type == "Constant" and len(node.attribute) == 1:
            values[node.output[0]] = numpy_helper.to_array(node.attribute[0].t)
            continue
        if not node.input or any(v not in values for v in node.input):
            continue
        args = [values[v] for v in node.input]
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        if node.op_type == "Unsqueeze":
            value = np.expand_dims(args[0], tuple(int(x) for x in args[1].flat))
        elif node.op_type == "Concat":
            value = np.concatenate(args, axis=attrs["axis"])
        elif node.op_type == "Cast":
            value = args[0].astype(onnx.helper.tensor_dtype_to_np_dtype(attrs["to"]))
        else:
            continue
        values[node.output[0]] = value
        if not any(v in descendants for v in node.input):
            continue
        descendants.update(node.output)
        if node.op_type == "Cast":
            if value.dtype != np.int64 or value.shape != (2,):
                raise ValueError("Unexpected fixed Resize size expression")
            changes.append(
                dict(node=node.name, output=node.output[0], size=value.tolist())
            )
            node.CopyFrom(
                onnx.helper.make_node(
                    "Constant",
                    [],
                    list(node.output),
                    name=node.name,
                    value=numpy_helper.from_array(value),
                )
            )
    if len(changes) != 2:
        raise ValueError("Expected two constant YOLOv3 Resize size expressions")
    return result, dict(changes=changes)
