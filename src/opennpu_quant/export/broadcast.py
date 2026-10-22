import copy
from collections import Counter
import numpy as np
import onnx
from onnx import helper, numpy_helper
from ..graph.partition import extract


def replace_parameter_tiles(source, *, expected=None):
    model = onnx.shape_inference.infer_shapes(
        copy.deepcopy(source), strict_mode=True, data_prop=True
    )
    nodes = list(model.graph.node)
    producers = {x: n for n in nodes for x in n.output}
    arrays = {x.name: numpy_helper.to_array(x) for x in model.graph.initializer}
    for n in nodes:
        if (
            n.op_type == "Constant"
            and len(n.attribute) == 1
            and n.attribute[0].name == "value"
        ):
            arrays[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    types = {
        v.name: v.type.tensor_type
        for v in [*model.graph.input, *model.graph.value_info, *model.graph.output]
    }

    def attrs(node):
        return {a.name: helper.get_attribute_value(a) for a in node.attribute}

    def vector(name):
        if name in arrays:
            a = arrays[name]
            if a.dtype != np.int64 or a.ndim > 1:
                raise ValueError("Expected int64 shape value")
            return list(map(int, a.reshape(-1)))
        n = producers.get(name)
        if n is None:
            raise ValueError("Unresolved repeat expression")
        at = attrs(n)
        if n.op_type == "Shape" and not n.attribute:
            t = types[n.input[0]]
            if (
                t.elem_type == onnx.TensorProto.INT64
                and len(t.shape.dim) == 1
                and t.shape.dim[0].HasField("dim_value")
            ):
                return [t.shape.dim[0].dim_value]
            return [("dim", n.input[0], i) for i in range(len(t.shape.dim))]
        if n.op_type == "Gather" and at.get("axis", 0) == 0:
            value = vector(n.input[0])
            indices = vector(n.input[1])
            return [value[i] for i in indices]
        if n.op_type == "Unsqueeze" and vector(n.input[1]) == [0]:
            return vector(n.input[0])
        if n.op_type == "Concat" and at.get("axis") == 0:
            return [x for name in n.input for x in vector(name)]
        if n.op_type == "ConstantOfShape":
            shape = vector(n.input[0])
            value = numpy_helper.to_array(at["value"])
            if (
                shape != [4]
                or value.dtype != np.int64
                or value.size != 1
                or int(value[0]) != 1
            ):
                raise ValueError("Unexpected parameter Expand shape")
            return [1] * 4
        raise ValueError("Unsupported repeat expression: " + n.op_type)

    changes = []
    for node in nodes:
        if node.op_type != "Tile":
            continue
        expand = producers.get(node.input[0])
        if (
            expand is None
            or expand.op_type != "Expand"
            or expand.input[0] not in arrays
        ):
            continue
        parameter = expand.input[0]
        value = arrays[parameter]
        if (
            value.dtype != np.float32
            or value.ndim != 4
            or value.shape[0] != 1
            or value.shape[2:] != (1, 1)
        ):
            continue
        consumers = [n for n in nodes if node.output[0] in n.input]
        if len(consumers) != 1 or consumers[0].op_type not in ("Add", "Mul"):
            raise ValueError("Parameter Tile has unsupported fan-out or consumer")
        consumer = consumers[0]
        position = list(consumer.input).index(node.output[0])
        feature = consumer.input[1 - position]
        t = types[feature]
        if (
            t.elem_type != onnx.TensorProto.FLOAT
            or len(t.shape.dim) != 4
            or not t.shape.dim[1].HasField("dim_value")
            or t.shape.dim[1].dim_value != value.shape[1]
        ):
            raise ValueError("Parameter/feature channel contract differs")
        if vector(expand.input[1]) != [1] * 4 or vector(node.input[1]) != [
            1,
            1,
            ("dim", feature, 2),
            ("dim", feature, 3),
        ]:
            raise ValueError(
                "Tile repeats are not the consumer feature spatial dimensions"
            )
        consumer.input[position] = parameter
        changes.append(
            dict(
                tile=node.name,
                parameter=parameter,
                shape=list(value.shape),
                consumer=consumer.name,
                operation=consumer.op_type,
                input_position=position,
                feature=feature,
                output=consumer.output[0],
            )
        )
    counts = dict(Counter(c["operation"] for c in changes))
    if expected is not None and counts != expected:
        raise ValueError("Unexpected parameter Tile counts: " + str(counts))
    result = extract(
        model, [v.name for v in model.graph.input], [v.name for v in model.graph.output]
    )
    onnx.checker.check_model(result, full_check=True)
    return result, dict(
        kind="parameter_repeat_to_broadcast",
        changes=changes,
        counts=counts,
        contract="float32 [1,C,1,1], Expand identity, repeats [1,1,H,W] from the same feature; operands and initializer values preserved",
    )
