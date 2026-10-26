import numpy as np
import onnx
from ..graph.partition import extract, rename_input
from ..ort.session import create_session
from ..ort.config import OrtConfig
from ..ort.fetch import run_tensors
from .contract import validate_parts
from .parity import framework_compare as framework_compare, validate_framework


def compare(expected, actual):
    if set(expected) != set(actual):
        raise ValueError("Output names changed")
    result = {}
    for name, a in expected.items():
        b = actual[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError("Output shape/dtype changed: " + name)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Nonfinite export output")
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5, err_msg=name)
        result[name] = (
            float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
            if a.size
            else 0.0
        )
    return result


def outputs(model, feeds):
    session = create_session(model, OrtConfig.cpu())
    names = [v.name for v in session.get_outputs()]
    return [dict(zip(names, run_tensors(session, names, feed))) for feed in feeds]


def finalize(raw, recipe, feeds, references, destination, *, validation="strict"):
    if validation not in ("strict", "functional"):
        raise ValueError("Unknown export validation mode")
    if not feeds or len(feeds) != len(references):
        raise ValueError("Export requires one framework reference per input fixture")
    model = onnx.load(raw)
    onnx.checker.check_model(model, full_check=True)
    raw_values = outputs(model, feeds)
    checks = [
        dict(framework_raw=validate_framework(a, b, recipe, validation))
        for a, b in zip(references, raw_values, strict=True)
    ]
    transformations = {}
    if recipe["export"].get("fold_constant_floors"):
        from .boundaries import fold_constant_floors

        model, transformations["constant_floors"] = fold_constant_floors(model)
        if recipe["name"] == "yolov3_full":
            from .boundaries import fold_resize_sizes

            model, transformations["resize_sizes"] = fold_resize_sizes(
                model, transformations["constant_floors"]["changes"]
            )

    if recipe["name"] == "yolov7_face":
        from .broadcast import replace_parameter_tiles
        from .boundaries import materialize_tensor_constants

        model, transformations["broadcast"] = replace_parameter_tiles(
            model, expected={"Add": 3, "Mul": 3}
        )
        model, transformations["constants"] = materialize_tensor_constants(model)
    model = onnx.shape_inference.infer_shapes(model, check_type=True)
    if recipe["export"].get("simplify", False):
        import onnxsim

        model, checked = onnxsim.simplify(
            model,
            check_n=0,
            perform_optimization=recipe["export"].get("simplify_optimization", True),
            skipped_optimizers=recipe["export"].get("skipped_optimizers"),
        )
        if not checked:
            raise ValueError("onnxsim validation failed")
    model = onnx.shape_inference.infer_shapes(model, check_type=True)
    values = outputs(model, feeds)
    for check, a, b in zip(checks, raw_values, values, strict=True):
        check["raw_final"] = compare(a, b)
    available = {
        v.name
        for v in [*model.graph.input, *model.graph.value_info, *model.graph.output]
    }
    split = recipe["export"].get("split", {})
    boundary = split.get("network_input", "images")
    ends = split.get("network_outputs", [v.name for v in model.graph.output])
    if recipe["export"]["family"] == "retinanet":
        convs = [n for n in model.graph.node if n.op_type == "Conv"]
        stems = [n for n in convs if n.name.endswith("/backbone/body/conv1/Conv")]
        if len(stems) != 1 or stems[0] != convs[0]:
            raise ValueError("Ambiguous RetinaNet input transform boundary")
        boundary = stems[0].input[0]
    # tf2onnx suffix counters may differ. Resolve by verified dataflow and head semantics.
    if recipe["export"]["family"] == "tf_ssd":
        convs = [n for n in model.graph.node if n.op_type == "Conv"]
        if not convs or "Conv1" not in convs[0].name:
            raise ValueError("Ambiguous SSD stem")
        boundary = convs[0].input[0]
        ends = []
        for k in reversed(range(6)):
            for kind, tail in [
                ("Class", "ClassPredictor"),
                ("Box", "BoxEncodingPredictor"),
            ]:
                matches = [
                    x
                    for x in available
                    if x.endswith(
                        f"BoxPredictor/Convolutional{kind}Head_{k}/{tail}/BiasAdd:0"
                    )
                ]
                if len(matches) != 1:
                    raise ValueError("Ambiguous SSD head boundary")
                ends += matches
    if boundary not in available or any(x not in available for x in ends):
        raise ValueError(
            "Export boundary tensors changed; explicit resolver update required"
        )
    parts = {}
    if boundary != "images":
        parts["preprocess"] = extract(model, ["images"], [boundary])
        if any(
            n.op_type in ("Conv", "Gemm", "MatMul")
            for n in parts["preprocess"].graph.node
        ):
            raise ValueError("Network leaked into CPU input graph")
    if recipe["export"].get("split_detector"):
        from .boundaries import split_detector

        net, post, transformations["boundary"] = split_detector(model, recipe["name"])
        ends = [v.name for v in net.graph.output]
        parts["network"] = (
            rename_input(extract(net, [boundary], ends), boundary, "images")
            if boundary != "images"
            else net
        )
        parts["postprocess"] = post
        transformations["boundary"]["network_input_map"] = {"images": boundary}
        final_names = [v.name for v in model.graph.output]
    else:
        parts["network"] = rename_input(
            extract(model, [boundary], ends), boundary, "images"
        )
        final_names = [v.name for v in model.graph.output]
        if ends != final_names:
            parts["postprocess"] = extract(model, ends, final_names)
    if set(parts) != set(recipe["graphs"]):
        raise ValueError("Unexpected host partition topology")
    sessions = {k: create_session(v, OrtConfig.cpu()) for k, v in parts.items()}
    for i, feed in enumerate(feeds):
        context = dict(feed)
        if "preprocess" in sessions:
            p = sessions["preprocess"]
            names = [v.name for v in p.get_outputs()]
            context.update(zip(names, run_tensors(p, names, feed)))
        net = sessions["network"]
        names = [v.name for v in net.get_outputs()]
        context.update(
            zip(names, run_tensors(net, names, {"images": context[boundary]}))
        )
        if "postprocess" in sessions:
            p = sessions["postprocess"]
            names = [v.name for v in p.get_outputs()]
            context.update(
                zip(
                    names,
                    run_tensors(
                        p, names, {v.name: context[v.name] for v in p.get_inputs()}
                    ),
                )
            )
        checks[i]["partition"] = compare(
            values[i], {n: context[n] for n in final_names}
        )
    structure = validate_parts(parts, recipe["name"])
    for key, part in parts.items():
        onnx.checker.check_model(part, full_check=True)
        onnx.save_model(part, destination / recipe["graphs"][key])
    return dict(
        checks=checks,
        network_input_map={"images": boundary},
        network_outputs=ends,
        atol=1e-5,
        rtol=1e-4,
        fixture_count=len(feeds),
        state="passed",
        mode=validation,
        framework_parity_passed=all(
            c["framework_raw"].get("strict_passed", False) for c in checks
        ),
        framework_numerical_action="record_only",
        framework_raw_numerical_passed=all(
            c["framework_raw"]["raw_numerical_passed"] for c in checks
        ),
        framework_diagnostics_state=(
            "agreement"
            if all(c["framework_raw"]["state"] == "agreement" for c in checks)
            else "differences"
        ),
        transformations=transformations,
        release_structure=structure,
        execution_scope=dict(
            framework="cpu",
            network="cpu",
            host="cpu",
            gpu_network="not_checked",
        ),
    )
