"""Structural parity is independent of framework output tolerances."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh

from opennpu_quant._io import object_hash
from opennpu_quant.export.contract import graph_contract, validate_parts
from opennpu_quant.export.finalize import finalize, outputs


def graph():
    nodes = [
        h.make_node("Conv", ["images", "weight"], ["head"], name="stem"),
        h.make_node("LeakyRelu", ["head"], ["a"], alpha=0.1, name="activation"),
        h.make_node("Neg", ["head"], ["b"], name="negative"),
        h.make_node("Sub", ["a", "b"], ["output"], name="merge"),
        h.make_node("Shape", ["head"], ["size"], name="shape"),
    ]
    g = h.make_graph(
        nodes,
        "test",
        [h.make_tensor_value_info("images", 1, [1, 1, "h", "w"])],
        [
            h.make_tensor_value_info("output", 1, [1, 1, "h", "w"]),
            h.make_tensor_value_info("size", 7, [4]),
        ],
        [nh.from_array(np.ones((1, 1, 1, 1), np.float32), "weight")],
    )
    return h.make_model(g, ir_version=8, opset_imports=[h.make_opsetid("", 17)])


def contract(m):
    return {k: v for k, v in graph_contract(m, network=True).items() if k != "names"}


class ExportContractTests(unittest.TestCase):
    def test_equal_value_nodes_do_not_hide_changed_sharing(self):
        model = graph()
        model.graph.node[2].op_type = "LeakyRelu"
        model.graph.node[2].attribute.extend(model.graph.node[1].attribute)
        expected = contract(model)
        model.graph.node[3].input[1] = "a"
        model.graph.output.append(h.make_tensor_value_info("b", 1, [1, 1, "h", "w"]))
        self.assertNotEqual(expected, contract(model))

    def test_initializer_input_defaults_are_part_of_the_contract(self):
        model = graph()
        model.graph.input.append(h.make_tensor_value_info("weight", 1, [1, 1, 1, 1]))
        expected = graph_contract(model)["structure"]
        model.graph.initializer[0].CopyFrom(
            nh.from_array(np.zeros((1, 1, 1, 1), np.float32), "weight")
        )
        self.assertNotEqual(expected, graph_contract(model)["structure"])

    def test_names_metadata_and_independent_node_order_are_irrelevant(self):
        original = graph()
        changed = copy.deepcopy(original)
        nodes = list(changed.graph.node)
        del changed.graph.node[:]
        changed.graph.node.extend([nodes[i] for i in (0, 2, 1, 4, 3)])
        for node in changed.graph.node:
            node.name = "new_" + node.name
            for field in (node.input, node.output):
                for i, name in enumerate(field):
                    field[i] = "new_" + name
        for value in [
            *changed.graph.input,
            *changed.graph.output,
            *changed.graph.initializer,
        ]:
            value.name = "new_" + value.name
        for value in [*changed.graph.input, *changed.graph.output]:
            for dim in value.type.tensor_type.shape.dim:
                if dim.dim_param:
                    dim.dim_param = "new_" + dim.dim_param
        changed.producer_name = "different"
        changed.graph.doc_string = "metadata"
        self.assertEqual(contract(original), contract(changed))

    def test_parameter_attribute_connection_opset_and_shape_changes_fail(self):
        original = graph()
        mutations = []
        m = copy.deepcopy(original)
        value = nh.to_array(m.graph.initializer[0]).copy()
        value.flat[0] = np.nextafter(value.flat[0], np.float32(2))
        m.graph.initializer[0].CopyFrom(nh.from_array(value, "weight"))
        mutations.append(m)
        m = copy.deepcopy(original)
        m.graph.node[1].attribute[0].f = 0.2
        mutations.append(m)
        m = copy.deepcopy(original)
        m.graph.node[3].input.reverse()
        mutations.append(m)
        m = copy.deepcopy(original)
        m.opset_import[0].version = 18
        mutations.append(m)
        m = copy.deepcopy(original)
        m.graph.output[0].type.tensor_type.shape.dim[3].dim_param = "h"
        mutations.append(m)
        m = copy.deepcopy(original)
        del m.graph.output[1]
        del m.graph.node[4]
        mutations.append(m)
        for changed in mutations:
            with self.subTest(model=changed.SerializeToString()[-30:]):
                self.assertNotEqual(contract(original), contract(changed))

    def test_constants_are_not_equivalent_to_initializers(self):
        original = graph()
        changed = copy.deepcopy(original)
        weight = changed.graph.initializer.pop()
        changed.graph.node.insert(
            0, h.make_node("Constant", [], ["weight"], value=weight)
        )
        self.assertNotEqual(contract(original), contract(changed))

    def test_nested_graph_capture_and_constant_bits_are_checked(self):
        sub = h.make_graph(
            [h.make_node("Add", ["x", "bias"], ["result"])],
            "branch",
            [],
            [h.make_tensor_value_info("result", 1, [1])],
            [nh.from_array(np.array([1], np.float32), "bias")],
        )
        outer = h.make_graph(
            [
                h.make_node(
                    "If", ["condition"], ["out"], then_branch=sub, else_branch=sub
                )
            ],
            "outer",
            [
                h.make_tensor_value_info("x", 1, [1]),
                h.make_tensor_value_info("condition", 9, []),
            ],
            [h.make_tensor_value_info("out", 1, [1])],
        )
        m = h.make_model(outer, opset_imports=[h.make_opsetid("", 17)])
        before = graph_contract(m)
        t = m.graph.node[0].attribute[0].g.initializer[0]
        v = nh.to_array(t).copy()
        v[0] = np.nextafter(v[0], np.float32(2))
        t.CopyFrom(nh.from_array(v, "bias"))
        self.assertNotEqual(before["structure"], graph_contract(m)["structure"])

    def test_nested_graph_preserves_parent_dynamic_dimension_relationship(self):
        sub = h.make_graph(
            [h.make_node("Identity", ["x"], ["result"])],
            "branch",
            [],
            [h.make_tensor_value_info("result", 1, ["batch"])],
        )
        outer = h.make_graph(
            [
                h.make_node(
                    "If", ["condition"], ["out"], then_branch=sub, else_branch=sub
                )
            ],
            "outer",
            [
                h.make_tensor_value_info("x", 1, ["batch"]),
                h.make_tensor_value_info("condition", 9, []),
            ],
            [h.make_tensor_value_info("out", 1, ["batch"])],
        )
        model = h.make_model(outer, opset_imports=[h.make_opsetid("", 17)])
        before = graph_contract(model)["structure"]
        model.graph.node[0].attribute[0].g.output[0].type.tensor_type.shape.dim[
            0
        ].dim_param = "unrelated"
        self.assertNotEqual(before, graph_contract(model)["structure"])

    def test_structure_guard_is_required_in_both_validation_modes(self):
        model = graph()
        recipe = dict(
            name="contract_fixture",
            export=dict(family="fixture"),
            graphs=dict(network="network.onnx"),
        )
        feed = {"images": np.ones((1, 1, 2, 3), np.float32)}
        expected = outputs(model, [feed])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw.onnx"
            onnx.save(model, raw)
            for mode in ("strict", "functional"):
                with (
                    patch(
                        "opennpu_quant.export.finalize.validate_framework",
                        return_value={},
                    ),
                    patch(
                        "opennpu_quant.export.finalize.validate_parts",
                        side_effect=ValueError("structure mismatch"),
                    ) as guard,
                    self.assertRaisesRegex(ValueError, "structure mismatch"),
                ):
                    finalize(raw, recipe, [feed], expected, root, validation=mode)
                guard.assert_called_once()
                self.assertFalse((root / "network.onnx").exists())

    def test_partition_wiring_is_checked_across_renamed_graphs(self):
        network = graph()
        network.graph.output.append(
            h.make_tensor_value_info("head", 1, [1, 1, "h", "w"])
        )
        post = h.make_model(
            h.make_graph(
                [h.make_node("Identity", ["output"], ["prediction"])],
                "host",
                [h.make_tensor_value_info("output", 1, [1, 1, "h", "w"])],
                [h.make_tensor_value_info("prediction", 1, [1, 1, "h", "w"])],
            ),
            opset_imports=[h.make_opsetid("", 17)],
        )
        models = {
            "fixture": {
                "network": graph_contract(network, network=True),
                "postprocess": graph_contract(post),
            }
        }
        with tempfile.TemporaryDirectory() as temp:
            reference = dict(models=models, id=object_hash(models))
            (Path(temp) / "export-contracts.json").write_text(json.dumps(reference))
            with patch("opennpu_quant.export.contract.files", return_value=Path(temp)):
                self.assertTrue(
                    validate_parts(
                        {"network": network, "postprocess": post}, "fixture"
                    )["passed"]
                )
                network.graph.node[3].output[0] = "renamed_output"
                network.graph.output[0].name = "renamed_output"
                post.graph.input[0].name = "renamed_output"
                post.graph.node[0].input[0] = "renamed_output"
                self.assertTrue(
                    validate_parts(
                        {"network": network, "postprocess": post}, "fixture"
                    )["passed"]
                )
                post.graph.input[0].name = "head"
                post.graph.node[0].input[0] = "head"
                self.assertEqual(
                    models["fixture"]["postprocess"]["structure"],
                    graph_contract(post)["structure"],
                )
                with self.assertRaisesRegex(ValueError, "connection mismatch"):
                    validate_parts({"network": network, "postprocess": post}, "fixture")

    def test_quantization_edges_are_checked_even_when_graph_digest_matches(self):
        model = graph()
        wanted = graph_contract(model, network=True)
        models = {"fixture": {"network": wanted}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "export-contracts.json"
            reference = dict(models=models, id=object_hash(models))
            path.write_text(json.dumps(reference))
            with patch("opennpu_quant.export.contract.files", return_value=Path(temp)):
                self.assertTrue(validate_parts({"network": model}, "fixture")["passed"])
                wanted["quantization"]["all"]["edges"] = "0" * 64
                reference["id"] = object_hash(models)
                path.write_text(json.dumps(reference))
                with self.assertRaisesRegex(ValueError, "structure mismatch"):
                    validate_parts({"network": model}, "fixture")
