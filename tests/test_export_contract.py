"""Structural parity is independent of framework output tolerances."""

import copy
import json
from importlib.resources import files
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


class PackagedExportContractTests(unittest.TestCase):
    def setUp(self):
        self.reference = json.loads(
            files("opennpu_quant.resources")
            .joinpath("export-contracts.json")
            .read_text()
        )
        self.baseline = json.loads(
            (
                Path(__file__).parent / "fixtures/release_export_reference.json"
            ).read_text()
        )

    def test_packaged_reference_checksum_and_release_identity(self):
        self.assertEqual(1, self.reference["schema_version"])
        self.assertEqual(object_hash(self.reference["models"]), self.reference["id"])
        self.assertEqual(self.baseline["reference_id"], self.reference["id"])

    def test_all_packaged_models_and_graph_roles_match_recipes_and_selection(self):
        from opennpu_quant.models.spec import ModelSpec, model_names
        from opennpu_quant.reproduction.assets import catalog

        names = model_names()
        selected = {
            e["model"]: e for e in catalog()["entries"] if e["condition"] == "fp32"
        }
        self.assertEqual(38, len(names))
        self.assertEqual(set(names), set(self.reference["models"]))
        self.assertEqual(set(names), set(selected))
        for name in names:
            with self.subTest(model=name):
                roles = set(self.reference["models"][name])
                self.assertEqual(set(ModelSpec.load(name).recipe["graphs"]), roles)
                self.assertEqual(set(selected[name]["graphs"]), roles)
                self.assertIn("network", roles)

    def test_export_policies_match_the_reviewed_release_baseline(self):
        from opennpu_quant.models.spec import ModelSpec, model_names

        # Graph digests do not expose recipe fields such as simplify or opset.
        # Keep a reviewed snapshot so policy regressions fail without an export.
        self.assertEqual(set(model_names()), set(self.baseline["export_settings"]))
        for name, expected in self.baseline["export_settings"].items():
            with self.subTest(model=name):
                self.assertEqual(expected, ModelSpec.load(name).recipe["export"])


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

    def test_numerical_diagnostic_continues_to_graph_checks_and_saves_model(self):
        model = graph()
        recipe = dict(
            name="contract_fixture",
            task="detection",
            export=dict(family="fixture"),
            graphs=dict(network="network.onnx"),
        )
        feed = {"images": np.ones((1, 1, 2, 3), np.float32)}
        expected = outputs(model, [feed])
        expected[0]["output"] += np.float32(0.5)
        for mode in ("strict", "functional"):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                raw = root / "raw.onnx"
                onnx.save(model, raw)
                with patch(
                    "opennpu_quant.export.finalize.validate_parts",
                    autospec=True,
                    return_value={"passed": True},
                ) as guard:
                    result = finalize(
                        raw, recipe, [feed], expected, root, validation=mode
                    )
                guard.assert_called_once()
                self.assertTrue((root / "network.onnx").is_file())
                self.assertEqual("passed", result["state"])
                self.assertEqual("differences", result["framework_diagnostics_state"])
                self.assertFalse(result["framework_parity_passed"])
                self.assertFalse(result["framework_raw_numerical_passed"])
                self.assertEqual(
                    0.5, result["checks"][0]["framework_raw"]["errors"]["output"]
                )
                self.assertEqual(0.0, result["checks"][0]["raw_final"]["output"])
                self.assertEqual(0.0, result["checks"][0]["partition"]["output"])

    def test_missing_fixtures_cannot_be_reported_as_success(self):
        for feeds, references in (([], []), ([{}], []), ([{}], [{}, {}])):
            with self.subTest(feeds=feeds, references=references):
                with self.assertRaisesRegex(ValueError, "one framework reference"):
                    finalize(None, {}, feeds, references, None)

    def test_raw_final_numerical_failure_is_still_fatal(self):
        model = graph()
        recipe = dict(
            name="contract_fixture",
            task="detection",
            export=dict(family="fixture"),
            graphs=dict(network="network.onnx"),
        )
        feed = {"images": np.ones((1, 1, 2, 3), np.float32)}
        expected = outputs(model, [feed])
        changed = copy.deepcopy(expected)
        changed[0]["output"] += np.float32(0.5)
        for mode in ("strict", "functional"):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                raw = root / "raw.onnx"
                onnx.save(model, raw)
                with (
                    patch(
                        "opennpu_quant.export.finalize.outputs",
                        autospec=True,
                        side_effect=[expected, changed],
                    ),
                    self.assertRaises(AssertionError),
                ):
                    finalize(raw, recipe, [feed], expected, root, validation=mode)
                self.assertFalse((root / "network.onnx").exists())

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
