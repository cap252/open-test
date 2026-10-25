"""Release-compatible export cuts and evaluation-identical SSD validation inputs."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import helper as h, numpy_helper

from opennpu_quant._io import atomic_json, read_json, sha256
from opennpu_quant.export import worker
from opennpu_quant.export.finalize import finalize, outputs
from opennpu_quant.export.fixtures import synthetic_image
from opennpu_quant.models.bundle import ModelBundle
from opennpu_quant.models.preprocess import preprocess
from opennpu_quant.models.spec import ModelSpec
from opennpu_quant.reproduction.assets import catalog


def head_graph(model, count=None):
    count = count or (10 if model == "retinanet_resnet50_fpn" else 4)
    nodes, tensors = [], []
    feature = "images"
    if model == "retinanet_resnet50_fpn":
        tensors.extend(
            numpy_helper.from_array(np.full((1, 3, 1, 1), v, np.float32), name)
            for v, name in ((0.5, "mean"), (0.25, "std"))
        )
        nodes.extend(
            [
                h.make_node("Sub", [feature, "mean"], ["centered"]),
                h.make_node("Div", ["centered", "std"], ["normalized"]),
            ]
        )
        tensors.append(
            numpy_helper.from_array(
                np.eye(3, dtype=np.float32)[:, :, None, None], "stem_w"
            )
        )
        nodes.append(
            h.make_node(
                "Conv",
                ["normalized", "stem_w"],
                ["stem"],
                name="/model/backbone/body/conv1/Conv",
            )
        )
        feature = "stem"
    heads, ends = [], []
    for i in range(count):
        weight = f"weight_{i}"
        head = f"raw_{i}"
        tensors.append(
            numpy_helper.from_array(
                np.full((1, 3, 1, 1), (i + 1) / 10, np.float32), weight
            )
        )
        nodes.append(h.make_node("Conv", [feature, weight], [head], name=f"head_{i}"))
        heads.append(head)
        if model != "posenet_101" or i == 0:
            end = f"probability_{i}"
            nodes.append(h.make_node("Sigmoid", [head], [end]))
        else:
            end = head
        ends.append(end)
    graph = h.make_graph(
        nodes,
        "heads",
        [h.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [1, 3, 4, 4])],
        [
            h.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1, 1, 4, 4])
            for name in ends
        ],
        tensors,
    )
    return h.make_model(
        graph, ir_version=8, opset_imports=[h.make_opsetid("", 18)]
    ), heads


class ExportBoundaryTests(unittest.TestCase):
    def test_packaged_graph_roles_match_the_release_selection(self):
        entries = catalog()["entries"]
        for name in (
            "nanodet_plus_m_320",
            "posenet_101",
            "retinanet_resnet50_fpn",
            "ssd_mobilenet_v2_320",
        ):
            with self.subTest(model=name):
                selected = next(
                    e
                    for e in entries
                    if e["model"] == name and e["condition"] == "fp32"
                )
                self.assertEqual(
                    set(selected["graphs"]), set(ModelSpec.load(name).recipe["graphs"])
                )

    def check_split(self, name):
        recipe = copy.deepcopy(ModelSpec.load(name).recipe)
        recipe["export"]["simplify"] = False
        model, heads = head_graph(name)
        value = np.linspace(-2, 2, 48, dtype=np.float32).reshape(1, 3, 4, 4)
        feeds = [{"images": value}]
        expected = outputs(model, feeds)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw.onnx"
            onnx.save(model, raw)
            with patch(
                "opennpu_quant.export.finalize.validate_parts",
                return_value={"passed": True},
            ):
                proof = finalize(raw, recipe, feeds, expected, root)
            network = onnx.load(root / "network.onnx")
            host = onnx.load(root / "postprocess.onnx")
            self.assertEqual(heads, [v.name for v in network.graph.output])
            self.assertFalse(any(n.op_type == "Sigmoid" for n in network.graph.node))
            self.assertFalse(any(n.op_type == "Conv" for n in host.graph.node))
            atomic_json(
                root / "model.json",
                dict(
                    files={
                        file: sha256(root / file) for file in recipe["graphs"].values()
                    },
                    network_input_map=proof["network_input_map"],
                ),
            )
            bundle = ModelBundle.from_directory(ModelSpec(recipe), root)
            network_feeds, metadata = bundle.prepare(value, {})
            if name == "retinanet_resnet50_fpn":
                np.testing.assert_array_equal(
                    (value - np.float32(0.5)) / np.float32(0.25),
                    network_feeds["images"],
                )
                self.assertFalse(
                    any(n.op_type in ("Sub", "Div") for n in network.graph.node)
                )
            else:
                np.testing.assert_array_equal(value, network_feeds["images"])
            actual = bundle.finish(outputs(network, [network_feeds])[0], metadata)
            self.assertEqual(set(expected[0]), set(actual))
            for key in actual:
                self.assertEqual(expected[0][key].tobytes(), actual[key].tobytes())

    def test_nanodet_sigmoid_is_after_four_raw_heads(self):
        self.check_split("nanodet_plus_m_320")

    def test_posenet_preserves_sigmoid_and_forwarded_outputs(self):
        self.check_split("posenet_101")

    def test_retinanet_separates_both_input_transform_and_ten_heads(self):
        self.check_split("retinanet_resnet50_fpn")

    def test_missing_head_or_unrecognized_stem_is_rejected(self):
        for name in ("nanodet_plus_m_320", "posenet_101", "retinanet_resnet50_fpn"):
            with self.subTest(model=name), tempfile.TemporaryDirectory() as temp:
                recipe = copy.deepcopy(ModelSpec.load(name).recipe)
                recipe["export"]["simplify"] = False
                model, _ = head_graph(name, count=3)
                if name == "retinanet_resnet50_fpn":
                    next(
                        n for n in model.graph.node if n.op_type == "Conv"
                    ).name = "unexpected_stem"
                raw = Path(temp) / "raw.onnx"
                onnx.save(model, raw)
                feeds = [{"images": np.ones((1, 3, 4, 4), np.float32)}]
                with self.assertRaisesRegex(ValueError, "Ambiguous"):
                    finalize(raw, recipe, feeds, outputs(model, feeds), Path(temp))

    def test_ssd_worker_passes_runtime_preprocessing_to_framework_and_onnx(self):
        recipe = ModelSpec.load("ssd_mobilenet_v2_320").recipe
        captured = []

        def export(recipe, weight, feeds, raw, scratch):
            captured.extend(feeds)
            return [{"scores": np.ones(1, np.float32)} for _ in feeds]

        def finish(raw, recipe, feeds, references, destination, **kwargs):
            self.assertIs(feeds[0], captured[0])
            for file in recipe["graphs"].values():
                (destination / file).write_bytes(b"fixture")
            return dict(
                network_input_map={"images": "images"}, network_outputs=["scores"]
            )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            weight = root / "weights"
            weight.write_bytes(b"fixture")
            with (
                patch(
                    "sys.argv",
                    [
                        "export",
                        recipe["name"],
                        "--destination",
                        str(root / "model"),
                        "--weights-dir",
                        str(root),
                    ],
                ),
                patch.object(worker, "checkpoint", return_value=weight),
                patch.object(worker, "preprocess", wraps=preprocess) as prepare_input,
                patch(
                    "opennpu_quant.export.tensorflow_models.export_model",
                    side_effect=export,
                ),
                patch.object(worker, "finalize", side_effect=finish),
                patch("opennpu_quant.envcheck.info", return_value={}),
            ):
                worker.main()
            self.assertEqual(3, prepare_input.call_count)
        for index, feed in enumerate(captured):
            expected, _ = preprocess(
                synthetic_image(index, 373, 341),
                recipe["preprocess"],
                recipe["input"]["layout"],
            )
            self.assertEqual(np.uint8, feed["images"].dtype)
            self.assertEqual((1, 320, 320, 3), feed["images"].shape)
            self.assertEqual(expected.tobytes(), feed["images"].tobytes())

    def test_worker_records_the_export_settings_it_used(self):
        recipe = ModelSpec.load("yolov3_tiny").recipe

        def finish(raw, recipe, feeds, references, destination, **kwargs):
            for file in recipe["graphs"].values():
                (destination / file).write_bytes(b"fixture")
            return dict(network_input_map={"images": "images"}, network_outputs=["raw"])

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            weight = root / "weights"
            weight.write_bytes(b"fixture")
            with (
                patch(
                    "sys.argv",
                    [
                        "export",
                        recipe["name"],
                        "--destination",
                        str(root / "model"),
                        "--weights-dir",
                        str(root),
                    ],
                ),
                patch.object(worker, "checkpoint", return_value=weight),
                patch.object(worker, "checkout", return_value=root),
                patch(
                    "opennpu_quant.export.torch_models.export_model",
                    return_value=[{}, {}, {}],
                ),
                patch.object(worker, "finalize", side_effect=finish),
                patch("opennpu_quant.envcheck.info", return_value={}),
            ):
                worker.main()
            record = read_json(root / "model/model.json")
        self.assertEqual(recipe["export"], record["validation"]["export_settings"])
        self.assertEqual({"images": "images"}, record["network_input_map"])
        self.assertEqual(set(recipe["graphs"].values()), set(record["files"]))
