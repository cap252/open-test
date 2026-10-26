"""Model catalog, export, cache, decoder, quantization and metric contracts."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from onnx import helper as h, numpy_helper as n
from unittest.mock import patch
from opennpu_quant import AdaroundConfig, QuantizationConfig, quantize
from opennpu_quant.configuration import load_config
from opennpu_quant.models.spec import ModelSpec, model_names
from opennpu_quant.export.boundaries import split_detector, fold_constant_floors
from opennpu_quant.export.finalize import validate_framework, compare, outputs
from opennpu_quant.ort.config import OrtConfig
from opennpu_quant.quant import adaround
from opennpu_quant.quant.adaround_cache import (
    ActivationCacheConfig,
    LayerActivationCache,
)
from opennpu_quant.quant.parameters import verify_parameters, parameter_identity
from tests.library_fixtures import toy_model, feeds


class Cnn38Tests(unittest.TestCase):
    def test_catalog_and_public_policies(self):
        names = model_names()
        config = load_config("cnn30")
        self.assertEqual(38, len(names))
        self.assertEqual(30, len(config["models"]))
        self.assertEqual(7, len(config["conditions"]))
        self.assertEqual(5, len(load_config("core10")["conditions"]))
        self.assertTrue(set(load_config("core10")["models"]) < set(config["models"]))
        for name in names:
            recipe = ModelSpec.load(name).recipe
            self.assertEqual(64, len(recipe["source"]["sha256"]))
            if "repo" in recipe["source"]:
                self.assertRegex(recipe["source"]["commit"], "^[0-9a-f]{40}$")
            if name in config["models"]:
                self.assertEqual(
                    set(config["schemes"]), set(config["percentiles"][name])
                )
                self.assertEqual({99.99}, set(config["percentiles"][name].values()))

    def test_additional_models_remain_in_reference_selection(self):
        from opennpu_quant.reproduction.assets import catalog

        additional = {
            "efficientnet_lite0",
            "yolov9_t",
            "posenet_101",
            "deeplabv3plus_r50_voc",
            "yolov3_full",
            "yolov4_full",
            "yolov7_full",
            "yolov9_s",
        }
        self.assertEqual(
            additional, set(model_names()) - set(load_config("cnn30")["models"])
        )
        entries = catalog()["entries"]
        self.assertEqual(266, len(entries))
        self.assertEqual(set(model_names()), {e["model"] for e in entries})

    def test_cnn30_exact_order_tasks_and_experiment_settings(self):
        from collections import Counter

        expected = [
            "resnet18",
            "resnet50",
            "mobilenet_v2",
            "mobilenet_v3_large",
            "efficientnet_b0",
            "inception_v3",
            "yolov5s",
            "ssd_mobilenet_v2_320",
            "deeplabv3_mnv2_voc",
            "retinaface_mnet025_640",
            "resnet101",
            "mobilenet_v1",
            "efficientnet_lite4",
            "squeezenet1_1",
            "shufflenet_v2_x1_0",
            "vgg16",
            "yolov3_tiny",
            "yolov4_tiny",
            "yolov5n",
            "yolov7_tiny",
            "yolov8n",
            "nanodet_plus_m_320",
            "retinanet_resnet50_fpn",
            "yolox",
            "yolov7_face",
            "unet_resnet50_voc",
            "deeplabv3_r50_voc",
            "fcn_r50_voc",
            "yolov5s_seg",
            "mobilefacenet_relu_foamliu",
        ]
        current = load_config("cnn30")
        self.assertEqual(expected, current["models"])
        self.assertEqual(30, len(set(current["models"])))
        self.assertEqual(load_config("core10")["models"], expected[:10])
        self.assertEqual(
            dict(
                classification=12,
                detection=10,
                face=2,
                segmentation=4,
                instance_segmentation=1,
                face_verification=1,
            ),
            Counter(ModelSpec.load(m).recipe["task"] for m in expected),
        )
        self.assertEqual(7, len(current["conditions"]))
        self.assertIsNone(current["evaluation"]["limit"])
        self.assertEqual("S-640", ModelSpec.load("yolox").recipe["source"]["variant"])

    def test_both_export_modes_record_numerical_differences(self):
        recipe = ModelSpec.load("resnet101").recipe
        a = {"logits": np.arange(6, dtype=np.float32)[None]}
        b = {"logits": a["logits"] + 0.2}
        self.assertFalse(
            validate_framework(a, b, recipe, "functional")["strict_passed"]
        )
        self.assertFalse(validate_framework(a, b, recipe, "strict")["strict_passed"])
        for value in [
            a["logits"].astype(np.float64),
            a["logits"][:, :3],
            np.full_like(a["logits"], np.nan),
        ]:
            for mode in ("strict", "functional"):
                with self.assertRaises(ValueError):
                    validate_framework(a, {"logits": value}, recipe, mode)

    def test_adaround_functional_mode_records_only_finite_numerical_errors(self):
        from opennpu_quant.quant.config import AdaroundConfig
        from opennpu_quant.quant.adaround import _assert_parity

        a = np.array([1.0, 2.0], np.float32)
        b = a + 0.01
        with self.assertRaisesRegex(ValueError, "parity failed"):
            _assert_parity(a, b, AdaroundConfig(), "test")
        config = AdaroundConfig(validation="functional")
        evidence = _assert_parity(a, b, config, "test")
        self.assertGreater(evidence["max_abs"], 0.009)
        for invalid in (a[:1], a.astype(np.float64), np.full_like(a, np.nan)):
            with self.assertRaisesRegex(ValueError, "structural"):
                _assert_parity(a, invalid, config, "test")
        with self.assertRaises(ValueError):
            AdaroundConfig(validation="ignored")

    def test_variable_cache_reuse_relocation_and_tamper(self):
        from opennpu_quant.data.variable_cache import prepare_variable_feed_cache

        calls = []

        def factory():
            calls.append(1)
            for shape in [(1, 3, 4, 6), (1, 3, 6, 4)]:
                yield {"images": np.ones(shape, np.float32)}

        with tempfile.TemporaryDirectory() as temp:
            kwargs = dict(
                identity={"recipe": "dynamic"},
                verify_source=lambda: None,
                expected_samples=2,
                max_bytes=10000,
            )
            cache = prepare_variable_feed_cache(Path(temp) / "cache", factory, **kwargs)
            self.assertEqual(
                [(1, 3, 4, 6), (1, 3, 6, 4)], [f["images"].shape for f in cache()]
            )
            cache = prepare_variable_feed_cache(Path(temp) / "cache", factory, **kwargs)
            self.assertTrue(cache.reused)
            self.assertEqual(2, len(calls))
            payload = next((Path(temp) / "cache").glob("*.npy"))
            payload.chmod(0o644)
            payload.write_bytes(payload.read_bytes() + b"tampered")
            with self.assertRaises(ValueError):
                list(cache())

    def test_fixed_upsample_preserves_original_framework_output(self):
        import torch
        from opennpu_quant.export.extended_torch import fixed_nearest_upsample

        model = torch.nn.Upsample(
            scale_factor=2, mode="nearest", recompute_scale_factor=True
        )
        feeds = [{"images": np.arange(18, dtype=np.float32).reshape(1, 1, 3, 6)}]
        rewritten = fixed_nearest_upsample(model, feeds)
        self.assertTrue(model.recompute_scale_factor)
        self.assertFalse(rewritten.recompute_scale_factor)
        np.testing.assert_array_equal(
            model(torch.from_numpy(feeds[0]["images"])).numpy(),
            rewritten(torch.from_numpy(feeds[0]["images"])).numpy(),
        )

    def test_warm_face_grid_exports_for_other_resolutions(self):
        import torch
        from opennpu_quant.export.extended_torch import dynamic_face_grids
        from opennpu_quant.ort.session import create_session

        class IKeypoint(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.nl = 1
                self.grid = [torch.zeros(1)]

            def forward(self, features):
                x = features[0]
                h, w = x.shape[2:]
                if self.grid[0].shape[2:] != x.shape[2:]:
                    yy, xx = torch.meshgrid(
                        torch.arange(h), torch.arange(w), indexing="ij"
                    )
                    self.grid[0] = torch.stack((xx, yy))[None].float()
                return x + self.grid[0]

        class Detector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = IKeypoint()

            def forward(self, x):
                return self.head([x])

        model = Detector().eval()
        feeds = [
            {"images": np.ones((1, 2, h, w), np.float32)}
            for h, w in [(4, 4), (4, 2), (2, 4)]
        ]
        rewritten = dynamic_face_grids(model, feeds)
        # Reproduce a warmed cache matching the tracing resolution.
        x = torch.from_numpy(feeds[0]["images"])
        rewritten(x)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "face.onnx"
            torch.onnx.export(
                rewritten,
                x,
                path,
                input_names=["images"],
                output_names=["output"],
                opset_version=17,
                dynamo=False,
                dynamic_axes={"images": {2: "h", 3: "w"}, "output": {2: "h", 3: "w"}},
            )
            session = create_session(path, OrtConfig.cpu())
            for feed in feeds:
                np.testing.assert_array_equal(
                    model(torch.from_numpy(feed["images"])).numpy(),
                    session.run(None, feed)[0],
                )

    def test_dynamic_letterbox_rounding_and_coordinate_metadata(self):
        from PIL import Image
        from opennpu_quant.models.preprocess import preprocess
        from opennpu_quant.models.extended_decode import restore_boxes

        # The short edge is 304.64 pixels: flooring before stride rounding would
        # incorrectly select 320 instead of 352, and odd padding needs halves.
        cfg = ModelSpec.load("yolov9_t").recipe["preprocess"]
        feed, meta = preprocess(Image.new("RGB", (1000, 476)), cfg, "NCHW")
        self.assertEqual((1, 3, 352, 672), feed.shape)
        self.assertEqual([16.0, 24.0], meta["pad"])
        self.assertEqual([304 / 476, 304 / 476], meta["scale"])
        cfg = ModelSpec.load("yolov5s_seg").recipe["preprocess"]
        _, meta = preprocess(Image.new("RGB", (1000, 477)), cfg, "NCHW")
        self.assertEqual([16.0, 23.5], meta["pad"])
        cfg = ModelSpec.load("yolov7_face").recipe["preprocess"]
        _, meta = preprocess(Image.new("RGB", (1000, 479)), cfg, "NCHW")
        expected = np.array([[20, 30, 100, 120]], np.float32)
        boxes = np.array([[20, 30, 120, 150]], np.float32) * 0.64
        boxes += np.tile(meta["pad"], 2)
        np.testing.assert_allclose(restore_boxes(boxes, meta), expected, atol=1e-5)

    def test_depth_to_space_qdq_audit_and_adaround(self):
        from opennpu_quant import (
            calibrate,
            quantize,
            apply_adaround,
            CalibrationConfig,
            QuantizationConfig,
            AdaroundConfig,
        )
        from opennpu_quant.runner import runtime_audit
        from opennpu_quant.ort.session import create_session

        model = h.make_model(
            h.make_graph(
                [
                    h.make_node("Conv", ["images", "w"], ["feature"], name="conv"),
                    h.make_node(
                        "DepthToSpace", ["feature"], ["sr"], blocksize=2, name="shuffle"
                    ),
                ],
                "sr",
                [h.make_tensor_value_info("images", 1, [1, 1, 2, 3])],
                [h.make_tensor_value_info("sr", 1, [1, 1, 4, 6])],
                [
                    n.from_array(
                        np.arange(4, dtype=np.float32).reshape(4, 1, 1, 1) / 4, "w"
                    )
                ],
            ),
            opset_imports=[h.make_opsetid("", 17)],
            ir_version=10,
        )

        def feeds():
            yield {"images": np.arange(6, dtype=np.float32).reshape(1, 1, 2, 3) / 6}

        ort = OrtConfig.cpu()
        stats = calibrate(
            model, feeds, config=CalibrationConfig.percentile(99.99), ort=ort
        )
        q = quantize(
            model,
            stats,
            config=QuantizationConfig(
                activation_scale_policy="pot_ceil", weight_scale_policy="pot_ceil"
            ),
            ort=ort,
        )
        session = create_session(q.model, ort)
        runtime_audit(q, ort)
        self.assertEqual((1, 1, 4, 6), session.run(None, next(feeds()))[0].shape)
        rounded = apply_adaround(
            model,
            q,
            feeds,
            config=AdaroundConfig(
                steps=2, batch_size=1, window_samples=1, window_steps=1, device="cpu"
            ),
            ort=ort,
        )
        self.assertTrue(
            np.isfinite(
                create_session(rounded.model, ort).run(None, next(feeds()))[0]
            ).all()
        )

    def test_split_graph_retains_dynamic_shape_dependency(self):
        from opennpu_quant.export.boundaries import split_detector

        model = h.make_model(
            h.make_graph(
                [
                    h.make_node("Conv", ["images", "w"], ["features"], name="conv"),
                    h.make_node("Shape", ["images"], ["shape"]),
                    h.make_node("Reshape", ["features", "shape"], ["predictions"]),
                ],
                "detector",
                [h.make_tensor_value_info("images", 1, [1, 1, "h", "w"])],
                [h.make_tensor_value_info("predictions", 1, [1, 1, "h", "w"])],
                [n.from_array(np.ones((1, 1, 1, 1), np.float32), "w")],
            ),
            opset_imports=[h.make_opsetid("", 17)],
            ir_version=10,
        )
        net, post, _ = split_detector(model, "test")
        for shape in [(1, 1, 3, 5), (1, 1, 5, 3)]:
            feed = {
                "images": np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            }
            result = outputs(net, [feed])[0]
            inputs = {v.name: {**feed, **result}[v.name] for v in post.graph.input}
            compare(outputs(model, [feed])[0], outputs(post, [inputs])[0])

    def test_coco_perfect_mask_and_keypoints(self):
        from opennpu_quant.evaluation.coco import CocoMetric
        from pycocotools import mask
        from opennpu_quant.models.contracts import EvaluationSample

        segmentation = mask.encode(
            np.asfortranarray(np.pad(np.ones((40, 40), np.uint8), 10))
        )
        segmentation["counts"] = segmentation["counts"].decode("ascii")
        keypoints = [v for i in range(17) for v in (20 + i, 20 + i, 2)]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "annotations.json"
            path.write_text(
                json.dumps(
                    dict(
                        info={},
                        images=[dict(id=1, width=60, height=60)],
                        categories=[
                            dict(
                                id=1,
                                name="person",
                                keypoints=[str(i) for i in range(17)],
                                skeleton=[],
                            )
                        ],
                        annotations=[
                            dict(
                                id=1,
                                image_id=1,
                                category_id=1,
                                bbox=[10, 10, 40, 40],
                                area=1600,
                                iscrowd=0,
                                segmentation=segmentation,
                                keypoints=keypoints,
                                num_keypoints=17,
                            )
                        ],
                    )
                )
            )
            for kind, prediction in [
                ("segm", dict(segmentation=segmentation)),
                ("keypoints", dict(keypoints=keypoints)),
            ]:
                metric = CocoMetric(path, iou_type=kind)
                metric.update(
                    [dict(category_id=1, score=1.0, **prediction)],
                    EvaluationSample("1", {}, None),
                )
                with contextlib.redirect_stderr(io.StringIO()):
                    values = metric.finalize()
                self.assertAlmostEqual(1, values["AP"])
                if kind == "keypoints":
                    self.assertNotIn("AP_small", values)

    def test_mask_metric_uses_mask_area_for_false_positive_ranges(self):
        from opennpu_quant.evaluation.coco import CocoMetric
        from pycocotools import mask
        from opennpu_quant.models.contracts import EvaluationSample

        def encode(top, left, size):
            pixels = np.zeros((60, 60), np.uint8)
            pixels[top : top + size, left : left + size] = 1
            rle = mask.encode(np.asfortranarray(pixels))
            rle["counts"] = rle["counts"].decode("ascii")
            return rle

        true_mask, false_mask = encode(5, 5, 20), encode(40, 40, 10)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "mask.json"
            path.write_text(
                json.dumps(
                    dict(
                        info={},
                        images=[dict(id=1, width=60, height=60)],
                        categories=[dict(id=1, name="thing")],
                        annotations=[
                            dict(
                                id=1,
                                image_id=1,
                                category_id=1,
                                bbox=[5, 5, 20, 20],
                                area=400,
                                iscrowd=0,
                                segmentation=true_mask,
                            )
                        ],
                    )
                )
            )
            metric = CocoMetric(path, iou_type="segm")
            metric.update(
                [
                    dict(
                        category_id=1,
                        score=1.0,
                        segmentation=false_mask,
                        bbox=[0, 0, 60, 60],
                    ),
                    dict(
                        category_id=1,
                        score=0.9,
                        segmentation=true_mask,
                        bbox=[5, 5, 20, 20],
                    ),
                ],
                EvaluationSample("1", {}, None),
            )
            with contextlib.redirect_stderr(io.StringIO()):
                values = metric.finalize()
            self.assertAlmostEqual(0.5, values["AP_small"])

    def test_float_scale_audit_catches_bias_product_with_rehashed_parameters(self):
        result = quantize(
            toy_model(),
            feeds,
            config=QuantizationConfig(scope="all"),
            ort=OrtConfig.cpu(),
        )
        audit = copy.deepcopy(result.audit)
        self.assertTrue(audit["scale_audit"]["biases"])
        row = audit["scale_audit"]["biases"][0]
        node = next(x for x in result.model.graph.node if x.name == row["dq_node"])
        scale = next(
            x for x in result.model.graph.initializer if x.name == node.input[1]
        )
        scale.CopyFrom(n.from_array(n.to_array(scale) * np.float32(2), scale.name))
        audit["parameter_identity"] = parameter_identity(result.model)
        with self.assertRaisesRegex(ValueError, "Bias scale must equal"):
            verify_parameters(result.model, audit)

    def test_dynamic_window_replays_overflowing_sample_with_and_without_cache(self):
        import torch

        torch.set_num_threads(1)
        model = h.make_model(
            h.make_graph(
                [h.make_node("Conv", ["x", "w"], ["y"], name="conv")],
                "dynamic",
                [h.make_tensor_value_info("x", 1, [1, 1, "H", "W"])],
                [h.make_tensor_value_info("y", 1, [1, 1, "H", "W"])],
                [n.from_array(np.asarray([[[[0.17]]]], np.float32), "w")],
            ),
            opset_imports=[h.make_opsetid("", 17)],
            ir_version=10,
        )

        def factory():
            for size in (2, 4):
                yield {
                    "x": np.arange(size * size, dtype=np.float32).reshape(
                        1, 1, size, size
                    )
                    / 7
                }

        result = quantize(
            model, factory, config=QuantizationConfig(scope="all"), ort=OrtConfig.cpu()
        )
        row = adaround._targets(model, result.model, result.audit)[0]
        signatures, _ = adaround._scan_feeds(factory)
        cfg = AdaroundConfig(
            steps=4,
            window_samples=2,
            window_steps=1,
            window_bytes=144,
            batch_size=1,
            device="cpu",
        )
        runs = []
        for cache in (None, ActivationCacheConfig(host_bytes=1024)):
            with adaround._torch_policy(torch, cfg, "cpu"):
                codes, details = adaround._fit_layer(
                    row,
                    model,
                    result.model,
                    factory,
                    signatures,
                    cfg,
                    OrtConfig.cpu(),
                    "cpu",
                    cache_config=cache,
                )
            self.assertEqual(2, details["reconstruction_samples_seen"])
            self.assertLessEqual(details["peak_activation_window_bytes"], 144)
            self.assertEqual(1, details["error_samples"])
            runs.append(codes)
        np.testing.assert_array_equal(*runs)

    def test_reclaims_only_sufficient_unused_cuda_reservations(self):
        pair = (np.ones(2, np.float32), np.ones(2, np.float32))
        cache = LayerActivationCache(
            ActivationCacheConfig(device_bytes=32, device_reserve_bytes=8)
        )
        with (
            patch("torch.cuda.mem_get_info", side_effect=[(12, 100), (28, 100)]),
            patch("torch.cuda.memory_reserved", return_value=24),
            patch("torch.cuda.memory_allocated", return_value=8),
            patch("torch.cuda.empty_cache") as empty,
            patch("torch.tensor", side_effect=lambda x, device: x),
        ):
            cache.device_window([pair], "cuda")
            empty.assert_called_once()
        self.assertEqual(1, cache.metrics["device_cache_reclaims"])
        self.assertEqual(16, cache.metrics["device_reclaimed_bytes"])
        with (
            patch("torch.cuda.mem_get_info", return_value=(2, 100)),
            patch("torch.cuda.memory_reserved", return_value=8),
            patch("torch.cuda.memory_allocated", return_value=8),
            patch("torch.cuda.empty_cache") as empty,
        ):
            self.assertEqual([pair], cache.device_window([pair], "cuda"))
            empty.assert_not_called()

    def test_dfl_shape_dependency_does_not_duplicate_decoder_in_network(self):
        nodes = [
            h.make_node("Conv", ["images", "w"], [f"head{i}"], name=f"head{i}")
            for i in range(6)
        ]
        last = "head0"
        for i in range(1, 6):
            nodes.append(h.make_node("Add", [last, f"head{i}"], [f"sum{i}"]))
            last = f"sum{i}"
        nodes += [
            h.make_node("Softmax", [last], ["prob"], axis=1),
            h.make_node("Conv", ["prob", "dfl"], ["predictions"], name="dfl"),
            h.make_node("Shape", ["predictions"], ["shape"]),
        ]
        model = h.make_model(
            h.make_graph(
                nodes,
                "dfl",
                [h.make_tensor_value_info("images", 1, [1, 16, 2, 2])],
                [
                    h.make_tensor_value_info("predictions", 1, [1, 1, 2, 2]),
                    h.make_tensor_value_info("shape", 7, [4]),
                ],
                [
                    n.from_array(
                        np.eye(16, dtype=np.float32).reshape(16, 16, 1, 1), "w"
                    ),
                    n.from_array(
                        np.arange(16, dtype=np.float32).reshape(1, 16, 1, 1), "dfl"
                    ),
                ],
            ),
            opset_imports=[h.make_opsetid("", 17)],
            ir_version=10,
        )
        net, post, proof = split_detector(model, "yolov8n")
        self.assertFalse(proof["shape_outputs"])
        self.assertEqual(6, sum(x.op_type == "Conv" for x in net.graph.node))
        self.assertEqual(1, sum(x.op_type == "Conv" for x in post.graph.node))
        feed = {"images": np.arange(64, dtype=np.float32).reshape(1, 16, 2, 2)}
        context = {**feed, **outputs(net, [feed])[0]}
        compare(
            outputs(model, [feed])[0],
            outputs(post, [{x.name: context[x.name] for x in post.graph.input}])[0],
        )

    def test_constant_floor_does_not_fold_feature_floor(self):
        model = h.make_model(
            h.make_graph(
                [
                    h.make_node("Floor", ["c"], ["a"]),
                    h.make_node("Floor", ["x"], ["b"]),
                ],
                "floor",
                [h.make_tensor_value_info("x", 1, [1])],
                [
                    h.make_tensor_value_info("a", 1, [1]),
                    h.make_tensor_value_info("b", 1, [1]),
                ],
                [n.from_array(np.array([2.7], np.float32), "c")],
            ),
            opset_imports=[h.make_opsetid("", 17)],
            ir_version=10,
        )
        actual, proof = fold_constant_floors(model)
        self.assertEqual(1, len(proof["changes"]))
        self.assertEqual("Floor", actual.graph.node[1].op_type)
        feed = {"x": np.array([1.7], np.float32)}
        compare(outputs(model, [feed])[0], outputs(actual, [feed])[0])


class FrameworkDiagnosticTests(unittest.TestCase):
    def test_common_statistics_and_input_preservation(self):
        recipe = ModelSpec.load("resnet18").recipe
        a = np.array([[0, 1, 2, 3, 4, 5]], np.float32)
        b = a.copy()
        b[0, 0] = 0.25
        before = (a.tobytes(), b.tobytes())
        for mode in ("strict", "functional"):
            result = validate_framework({"logits": a}, {"logits": b}, recipe, mode)
            output = result["outputs"]["logits"]
            self.assertEqual(0.25, output["max_abs"])
            self.assertEqual(1, output["above_tolerance"])
            self.assertEqual(1 / 6, output["above_tolerance_fraction"])
            self.assertFalse(result["raw_numerical_passed"])
            self.assertFalse(result["strict_passed"])
            self.assertEqual("record_only", result["numerical_action"])
            self.assertEqual("not_evaluated", result["quality"]["accuracy"])
            json.dumps(result, allow_nan=False)
        self.assertEqual(before, (a.tobytes(), b.tobytes()))

    def test_zero_reference_and_signed_zero_are_reported(self):
        recipe = dict(task="detection", export=dict(family="fixture"))
        a = np.array([1, -0.0], np.float32)
        b = np.zeros(2, np.float32)
        result = validate_framework({"x": a}, {"x": b}, recipe, "strict")
        self.assertEqual(1, result["outputs"]["x"]["zero_reference_mismatches"])
        self.assertEqual(0, result["outputs"]["x"]["max_relative_nonzero"])
        result = validate_framework({"x": a[1:]}, {"x": b[1:]}, recipe, "strict")
        self.assertTrue(result["common_passed"])
        self.assertFalse(result["outputs"]["x"]["bitwise_equal"])

    def test_invalid_outputs_and_mode_remain_errors(self):
        recipe = ModelSpec.load("resnet18").recipe
        a = {"logits": np.ones((1, 5), np.float32)}
        for mode in ("strict", "functional"):
            for b in (
                {},
                {"wrong": a["logits"]},
                {"logits": np.ones((1, 6), np.float32)},
                {"logits": a["logits"].astype(np.float64)},
                {"logits": np.full_like(a["logits"], np.inf)},
                {"logits": np.full_like(a["logits"], np.nan)},
            ):
                with self.subTest(mode=mode, value=b), self.assertRaises(ValueError):
                    validate_framework(a, b, recipe, mode)
            with self.assertRaises(ValueError):
                validate_framework({}, {}, recipe, mode)
        with self.assertRaisesRegex(ValueError, "mode"):
            validate_framework(a, a, recipe, "ignored")

    def test_segmentation_quality_runs_even_when_allclose_passes(self):
        recipe = ModelSpec.load("unet_resnet50_voc").recipe
        a = np.array([[[[1.0]], [[1.000001]]]], np.float32)
        b = a[:, ::-1].copy()
        result = validate_framework({"logits": a}, {"logits": b}, recipe, "strict")
        self.assertTrue(result["common_passed"])
        self.assertTrue(result["raw_numerical_passed"])
        self.assertFalse(result["effective_passed"])
        self.assertEqual(1, result["quality"]["decisions"]["raw_argmax"]["different"])
        self.assertEqual("differences", result["state"])

    def test_nanodet_exception_is_restricted_to_verified_channels_and_model(self):
        recipe = ModelSpec.load("nanodet_plus_m_320").recipe
        a = np.zeros((1, 2, 112), np.float32)
        b = a.copy()
        b[..., 80:] = 0.00015
        result = validate_framework(
            {"predictions": a}, {"predictions": b}, recipe, "strict"
        )
        self.assertFalse(result["common_passed"])
        self.assertTrue(result["raw_numerical_passed"])
        self.assertEqual(64, result["outputs"]["predictions"]["above_tolerance"])
        self.assertEqual(0.0, result["output_policies"]["regression_logits"]["rtol"])
        self.assertIsNone(result["quality"]["decisions_passed"])
        for index, value in ((0, 0.00015), (80, 0.00021)):
            b = a.copy()
            b[..., index] = value
            result = validate_framework(
                {"predictions": a}, {"predictions": b}, recipe, "strict"
            )
            self.assertFalse(result["raw_numerical_passed"])
        other = copy.deepcopy(recipe)
        other["name"] = "another_nanodet"
        b[..., 80] = 0.00015
        result = validate_framework(
            {"predictions": a}, {"predictions": b}, other, "strict"
        )
        self.assertFalse(result["raw_numerical_passed"])
        with self.assertRaisesRegex(ValueError, "80 probabilities"):
            validate_framework(
                {"predictions": a[..., :-1]},
                {"predictions": a[..., :-1]},
                recipe,
                "strict",
            )

    def test_ssd_raw_and_selected_outputs_have_separate_verdicts(self):
        recipe = ModelSpec.load("ssd_mobilenet_v2_320").recipe
        a = {
            "raw_detection_boxes": np.zeros((1, 2, 4), np.float32),
            "raw_detection_scores": np.ones((1, 2, 91), np.float32),
            "num_detections": np.array([2], np.float32),
            "detection_classes": np.array([[1, 2]], np.float32),
            "detection_anchor_indices": np.array([[3, 4]], np.float32),
        }
        b = {k: v.copy() for k, v in a.items()}
        b["detection_anchor_indices"][0] = [4, 3]
        result = validate_framework(a, b, recipe, "strict")
        self.assertTrue(result["raw_numerical_passed"])
        self.assertFalse(result["common_passed"])
        self.assertFalse(result["quality"]["decisions_passed"])
        self.assertFalse(result["effective_passed"])
        del b["raw_detection_scores"]
        del a["raw_detection_scores"]
        with self.assertRaisesRegex(ValueError, "raw diagnostic"):
            validate_framework(a, b, recipe, "strict")

    def test_known_exceptions_do_not_replace_common_diagnostics(self):
        cases = [
            ("resnet18", "logits", np.arange(6, dtype=np.float32)[None], (0, 0), 5e-5),
            (
                "retinaface_mnet025_640",
                "landmarks",
                np.zeros((1, 1, 10), np.float32),
                (0, 0, 0),
                5e-5,
            ),
            (
                "yolov5s",
                "predictions",
                np.zeros((1, 1, 85), np.float32),
                (0, 0, 0),
                0.001,
            ),
        ]
        for model, name, a, index, delta in cases:
            with self.subTest(model=model):
                b = a.copy()
                b[index] += delta
                result = validate_framework(
                    {name: a}, {name: b}, ModelSpec.load(model).recipe, "strict"
                )
                self.assertFalse(result["common_passed"])
                self.assertTrue(result["raw_numerical_passed"])
                self.assertEqual("differences", result["state"])

    def test_diagnostic_is_visible_and_reports_quality_only_differences(self):
        from opennpu_quant.export.parity import diagnostic_lines

        recipe = ModelSpec.load("resnet18").recipe
        a = np.array([[0, 1, 2, 3, 4, 4.000001]], np.float32)
        b = a.copy()
        b[0, -2:] = b[0, -2:][::-1]
        check = validate_framework({"logits": a}, {"logits": b}, recipe, "strict")
        lines = list(
            diagnostic_lines("resnet18", dict(checks=[dict(framework_raw=check)]))
        )
        self.assertEqual(1, len(lines))
        self.assertIn("above_tolerance=0/6", lines[0])
        self.assertIn("decision checks passed=False", lines[0])
        self.assertIn("Export continued", lines[0])
        self.assertIn("model.json", lines[0])


if __name__ == "__main__":
    unittest.main()
