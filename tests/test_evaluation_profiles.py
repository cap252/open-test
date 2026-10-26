import copy
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from opennpu_quant.models.spec import ModelSpec
from opennpu_quant.models.evaluation_profiles import apply_profile, identity, HISTORICAL
from opennpu_quant.models.historical_yolo import candidates, restored_boxes, postprocess
from opennpu_quant.models.decode import class_nms
from opennpu_quant.evaluation.coco import evaluate_predictions
from opennpu_quant.ort.cuda_libs import compare_libraries
from opennpu_quant.ort.config import OrtConfig
from opennpu_quant.runner import _ModelRun, run
from opennpu_quant._io import read_json
from tests.test_result_reliability import fixture


class EvaluationProfileTests(unittest.TestCase):
    def test_yolox_keeps_outside_and_more_than_300_without_mutating_input(self):
        recipe = apply_profile(ModelSpec.load("yolox").recipe, HISTORICAL)
        p = np.zeros((1, 350, 85), np.float32)
        p[0, :, 0] = np.arange(350) * 3 - 50
        p[0, :, 1] = 3
        p[0, :, 2:4] = 1
        p[0, :, 4:6] = 0.9
        before = p.copy()
        rows = postprocess(
            {"predictions": p},
            recipe,
            dict(
                original_size=[10, 10], scale=[1, 1], pad=[0, 0], input_size=[640, 640]
            ),
            list(range(80)),
            HISTORICAL,
        )
        self.assertEqual(350, len(rows))
        self.assertTrue(any(r["bbox"][0] < 0 for r in rows))
        self.assertTrue(any(r["bbox"][0] > 10 for r in rows))
        np.testing.assert_array_equal(before, p)

    def test_face_clipping_bug_is_isolated_in_explicit_profile(self):
        import torch

        boxes = torch.tensor([[-2.0, -2.0, 12.0, 12.0]] * 6)
        meta = dict(
            original_size=[10, 10], input_size=[10, 10], pad=[0, 0], scale=[1, 1]
        )
        old = restored_boxes(boxes, meta, "yolov7_face", HISTORICAL)
        fixed = restored_boxes(boxes, meta, "yolov7_face", "face_all_boxes_v1")
        np.testing.assert_array_equal(old[:4], [[0, 0, 10, 10]] * 4)
        np.testing.assert_array_equal(old[4:], boxes[4:])
        np.testing.assert_array_equal(fixed, [[0, 0, 10, 10]] * 6)
        np.testing.assert_array_equal(boxes, [[-2, -2, 12, 12]] * 6)

    def test_original_nms_boundary_and_equal_score_order(self):
        cfg = copy.deepcopy(ModelSpec.load("yolov9_t").recipe["decode"])
        cfg["nms"]["iou"] = 0.6
        p = np.zeros((1, 84, 2), np.float32)
        p[0, :4] = np.array([[5, 7.5], [5, 5], [10, 10], [10, 10]])
        p[0, 4] = [0.9, 0.8]
        kept = candidates(p, "yolov9_t", cfg)
        self.assertEqual(1, len(kept))
        old = class_nms(
            np.array([[0, 0, 10, 10], [2.5, 0, 12.5, 10]], np.float32),
            np.array([0.9, 0.8], np.float32),
            np.zeros(2),
            0.6,
            300,
        )
        self.assertEqual(2, len(old))
        p = np.zeros((1, 84, 3), np.float32)
        p[0, :4] = np.array([[5, 5, 25], [5, 5, 25], [10, 10, 10], [10, 10, 10]])
        p[0, 4] = 0.5
        kept = candidates(p, "yolov9_t", cfg)
        np.testing.assert_array_equal(kept[:, 0], [0, 20])

    def test_segmentation_coefficients_and_serialization(self):
        cfg = ModelSpec.load("yolov5s_seg").recipe["decode"]
        p = np.zeros((1, 1, 117), np.float32)
        p[0, 0, :4] = [5, 5, 8, 8]
        p[0, 0, 4] = 0.2
        p[0, 0, 5] = 0.9
        p[0, 0, 85:] = 2
        before = p.copy()
        selected = candidates(p, "yolov5s_seg", cfg)
        np.testing.assert_array_equal(
            selected[:, 6:], np.full((1, 32), 0.4, np.float32)
        )
        np.testing.assert_array_equal(p, before)
        recipe = apply_profile(ModelSpec.load("yolov9_t").recipe, HISTORICAL)
        p = np.zeros((1, 84, 1), np.float32)
        p[0, :4, 0] = [5.123456, 5, 8, 8]
        p[0, 4, 0] = 0.12345678
        rows = postprocess(
            {"predictions": p},
            recipe,
            dict(original_size=[10, 10], input_size=[10, 10], pad=[0, 0], scale=[1, 1]),
            list(range(80)),
            HISTORICAL,
        )
        self.assertEqual(0.12346, rows[0]["score"])
        self.assertEqual(1.123, rows[0]["bbox"][0])

    def test_coco_mask_area_profiles_and_input_preservation(self):
        from pycocotools.coco import COCO
        from pycocotools import mask as api

        gt = COCO()
        gt.dataset = dict(
            images=[dict(id=1, height=10, width=10)],
            categories=[dict(id=1, name="object")],
            annotations=[
                dict(
                    id=1,
                    image_id=1,
                    category_id=1,
                    bbox=[0, 0, 8, 8],
                    area=64,
                    iscrowd=0,
                    segmentation=[[0, 0, 0, 2, 2, 2, 2, 0]],
                )
            ],
        )
        gt.createIndex()
        mask = np.zeros((10, 10), np.uint8)
        mask[:2, :2] = 1
        rle = api.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode()
        pred = [
            dict(
                image_id=1,
                category_id=1,
                bbox=[0, 0, 8, 8],
                score=0.9,
                segmentation=rle,
            )
        ]
        before = copy.deepcopy(pred)
        native = gt.loadRes
        areas = []

        def capture(rows):
            result = native(rows)
            areas.append(result.dataset["annotations"][0]["area"])
            return result

        with patch.object(gt, "loadRes", side_effect=capture):
            evaluate_predictions(gt, pred, [1], "segm", mask_area="bbox")
            evaluate_predictions(gt, pred, [1], "segm", mask_area="mask")
        self.assertEqual([64, 4], areas)
        self.assertEqual(before, pred)

    def test_historical_segmentation_runs_through_model_evaluator(self):
        import json
        from opennpu_quant.models.adapter import ModelEvaluator
        from opennpu_quant.models.contracts import EvaluationSample

        with tempfile.TemporaryDirectory() as temporary:
            annotation = Path(temporary) / "truth.json"
            annotation.write_text(
                json.dumps(
                    dict(
                        images=[dict(id=1, width=10, height=10)],
                        categories=[dict(id=1, name="object")],
                        annotations=[
                            dict(
                                id=1,
                                image_id=1,
                                category_id=1,
                                bbox=[1, 1, 8, 8],
                                area=64,
                                iscrowd=0,
                                segmentation=[[1, 1, 9, 1, 9, 9, 1, 9]],
                            )
                        ],
                    )
                )
            )
            spec = ModelSpec(
                apply_profile(ModelSpec.load("yolov5s_seg").recipe, HISTORICAL)
            )
            bundle = SimpleNamespace(
                spec=spec, finish=lambda outputs, metadata: outputs
            )
            evaluator = ModelEvaluator(
                bundle, SimpleNamespace(annotations={"coco": annotation})
            )
            prediction = np.zeros((1, 1, 117), np.float32)
            prediction[0, 0, :6] = [5, 5, 8, 8, 0.9, 0.9]
            prediction[0, 0, 85:] = 1
            out = dict(
                predictions=prediction, prototypes=np.ones((1, 32, 2, 2), np.float32)
            )
            before = {k: v.copy() for k, v in out.items()}
            meta = dict(
                original_size=[10, 10], input_size=[10, 10], pad=[0, 0], scale=[1, 1]
            )
            evaluator.update(out, EvaluationSample("1", {}, None, metadata=meta))
            metrics = evaluator.finalize()
            self.assertAlmostEqual(1, metrics["bbox_AP"])
            self.assertAlmostEqual(1, metrics["mask_AP"])
            self.assertEqual("bbox", evaluator.metric_details["mask"]["mask_area"])
            for k in out:
                np.testing.assert_array_equal(out[k], before[k])

    def test_profile_changes_evaluation_identity_but_not_preprocessing(self):
        recipe = ModelSpec.load("yolov7_face").recipe
        fake = SimpleNamespace(
            dataset=SimpleNamespace(identity=lambda *args: "data"),
            graphs={"network": "graph"},
            ort=OrtConfig.cpu(),
            code="code",
            bundle=SimpleNamespace(mapping={"images": "images"}),
        )
        identities = []
        for name in (HISTORICAL, "face_all_boxes_v1"):
            chosen = apply_profile(recipe, name)
            fake.spec = ModelSpec(chosen)
            fake.evaluation_profile = identity(chosen)
            _ModelRun._identify_inputs(fake, 2)
            identities.append((fake.preprocessing_identity, fake.evaluation_identity))
        self.assertEqual(identities[0][0], identities[1][0])
        self.assertNotEqual(identities[0][1], identities[1][1])
        self.assertEqual(HISTORICAL, recipe["decode"]["profile"])

    def test_profile_config_change_rejects_completed_run_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            paths, cfg = fixture(Path(temp))
            cfg["conditions"] = {"fp32": {}}
            run(cfg, paths)
            original = read_json(paths.runs / cfg["name"] / "toy/fp32/result.json")
            cfg["evaluation"]["profiles"] = {"toy": "current_v1"}
            with self.assertRaisesRegex(ValueError, "evaluation.profiles"):
                run(cfg, paths)
            self.assertEqual(
                original, read_json(paths.runs / cfg["name"] / "toy/fp32/result.json")
            )

    def test_unknown_profile_and_unloaded_libraries_are_distinct(self):
        with self.assertRaisesRegex(ValueError, "not verified"):
            apply_profile(ModelSpec.load("resnet18").recipe, HISTORICAL)
        left = dict(libraries=[dict(name="libcudnn.so.9", sha256="a")])
        right = dict(
            libraries=[
                dict(name="libcudnn.so.9", sha256="b"),
                dict(name="libcublas.so.12", sha256="c"),
            ]
        )
        self.assertEqual(
            {"libcudnn.so.9": "content_changed", "libcublas.so.12": "not_loaded"},
            compare_libraries(left, right),
        )
