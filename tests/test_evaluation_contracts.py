import copy
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np

from opennpu_quant.models.adapter import ModelEvaluator
from opennpu_quant.models.contracts import EvaluationSample
from opennpu_quant.models.evaluation_contracts import resolve
from opennpu_quant.models.evaluation_profiles import apply_profile, CURRENT, HISTORICAL
from opennpu_quant.models.spec import ModelSpec, model_names
from opennpu_quant.reproduction.assets import catalog


class EvaluationContractTests(unittest.TestCase):
    def test_all_saved_recipes_resolve_to_same_defaults_and_accept_semantic_names(self):
        entries = catalog()["entries"]
        self.assertEqual(266, len(entries))
        for entry in entries:
            recipe = entry["recipe"]
            expected = resolve(ModelSpec.load(entry["model"]).recipe)
            actual = resolve(recipe)
            self.assertEqual(expected.identity(), actual.identity())
            explicit = resolve(apply_profile(recipe, actual.name))
            self.assertEqual(actual.identity(), explicit.identity())

    def test_semantic_names_work_through_experiment_configuration(self):
        from opennpu_quant.configuration import load_config, validate

        cfg = load_config("cnn38")
        cfg["evaluation"]["profiles"] = {
            name: resolve(ModelSpec.load(name).recipe).name for name in model_names()
        }
        validate(cfg)
        cfg["evaluation"]["profiles"]["retinaface_mnet025_640"] = (
            "posenet_multi_pose_v1"
        )
        with self.assertRaises(ValueError):
            validate(cfg)

    def test_current_keeps_generic_behavior_instead_of_becoming_default(self):
        for model in (
            "yolox",
            "yolov9_t",
            "yolov9_s",
            "yolov7_face",
            "yolov5s_seg",
            "resnet18",
        ):
            recipe = ModelSpec.load(model).recipe
            self.assertNotEqual(
                resolve(recipe).name, resolve(apply_profile(recipe, CURRENT)).name
            )
        for model in model_names():
            recipe = ModelSpec.load(model).recipe
            with self.subTest(model=model):
                if recipe["task"] == "classification":
                    apply_profile(recipe, "numpy_topk_v1")
                else:
                    with self.assertRaises(ValueError):
                        apply_profile(recipe, "numpy_topk_v1")

    def test_semantic_names_preserve_six_faces_and_input_arrays(self):
        recipe = ModelSpec.load("yolov7_face").recipe
        p = np.zeros((1, 6, 21), np.float32)
        p[0, :, :4] = [[i * 20 - 5, 5, 10, 10] for i in range(6)]
        p[0, :, 4:6] = 0.9
        before = p.copy()
        meta = dict(
            original_size=[10, 10], input_size=[10, 10], scale=[1, 1], pad=[0, 0]
        )
        values = []
        for name in (
            HISTORICAL,
            "yolov7_face_clip_first4_v1",
            "face_all_boxes_v1",
            "yolov7_face_clip_all_v1",
        ):
            r = apply_profile(recipe, name)
            values.append(resolve(r).postprocess({"predictions": p}, r, meta, []))
        self.assertEqual(values[0], values[1])
        self.assertEqual(values[2], values[3])
        self.assertEqual(values[0][:4], values[2][:4])
        self.assertNotEqual(values[0][4:], values[2][4:])
        np.testing.assert_array_equal(p, before)

    def test_python_replacement_receives_host_outputs_and_reuses_metric(self):
        recipe = ModelSpec.load("fcn_r50_voc").recipe
        events = []

        def finish(outputs, metadata):
            events.append("host")
            return {"sentinel": outputs["network"] + 1}

        def replacement(outputs, model, metadata, categories):
            events.append("python")
            self.assertEqual(2, outputs["sentinel"][0])
            self.assertEqual(recipe, model)
            self.assertEqual("original", metadata["restore"])
            self.assertEqual([], categories)
            return np.array([[0, 1], [1, 0]], dtype=np.int64)

        bundle = SimpleNamespace(spec=ModelSpec(recipe), finish=finish)
        evaluator = ModelEvaluator(bundle, SimpleNamespace(), postprocess=replacement)
        with patch(
            "opennpu_quant.models.evaluation_contracts.legacy_profile",
            side_effect=AssertionError("per-sample resolution"),
        ):
            evaluator.update(
                {"network": np.array([1.0], np.float32)},
                EvaluationSample(
                    "one", {}, np.array([[0, 1], [1, 0]]), {"restore": "original"}
                ),
            )
        self.assertEqual(["host", "python"], events)
        self.assertEqual({"mIoU": 1.0}, evaluator.finalize())
        self.assertEqual({"labels": [[0, 1], [1, 0]]}, evaluator.prediction)
        self.assertFalse(evaluator.contract.identity()["reusable"])
        self.assertFalse(
            evaluator.contract.identity()["original_reproduction_eligible"]
        )

        from onnx import helper as h
        from opennpu_quant import evaluate, OrtConfig

        model = h.make_model(
            h.make_graph(
                [h.make_node("Identity", ["images"], ["network"])],
                "hook",
                [h.make_tensor_value_info("images", 1, [1])],
                [h.make_tensor_value_info("network", 1, [1])],
            ),
            opset_imports=[h.make_opsetid("", 18)],
            ir_version=10,
        )
        sample = EvaluationSample(
            "one",
            {"images": np.array([1.0], np.float32)},
            np.array([[0, 1], [1, 0]]),
            {"restore": "original"},
        )
        result = evaluate(
            model,
            [sample],
            evaluator=evaluator,
            ort=OrtConfig.cpu(),
            expected_samples=1,
        )
        self.assertEqual({"mIoU": 1.0}, result.metrics)
        self.assertFalse(result.runtime["evaluation_contract"]["reusable"])
        self.assertFalse(
            result.runtime["evaluation_contract"]["original_reproduction_eligible"]
        )

    def test_hook_validation_and_scoring_policy_are_separate(self):
        r = ModelSpec.load("yolov5s_seg").recipe
        original = copy.deepcopy(r)
        a = resolve(r, postprocess=lambda *args: [])
        b = resolve(
            apply_profile(r, "yolov5_seg_scaled_coeff_mask_area_v1"),
            postprocess=lambda *args: [],
        )
        self.assertEqual(("bbox", "mask"), (a.mask_area, b.mask_area))
        self.assertTrue(a.bbox_metric and b.bbox_metric)
        self.assertEqual(original, r)
        with self.assertRaises(TypeError):
            resolve(r, postprocess="module:function")
        with self.assertRaises(ValueError):
            resolve(ModelSpec.load("resnet18").recipe, postprocess=lambda *args: [])
        with self.assertRaises(ValueError):
            apply_profile(r, "yolov7_face_clip_all_v1")


if __name__ == "__main__":
    unittest.main()
