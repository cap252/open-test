"""Canonical model defaults, local bundle migration, and comparison isolation."""

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import yaml

from opennpu_quant.models.adapter import ModelEvaluator
from opennpu_quant.models.contracts import EvaluationSample
from opennpu_quant.models.evaluation_profiles import (
    CURRENT,
    HISTORICAL,
    NUMPY_TOPK,
    profile_name,
)
from opennpu_quant.models.spec import ModelSpec, model_names
from opennpu_quant.reproduction.assets import catalog


class EvaluationDefaultTests(unittest.TestCase):
    def test_all_selected_conditions_match_packaged_evaluation_defaults(self):
        entries = catalog()["entries"]
        self.assertEqual(266, len(entries))
        self.assertEqual(set(model_names()), {e["model"] for e in entries})
        for entry in entries:
            with self.subTest(condition=entry["id"]):
                recipe = ModelSpec.load(entry["model"]).recipe
                self.assertEqual(profile_name(entry["recipe"]), profile_name(recipe))
                for key in ("input", "preprocess", "dataset", "task", "metric"):
                    self.assertEqual(entry["recipe"][key], recipe[key], key)
                expected = dict(entry["recipe"]["decode"])
                actual = dict(recipe["decode"])
                expected.setdefault("profile", CURRENT)
                actual.setdefault("profile", CURRENT)
                self.assertEqual(expected, actual)

    def test_old_local_bundle_uses_default_without_changing_its_graph_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "recipe.yaml"
            recipe = copy.deepcopy(ModelSpec.load("yolov9_t").recipe)
            recipe["decode"].pop("profile")
            recipe["graphs"]["postprocess"] = "local_host.onnx"
            recipe["network_input_map"] = {"images": "prepared_images"}
            recipe["preprocess"]["size"] = 320
            path.write_text(yaml.safe_dump(recipe))
            before = path.read_bytes()
            with self.assertWarnsRegex(UserWarning, "packaged default"):
                resolved = ModelSpec.for_run("yolov9_t", temp).recipe
            self.assertEqual(HISTORICAL, profile_name(resolved))
            self.assertEqual(before, path.read_bytes())
            self.assertEqual(CURRENT, profile_name(ModelSpec.load(path).recipe))
            self.assertEqual(recipe, {**resolved, "decode": recipe["decode"]})

    def test_explicit_comparison_takes_precedence_and_is_model_specific(self):
        with tempfile.TemporaryDirectory() as temp:
            default = ModelSpec.for_run("resnet18", temp)
            comparison = ModelSpec.for_run("resnet18", temp, profile=CURRENT)
            self.assertEqual(NUMPY_TOPK, profile_name(default.recipe))
            self.assertEqual(CURRENT, profile_name(comparison.recipe))
            self.assertEqual(
                NUMPY_TOPK, profile_name(ModelSpec.load("resnet18").recipe)
            )
            with self.assertRaisesRegex(ValueError, "not verified"):
                ModelSpec.for_run("resnet18", temp, profile=HISTORICAL)
            for model, alternative in (
                ("yolov7_face", "face_all_boxes_v1"),
                ("yolov5s_seg", "mask_area_v1"),
            ):
                spec = ModelSpec.for_run(model, temp, profile=alternative)
                self.assertEqual(alternative, profile_name(spec.recipe))

    def test_local_explicit_profile_requires_experiment_override_for_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            recipe = copy.deepcopy(ModelSpec.load("yolov7_face").recipe)
            recipe["decode"]["profile"] = "face_all_boxes_v1"
            (Path(temp) / "recipe.yaml").write_text(yaml.safe_dump(recipe))
            with self.assertWarnsRegex(UserWarning, "evaluation.profiles.yolov7_face"):
                spec = ModelSpec.for_run("yolov7_face", temp)
            self.assertEqual(HISTORICAL, profile_name(spec.recipe))
            spec = ModelSpec.for_run("yolov7_face", temp, profile="face_all_boxes_v1")
            self.assertEqual("face_all_boxes_v1", profile_name(spec.recipe))

    def test_custom_local_models_keep_their_recipe_and_reject_wrong_name(self):
        with tempfile.TemporaryDirectory() as temp:
            recipe = copy.deepcopy(ModelSpec.load("resnet18").recipe)
            recipe.update(name="custom_classifier")
            recipe["decode"].pop("profile")
            (Path(temp) / "recipe.yaml").write_text(yaml.safe_dump(recipe))
            self.assertEqual(
                recipe, ModelSpec.for_run("custom_classifier", temp).recipe
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                ModelSpec.for_run("resnet18", temp)

    def test_default_classifier_uses_original_tie_order_and_override_uses_stable(self):
        logits = {"logits": np.ones((1, 6), np.float32)}
        sample = EvaluationSample("tie", {}, 5)
        with tempfile.TemporaryDirectory() as temp:
            for profile, expected in ((None, 1.0), (CURRENT, 0.0)):
                spec = ModelSpec.for_run("resnet18", temp, profile=profile)
                evaluator = ModelEvaluator(
                    SimpleNamespace(spec=spec, finish=lambda outputs, meta: outputs),
                    None,
                )
                evaluator.update(logits, sample)
                self.assertEqual(expected, evaluator.finalize()["top1"])
                np.testing.assert_array_equal(logits["logits"], np.ones((1, 6)))
