"""Removed recipe settings fail explicitly; active and plugin settings stay usable."""

import copy
from pathlib import Path
import tempfile
import unittest

import yaml

from opennpu_quant.models.spec import ModelSpec, model_names


class RecipeSettingsTests(unittest.TestCase):
    def test_removed_keys_are_rejected_by_load_and_direct_construction(self):
        cases = [
            ("resnet18", "outputs", {}),
            ("resnet18", "network_outputs", ["logits"]),
            ("resnet18", "metric_params", {}),
            ("resnet18", "input.dtype", "float32"),
            ("resnet18", "export.framework_parity.rtol", 1e-4),
            ("resnet18", "export.framework_parity.atol", 1e-5),
            ("resnet18", "export.framework_parity.require_top5_order", True),
            ("ssd_mobilenet_v2_320", "preprocess.interpolation", "cv2_linear"),
            ("retinaface_mnet025_640", "preprocess.interpolation", "cv2_linear"),
            ("deeplabv3_mnv2_voc", "preprocess.pad", "zeros_bottom_right"),
            ("retinaface_mnet025_640", "decode.record", "wider_int_xywh"),
            ("retinaface_mnet025_640", "decode.rescale", "per_axis_clip"),
            ("retinaface_mnet025_640", "decode.priors.clip", False),
            ("ssd_mobilenet_v2_320", "decode.boxes_to", "clip_xywh"),
            ("yolov5s", "decode.score", "obj_x_cls"),
            ("yolov5s", "decode.boxes", "unletterbox_clip_xywh"),
            ("yolov5s", "decode.category_ids", "coco_sorted"),
            ("yolov5s", "decode.nms.impl", "cv2.dnn.NMSBoxes"),
            ("yolov5s", "decode.nms.per_class", True),
            ("yolov5s", "decode.nms.score_threshold", 0.0),
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "recipe.yaml"
            for model, key, former_value in cases:
                for value in (former_value, None):
                    recipe = copy.deepcopy(ModelSpec.load(model).recipe)
                    node = recipe
                    *parents, leaf = key.split(".")
                    for parent in parents:
                        node = node.setdefault(parent, {})
                    node[leaf] = value
                    path.write_text(yaml.safe_dump(recipe))
                    for entry, build in (
                        ("direct", lambda: ModelSpec(recipe)),
                        ("yaml", lambda: ModelSpec.load(path)),
                    ):
                        with self.subTest(
                            model=model, key=key, value=value, entry=entry
                        ):
                            with self.assertRaises(ValueError) as error:
                                build()
                            message = str(error.exception)
                            self.assertIn("Unknown setting(s)", message)
                            self.assertIn(key, message)
                            self.assertIn("Remove", message)

    def test_removed_key_errors_list_all_paths_in_order(self):
        recipe = copy.deepcopy(ModelSpec.load("resnet18").recipe)
        recipe["input"]["dtype"] = "int8"
        recipe["export"]["framework_parity"]["rtol"] = 1.0
        with self.assertRaisesRegex(
            ValueError, "export.framework_parity.rtol, input.dtype"
        ):
            ModelSpec(recipe)

    def test_packaged_and_active_settings_remain_valid(self):
        names = model_names()
        self.assertEqual(38, len(names))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "recipe.yaml"
            for name in names:
                with self.subTest(model=name):
                    recipe = copy.deepcopy(ModelSpec.load(name).recipe)
                    if recipe["preprocess"]["kind"] == "center_crop":
                        recipe["preprocess"]["interpolation"] = "bicubic"
                    if recipe["decode"]["kind"] == "tf_od_api":
                        for key in ("boxes", "scores", "classes", "count"):
                            recipe["decode"][key] = "custom_" + key
                    path.write_text(yaml.safe_dump(recipe))
                    self.assertEqual(recipe, ModelSpec(recipe).recipe)
                    self.assertEqual(recipe, ModelSpec.load(path).recipe)

    def test_custom_preprocessing_keeps_its_own_settings(self):
        recipe = copy.deepcopy(ModelSpec.load("resnet18").recipe)
        recipe["preprocess"].update(
            kind="custom_plugin:prepare",
            interpolation="nearest",
            pad="center",
            plugin_option={"enabled": True},
        )
        self.assertEqual(recipe, ModelSpec(recipe).recipe)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "recipe.yaml"
            path.write_text(yaml.safe_dump(recipe))
            self.assertEqual(recipe, ModelSpec.load(path).recipe)


if __name__ == "__main__":
    unittest.main()
