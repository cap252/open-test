"""ModelBundle: exported folders against model.json, explicit graph paths, host outputs."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from opennpu_quant._io import atomic_json, sha256
from opennpu_quant.models.bundle import ModelBundle
from opennpu_quant.models.spec import ModelSpec
from tests.test_result_reliability import fixture


class ModelBundleTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def toy(self, host=None):
        paths, _ = fixture(self.root, host=host)
        directory = paths.models / "toy"
        return ModelSpec.load(directory / "recipe.yaml"), directory

    def test_folder_without_model_json_uses_the_recipe_mapping(self):
        spec, directory = self.toy(host="preprocess")
        bundle = ModelBundle.from_directory(spec, directory)
        self.assertEqual({"images": "prepared"}, bundle.mapping)
        self.assertEqual({}, bundle.validation)
        spec.recipe.pop("network_input_map")
        bundle = ModelBundle.from_directory(spec, directory)
        self.assertEqual({"images": "images"}, bundle.mapping)

    def test_model_json_mapping_and_validation_take_precedence(self):
        spec, directory = self.toy(host="preprocess")
        atomic_json(
            directory / "model.json",
            dict(
                files={p.name: sha256(p) for p in directory.glob("*.onnx")},
                network_input_map={"images": "images"},
                validation={"state": "strict"},
            ),
        )
        bundle = ModelBundle.from_directory(spec, directory)
        self.assertEqual({"images": "images"}, bundle.mapping)
        self.assertEqual({"state": "strict"}, bundle.validation)
        graphs = {k: str(directory / v) for k, v in spec.recipe["graphs"].items()}
        bundle = ModelBundle(spec, graphs, manifest=str(directory / "model.json"))
        self.assertEqual({"images": "images"}, bundle.mapping)
        self.assertEqual({"state": "strict"}, bundle.validation)

    def test_changed_graph_is_rejected_before_its_session_and_missing_graph_named(self):
        spec, directory = self.toy(host="postprocess")
        atomic_json(
            directory / "model.json",
            dict(files={p.name: sha256(p) for p in directory.glob("*.onnx")}),
        )
        post = directory / "postprocess.onnx"
        post.write_bytes(post.read_bytes() + b"\0")
        with self.assertRaisesRegex(ValueError, "Model bundle changed: postprocess"):
            ModelBundle.from_directory(spec, directory)
        (directory / "network.onnx").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Export toy first"):
            ModelBundle.from_directory(spec, directory)

    def test_postprocess_runs_before_aliases_and_outputs_are_kept(self):
        spec, directory = self.toy(host="postprocess")
        graphs = {k: directory / v for k, v in spec.recipe["graphs"].items()}
        bundle = ModelBundle(spec, graphs, aliases={"renamed": "scores"})
        _, metadata = bundle.prepare(np.zeros((1, 3, 4, 4), np.float32), {})
        outputs = {"logits": np.arange(6, dtype=np.float32).reshape(1, 6)}
        before = outputs["logits"].copy()
        host = bundle.finish(outputs, metadata)
        self.assertEqual({"scores", "renamed"}, set(host))
        np.testing.assert_array_equal(before, host["scores"])
        self.assertIs(host["scores"], host["renamed"])
        self.assertEqual({"logits"}, set(outputs))
        np.testing.assert_array_equal(before, outputs["logits"])
        plain = ModelBundle(spec, {"network": graphs["network"]})
        self.assertIsNot(outputs, plain.finish(outputs, metadata))
        self.assertEqual({"logits"}, set(plain.finish(outputs, metadata)))

    def test_explicit_paths_do_not_read_a_neighbouring_model_json(self):
        spec, directory = self.toy()
        atomic_json(
            directory / "model.json",
            dict(
                files={"network.onnx": "0" * 64},
                network_input_map={"images": "other"},
                validation={"state": "strict"},
            ),
        )
        bundle = ModelBundle(spec, {"network": directory / "network.onnx"})
        default = spec.recipe.get("network_input_map", {"images": "images"})
        self.assertEqual(default, bundle.mapping)
        self.assertEqual({}, bundle.validation)
        with self.assertRaisesRegex(ValueError, "Model bundle changed"):
            ModelBundle.from_directory(spec, directory)


if __name__ == "__main__":
    unittest.main()
