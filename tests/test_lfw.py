"""LFW protocol: official pair order, ten folds, flip views, smoke subset and limits."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from opennpu_quant.evaluation.face_verification import verification, read_pairs


def write_pairs(path):
    path.write_text(
        "10\t300\n"
        + "".join("A\t1\t2\n" if i % 600 < 300 else "A\t1\tB\t1\n" for i in range(6000))
    )
    return read_pairs(path)


class LfwProtocolTests(unittest.TestCase):
    def test_lfw_folds_flip_missing_views_and_smoke_units(self):
        with tempfile.TemporaryDirectory() as temp:
            pairs = write_pairs(Path(temp) / "pairs.txt")
        first = np.array([1.0, 0.0], np.float32)
        second = np.array([0.0, 1.0], np.float32)
        vectors = {
            "A/A_0001.jpg": {0: first, 1: first},
            "A/A_0002.jpg": {0: first, 1: first},
            "B/B_0001.jpg": {0: second, 1: second},
        }
        metrics, details = verification(vectors, pairs, dimensions=2)
        self.assertEqual(1.0, metrics["verification_accuracy"])
        self.assertEqual(10, len(details["folds"]))
        self.assertTrue(all(row["threshold"] == 0.01 for row in details["folds"]))
        smoke, proof = verification(
            vectors, [pairs[0], pairs[300]], dimensions=2, smoke=True
        )
        self.assertEqual({"smoke_pair_accuracy": 1.0}, smoke)
        self.assertFalse(proof["full_protocol"])
        with self.assertRaisesRegex(ValueError, "Incomplete/reordered"):
            verification(vectors, [pairs[0], pairs[300]], dimensions=2)
        vectors["A/A_0001.jpg"].pop(1)
        with self.assertRaisesRegex(ValueError, "original and horizontal"):
            verification(vectors, pairs, dimensions=2)


class LfwManifestTests(unittest.TestCase):
    def smoke_manifest(self, root):
        """Two fixed pairs (three images, six views); returns path and pair hash."""
        from PIL import Image
        from opennpu_quant._io import sha256

        pair_path = root / "pairs.txt"
        pairs = write_pairs(pair_path)
        selected = [pairs[0], pairs[300]]
        keys = sorted({p[k] for p in selected for k in ("first", "second")})
        Image.new("RGB", (112, 112)).save(root / "image.png")
        (root / "alignment.json").write_text(
            json.dumps(dict(kind="lfw_mobilefacenet", smoke=True, failures=[]))
        )
        info = dict(
            dataset="lfw_mobilefacenet",
            root=str(root),
            smoke=True,
            expected_samples=6,
            annotations=dict(pairs="pairs.txt", alignment="alignment.json"),
            evaluation_pairs=selected,
            calibration=[
                dict(id="cal", path="image.png", image_key="C/C_0001.jpg", flip=0)
            ],
            validation=[
                dict(id=k + str(f), path="image.png", image_key=k, flip=f)
                for k in keys
                for f in (0, 1)
            ],
        )
        path = root / "manifest.json"
        path.write_text(json.dumps(info))
        return path, sha256(pair_path)

    def test_pair_mutation_is_rejected_and_changes_identity(self):
        from opennpu_quant.data.samples import Dataset

        with tempfile.TemporaryDirectory() as temp:
            path, pair_hash = self.smoke_manifest(Path(temp))
            with patch("opennpu_quant.data.lfw.PAIR_HASH", pair_hash):
                dataset = Dataset(path)
                before = dataset.identity("validation")
                dataset.info["evaluation_pairs"][0]["same"] = False
                self.assertNotEqual(before, dataset.identity("validation"))
                path.write_text(json.dumps(dataset.info))
                with self.assertRaisesRegex(ValueError, "pairs/fold order"):
                    Dataset(path)

    def test_limit_counts_pairs_and_keeps_both_views(self):
        from opennpu_quant.configuration import load_config
        from opennpu_quant.data.samples import Dataset

        with tempfile.TemporaryDirectory() as temp:
            path, pair_hash = self.smoke_manifest(Path(temp))
            with patch("opennpu_quant.data.lfw.PAIR_HASH", pair_hash):
                dataset = Dataset(path)
                rows = dataset.info["validation"]
                # The smoke preset limit selects every view of both fixed pairs,
                # exactly as tools/smoke.py runs it.
                limit = load_config("cnn38_smoke")["evaluation"]["limit"]
                self.assertEqual(rows, dataset.rows("validation", limit))
                self.assertEqual(
                    dataset.identity("validation"),
                    dataset.identity("validation", limit),
                )
                views = {
                    (r["image_key"], r["flip"]) for r in dataset.rows("validation", 1)
                }
                self.assertEqual(
                    {(k, f) for k in ("A/A_0001.jpg", "A/A_0002.jpg") for f in (0, 1)},
                    views,
                )
                self.assertEqual(1, len(dataset.rows("calibration", 1)))

    def test_smoke_preparation_is_lfw_only(self):
        from opennpu_quant.data.prepare import prepare

        with self.assertRaisesRegex(ValueError, "LFW pair subset only"):
            prepare("coco", ".", "manifest.json", smoke=True)

    def test_smoke_metric_cannot_replace_a_regular_recipe_metric(self):
        from types import SimpleNamespace
        from opennpu_quant.runner import _ModelRun

        run = object.__new__(_ModelRun)
        run.spec = SimpleNamespace(
            recipe=dict(task="face_verification", metric="verification_accuracy")
        )
        run.dataset = SimpleNamespace(info=dict(smoke=True))
        run.config = dict(
            report=dict(primary_metric=dict(face_verification="verification_accuracy"))
        )
        with self.assertRaisesRegex(ValueError, "smoke_pair_accuracy"):
            run._primary_metric()
        run.config["report"]["primary_metric"]["face_verification"] = (
            "smoke_pair_accuracy"
        )
        self.assertEqual("smoke_pair_accuracy", run._primary_metric())
        run.spec.recipe = dict(task="classification", metric="top3")
        self.assertEqual("top3", run._primary_metric())
