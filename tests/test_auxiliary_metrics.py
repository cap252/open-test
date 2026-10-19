"""Auxiliary metric schema compatibility and exact fixed-angle arithmetic."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from opennpu_quant.evaluation.face_verification import (
    _fixed_angle_verification,
    verification,
)
from opennpu_quant.reproduction.scoring import original_accuracy
from tests.test_lfw import write_pairs


class AuxiliaryComparisonTests(unittest.TestCase):
    def result(self, actual, expected, details=None):
        return dict(
            stages=dict(full_accuracy="executed"),
            measurement=dict(metrics=expected, samples=6000),
            metrics=actual,
            metric_details=details or {},
            sample_count=6000,
        )

    def compare(self, value):
        return original_accuracy(value, {}, Path("."))

    def test_canonical_fixed_angle_name_is_compared_without_mutating_records(self):
        value = self.result(
            {"fixed_angle_accuracy": 0.99}, {"fixed_angle_accuracy": 0.99}
        )
        before = copy.deepcopy(value)
        check = self.compare(value)
        self.assertEqual("pass", check["metrics_state"])
        self.assertTrue(check["metrics_bitwise_equal"])
        metric = check["metrics"]["fixed_angle_accuracy"]
        self.assertEqual(["metrics.fixed_angle_accuracy"], metric["actual_paths"])
        self.assertEqual(
            ["measurement.metrics.fixed_angle_accuracy"], metric["expected_paths"]
        )
        self.assertEqual(before, value)

    def test_runtime_does_not_translate_old_reference_names(self):
        for name in ("qualcomm_angle_accuracy", "qualcomm_fixed_angle_accuracy"):
            with self.subTest(name=name):
                value = self.result({"fixed_angle_accuracy": 0.99}, {name: 0.99})
                check = self.compare(value)
                self.assertEqual("fail", check["state"])
                self.assertEqual([name], check["missing_metrics"])
                self.assertFalse(check["metrics_bitwise_equal"])
                check = self.compare(
                    self.result({name: 0.99}, {"fixed_angle_accuracy": 0.99})
                )
                self.assertEqual("fail", check["state"])
                self.assertEqual(["fixed_angle_accuracy"], check["missing_metrics"])

    def test_only_documented_details_fill_a_missing_metric(self):
        value = self.result(
            {"mIoU": 0.75},
            {"mIoU": 0.75, "pixel_accuracy": 0.95},
            {"pixel_accuracy": 0.95},
        )
        check = self.compare(value)
        self.assertEqual("pass", check["state"])
        self.assertTrue(check["metrics_bitwise_equal"])
        self.assertEqual(
            ["metric_details.pixel_accuracy"],
            check["metrics"]["pixel_accuracy"]["actual_paths"],
        )
        value["metric_details"] = {"mIoU": 0.75}
        check = self.compare(value)
        self.assertEqual("fail", check["state"])
        self.assertEqual(["pixel_accuracy"], check["missing_metrics"])
        self.assertFalse(check["metrics_bitwise_equal"])
        value["metrics"] = {}
        self.assertIn("mIoU", self.compare(value)["missing_metrics"])

    def test_conflicting_diagnostic_locations_never_pass(self):
        for actual, details in (
            (0.95, 0.9),
            (0.95, np.nextafter(0.95, 1.0)),
            (0.0, -0.0),
        ):
            with self.subTest(actual=actual, details=details):
                value = self.result(
                    {"pixel_accuracy": actual},
                    {"pixel_accuracy": actual},
                    {"pixel_accuracy": details},
                )
                check = self.compare(value)
                self.assertEqual("fail", check["state"])
                self.assertEqual(
                    ["pixel_accuracy"], check["conflicting_metrics"]["actual"]
                )
                self.assertFalse(check["metrics_bitwise_equal"])
        value["metric_details"]["pixel_accuracy"] = value["metrics"]["pixel_accuracy"]
        self.assertEqual("pass", self.compare(value)["state"])

    def test_tolerance_does_not_claim_bitwise_or_override_sample_count(self):
        value = self.result(
            {"fixed_angle_accuracy": np.nextafter(0.99, 1.0)},
            {"fixed_angle_accuracy": 0.99},
        )
        check = self.compare(value)
        self.assertEqual("pass", check["state"])
        self.assertFalse(check["metrics_bitwise_equal"])
        value["metrics"]["fixed_angle_accuracy"] = 0.9
        self.assertEqual("fail", self.compare(value)["state"])
        value["metrics"]["fixed_angle_accuracy"] = 0.99
        value["sample_count"] -= 1
        self.assertEqual("fail", self.compare(value)["state"])
        value["stages"]["full_accuracy"] = "pending"
        self.assertEqual("pending", self.compare(value)["state"])

    def test_nonfinite_diagnostic_is_rejected(self):
        value = self.result(
            {}, {"pixel_accuracy": 0.95}, {"pixel_accuracy": float("nan")}
        )
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            self.compare(value)


class FixedAngleArithmeticTests(unittest.TestCase):
    def test_reference_row_normalization_bits_and_input_preservation(self):
        rng = np.random.default_rng(143)
        first = rng.standard_normal((6, 128)).astype(np.float32)
        second = rng.standard_normal((6, 128)).astype(np.float32)
        before = first.tobytes(), second.tobytes()
        accuracy, angles = _fixed_angle_verification(first, second, np.zeros(6, bool))
        self.assertEqual(1.0, accuracy)
        self.assertEqual(
            "d486b8429606c1424a8ab6422b58a64275afb042c737ab42",
            angles.tobytes().hex(),
        )
        self.assertEqual(before, (first.tobytes(), second.tobytes()))

    def test_threshold_compares_stored_angles_as_float64(self):
        first = np.ones((3, 128), np.float32)
        boundary = np.float32(74.18)
        angles = np.array(
            [np.nextafter(boundary, np.float32(0)), boundary, 90], np.float32
        )
        self.assertGreater(float(boundary), 74.18)
        with patch(
            "opennpu_quant.evaluation.face_verification.np.degrees", return_value=angles
        ):
            accuracy, _ = _fixed_angle_verification(
                first, first, np.array([True, False, False])
            )
        self.assertEqual(1.0, accuracy)

    def test_full_protocol_emits_canonical_metric_with_reference_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            pairs = write_pairs(Path(temp) / "pairs.txt")
        first = np.eye(1, 128, 0, dtype=np.float32)[0]
        second = np.eye(1, 128, 1, dtype=np.float32)[0]
        vectors = {
            "A/A_0001.jpg": {0: first, 1: first},
            "A/A_0002.jpg": {0: first, 1: first},
            "B/B_0001.jpg": {0: second, 1: second},
        }
        for pair in pairs[300:356]:
            pair["second"] = "A/A_0002.jpg"
        metrics, details = verification(vectors, pairs, dimensions=128)
        self.assertEqual(0.9906666666666666, metrics["fixed_angle_accuracy"])
        self.assertNotEqual(5944 / 6000, metrics["fixed_angle_accuracy"])
        self.assertEqual(74.18, details["fixed_angle_threshold_degrees"])
        self.assertFalse(any("qualcomm" in k for k in metrics.keys() | details.keys()))
        baseline, baseline_details = verification(
            {k: {i: v[:2] for i, v in views.items()} for k, views in vectors.items()},
            pairs,
            dimensions=2,
        )
        self.assertEqual(baseline_details["folds"], details["folds"])
        for key, value in baseline.items():
            self.assertEqual(
                np.float64(value).tobytes(), np.float64(metrics[key]).tobytes()
            )


class SavedAuxiliaryCheckTests(unittest.TestCase):
    def test_saved_labels_replay_without_inference_or_result_rewrite(self):
        from PIL import Image
        from tools.check_auxiliary_metrics import run
        from opennpu_quant._io import atomic_json, object_hash, sha256
        from opennpu_quant.reproduction.assets import file_record, seal, unseal

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "deeplabv3_mnv2_voc__fp32"
            folder.mkdir()
            Image.fromarray(np.array([[0, 1], [1, 255]], np.uint8)).save(
                root / "mask.png"
            )
            atomic_json(
                root / "dataset.json",
                dict(validation=[dict(id="one", target="mask.png")]),
            )
            source_path = root / "source.json"
            expected = dict(mIoU=1.0, pixel_accuracy=1.0)
            atomic_json(source_path, dict(result=dict(metrics=expected)))
            entry = dict(
                id=folder.name,
                model="deeplabv3_mnv2_voc",
                condition="fp32",
                measurement=dict(metrics=expected, samples=1),
                recipe={},
                graphs={},
                source_result=dict(
                    id="artifact://test/source.json", sha256=sha256(source_path)
                ),
                dataset=dict(
                    manifest=dict(
                        id="artifact://test/dataset.json",
                        sha256=sha256(root / "dataset.json"),
                    ),
                    root="artifact://test/",
                ),
            )
            prediction = folder / "prediction.json"
            atomic_json(prediction, dict(labels=[[0, 1], [1, 0]]))
            previous = dict(
                id=entry["id"],
                model=entry["model"],
                task="segmentation",
                stages=dict(full_accuracy="executed"),
                measurement=entry["measurement"],
                source_graphs=entry["graphs"],
                recipe_identity=object_hash(entry["recipe"]),
                input_identity="unchanged",
                environment=dict(implementation="old-package"),
                metrics=dict(mIoU=1.0),
                primary_metric="mIoU",
                sample_count=1,
                samples=[dict(id="one", prediction=file_record(prediction, folder))],
                original_accuracy=dict(
                    state="fail", missing_metrics=["pixel_accuracy"]
                ),
            )
            path = folder / "result.json"
            seal(path, previous)
            seal(
                root / "run.json",
                dict(
                    conditions=[
                        dict(
                            id=entry["id"],
                            state="completed",
                            result=file_record(path, root),
                        )
                    ]
                ),
            )
            output = root / "new-evidence.json"
            with (
                patch(
                    "tools.check_auxiliary_metrics.catalog",
                    return_value=dict(entries=[entry]),
                ),
                patch(
                    "onnxruntime.InferenceSession",
                    side_effect=AssertionError("No inference"),
                ),
            ):
                self.assertTrue(run(root, {"test": str(root)}, output))
                proof = unseal(output)
                self.assertTrue(proof["passed"])
                self.assertEqual("not_run", proof["gpu_execution"])
                self.assertEqual(
                    1.0, proof["conditions"][0]["metric_details"]["pixel_accuracy"]
                )
                self.assertEqual(previous, unseal(path))
                with self.assertRaises(FileExistsError):
                    run(root, {"test": str(root)}, output)
                entry["measurement"]["details"] = {
                    "fixed_angle_threshold_degrees": 74.18
                }
                previous["measurement"] = copy.deepcopy(entry["measurement"])
                previous["measurement"]["details"] = {
                    "qualcomm_angle_threshold_degrees": 74.18
                }
                seal(path, previous)
                seal(
                    root / "run.json",
                    dict(
                        conditions=[
                            dict(
                                id=entry["id"],
                                state="completed",
                                result=file_record(path, root),
                            )
                        ]
                    ),
                )
                with self.assertRaisesRegex(ValueError, "migrate-metadata"):
                    run(root, {"test": str(root)}, root / "implicit-migration.json")
                migrated = root / "explicit-migration.json"
                self.assertTrue(
                    run(root, {"test": str(root)}, migrated, migrate_metadata=True)
                )
                self.assertTrue(
                    unseal(migrated)["conditions"][0]["metadata_migrations"]
                )
                self.assertEqual(previous, unseal(path))
                previous["measurement"]["metrics"]["mIoU"] = np.nextafter(
                    1.0, 0.0
                ).item()
                seal(path, previous)
                seal(
                    root / "run.json",
                    dict(
                        conditions=[
                            dict(
                                id=entry["id"],
                                state="completed",
                                result=file_record(path, root),
                            )
                        ]
                    ),
                )
                with self.assertRaisesRegex(ValueError, "identities differ"):
                    run(
                        root,
                        {"test": str(root)},
                        root / "different-value.json",
                        migrate_metadata=True,
                    )
                previous["measurement"] = copy.deepcopy(entry["measurement"])
                seal(path, previous)
                seal(
                    root / "run.json",
                    dict(
                        conditions=[
                            dict(
                                id=entry["id"],
                                state="completed",
                                result=file_record(path, root),
                            )
                        ]
                    ),
                )
                atomic_json(prediction, dict(labels=[[1, 1], [1, 0]]))
                with self.assertRaisesRegex(ValueError, "Run artifact changed"):
                    run(root, {"test": str(root)}, root / "tampered.json")
                self.assertFalse((root / "tampered.json").exists())
