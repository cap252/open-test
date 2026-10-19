import copy
import hashlib
from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from opennpu_quant._io import atomic_json, read_json, sha256
from opennpu_quant.models.evaluation_profiles import (
    HISTORICAL,
    apply_profile,
    decode,
    profile_name,
)
from opennpu_quant.models.spec import ModelSpec
from opennpu_quant.reproduction.assets import catalog, unseal
from opennpu_quant.reproduction.prepare import prepare
from opennpu_quant.reproduction.execution import run
from opennpu_quant.reproduction.compare import compare, _pair
from opennpu_quant.reproduction.settings import training_config
from opennpu_quant.reproduction.scoring import original_accuracy
from tests.test_gpu_reproduction import fixture


class FinalReproductionTests(unittest.TestCase):
    def test_yolov9_family_and_final_selection(self):
        outputs = {"predictions": np.zeros((1, 84, 3), np.float32)}
        outputs["predictions"][0, :4] = [[5, 5, 25], [5, 5, 25], [8, 8, 8], [8, 8, 8]]
        outputs["predictions"][0, 4] = [0.123456, 0.123455, 0.1]
        before = outputs["predictions"].copy()
        meta = dict(
            original_size=[40, 40], input_size=[40, 40], pad=[0, 0], scale=[1, 1]
        )
        recipes = [
            apply_profile(ModelSpec.load(m).recipe, HISTORICAL)
            for m in ("yolov9_t", "yolov9_s")
        ]
        actual = [decode(outputs, r, meta, list(range(80))) for r in recipes]
        self.assertEqual(actual[0], actual[1])
        self.assertEqual(0.12346, actual[1][0]["score"])
        np.testing.assert_array_equal(before, outputs["predictions"])
        self.assertEqual(HISTORICAL, profile_name(ModelSpec.load("yolov9_s").recipe))
        entries = [e for e in catalog()["entries"] if e["model"] == "yolov9_s"]
        self.assertEqual(7, len(entries))
        self.assertTrue(all(profile_name(e["recipe"]) == HISTORICAL for e in entries))
        self.assertEqual(
            {99.99}, {e["percentile"] for e in entries if e["condition"] != "fp32"}
        )

    def test_condition_training_settings_preserve_source(self):
        selection = catalog()
        for entry in selection["entries"]:
            config = training_config(
                entry, selection["training_templates"][entry["model"]]
            )
            self.assertEqual(
                entry["execution"]["gpu_mem_limit"],
                config["runtime"]["cuda"]["gpu_mem_limit"],
            )
            self.assertEqual([entry["condition"]], list(config["conditions"]))
            self.assertEqual(
                profile_name(entry["recipe"]),
                config["evaluation"]["profiles"][entry["model"]],
            )
            self.assertEqual(
                selection["training_input_references"][entry["model"]][
                    "samples_identity"
                ],
                config["calibration"]["expected_samples_identity"][entry["model"]],
            )
            if entry.get("adaround"):
                self.assertEqual(entry["adaround"], config["adaround"])
            if entry["condition"] != "fp32":
                self.assertEqual(
                    entry["percentile"],
                    config["percentiles"][entry["model"]][entry["condition"]],
                )

    def test_research_topk_ties_preserve_public_default(self):
        from opennpu_quant.evaluation.metrics import TopKClassification
        from opennpu_quant.models.adapter import ModelEvaluator
        from opennpu_quant.models.contracts import EvaluationSample
        from opennpu_quant.models.evaluation_profiles import NUMPY_TOPK

        logits = {"logits": np.ones((1, 6), np.float32)}
        sample = EvaluationSample("0", {}, 5)
        default = TopKClassification()
        default.update(logits, sample)
        self.assertEqual({"top1": 0.0, "top5": 0.0}, default.finalize())
        recipe = apply_profile(ModelSpec.load("resnet18").recipe, NUMPY_TOPK)
        bundle = SimpleNamespace(spec=ModelSpec(recipe), finish=lambda out, meta: out)
        evaluator = ModelEvaluator(bundle, None)
        evaluator.update(logits, sample)
        self.assertEqual({"top1": 1.0, "top5": 1.0}, evaluator.finalize())
        self.assertEqual([5, 4, 3, 2, 1], evaluator.prediction["top5"])
        entries = [
            e for e in catalog()["entries"] if e["recipe"]["task"] == "classification"
        ]
        self.assertEqual(91, len(entries))
        self.assertTrue(all(profile_name(e["recipe"]) == NUMPY_TOPK for e in entries))

    def test_training_input_guard_rejects_drift_before_calibration(self):
        from opennpu_quant._feeds import feed_digest
        from opennpu_quant.configuration import load_config, validate
        from opennpu_quant.runner import _ModelRun

        feed = {"images": np.zeros((1, 3, 2, 2), np.float32)}
        digest = hashlib.sha256(feed_digest(feed).encode()).hexdigest()
        config = load_config("quick")
        name = config["models"][0]
        config["calibration"]["expected_samples_identity"] = {name: "bad"}
        with self.assertRaisesRegex(ValueError, "SHA256"):
            validate(config)
        config["calibration"]["input_cache"] = {"enabled": False}
        config["calibration"]["expected_samples_identity"] = {name: "0" * 64}
        validate(config)
        fake = SimpleNamespace(
            statistics=None, config=config, name=name, factory=lambda: iter([feed])
        )
        with patch("opennpu_quant.runner.calibrate") as calibrate:
            with self.assertRaisesRegex(ValueError, "Calibration inputs differ"):
                _ModelRun._collect_statistics(fake)
            calibrate.assert_not_called()
        config["calibration"]["expected_samples_identity"][name] = digest
        config["calibration"]["device"] = "cpu"
        with patch(
            "opennpu_quant.runner.runtime", side_effect=RuntimeError("guard accepted")
        ):
            with self.assertRaisesRegex(RuntimeError, "guard accepted"):
                _ModelRun._collect_statistics(fake)

    def test_pose_training_split_is_separate_and_pinned(self):
        from opennpu_quant.data.prepare import identifiers

        pose = identifiers("coco_keypoints")
        self.assertEqual(128, len(pose))
        self.assertEqual(["2459", "2697"], [r[0] for r in pose[:2]])
        self.assertNotEqual(pose, identifiers("coco"))
        self.assertEqual(
            "posenet", ModelSpec.load("posenet_101").recipe["preprocess"]["kind"]
        )

    def test_full_resolves_mixed_dataset_counts_and_reuses(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            assets = fixture(root)
            selection = read_json(root / "selection.json")
            other = copy.deepcopy(selection["entries"][0])
            other.update(id="resnet50__fp32", model="resnet50")
            info = read_json(assets / "data.json")
            info.update(validation=info["validation"][:2], expected_samples=2)
            atomic_json(assets / "second.json", info)
            other["dataset"]["manifest"] = dict(
                id="artifact://assets/second.json",
                sha256=sha256(assets / "second.json"),
            )
            selection["entries"].append(other)
            atomic_json(root / "selection.json", selection)
            prepare(
                bindings=root / "bindings.json",
                selection=root / "selection.json",
                output=root / "prepared",
                limit="full",
            )
            self.assertEqual(
                [3, 2],
                [
                    r["limit"]
                    for r in unseal(root / "prepared/manifest.json")["conditions"]
                ],
            )
            args = dict(
                package=root / "prepared",
                bindings=root / "bindings.json",
                output=root / "run",
            )
            self.assertEqual(2, run(**args)["completed"])
            self.assertEqual(2, run(**args)["completed"])
            self.assertEqual(
                "executed",
                unseal(root / "run/resnet50__fp32/result.json")["stages"][
                    "full_accuracy"
                ],
            )

    def test_full_rejects_incomplete_declared_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            assets = fixture(root)
            info = read_json(assets / "data.json")
            info["expected_samples"] = 4
            atomic_json(assets / "data.json", info)
            selection = read_json(root / "selection.json")
            selection["entries"][0]["dataset"]["manifest"]["sha256"] = sha256(
                assets / "data.json"
            )
            atomic_json(root / "selection.json", selection)
            result = prepare(
                bindings=root / "bindings.json",
                selection=root / "selection.json",
                output=root / "prepared",
                limit="full",
            )
            self.assertEqual({"fail": 1}, result["counts"])

    def test_compact_resume_and_comparison_do_not_invent_raw_tolerance(self):
        from opennpu_quant.models.adapter import ModelEvaluator

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            prepare(
                bindings=root / "bindings.json",
                selection=root / "selection.json",
                output=root / "prepared",
            )
            args = dict(package=root / "prepared", bindings=root / "bindings.json")
            original = ModelEvaluator.update
            calls = []

            def interrupted(evaluator, values, sample):
                calls.append(sample.sample_id)
                if len(calls) == 2:
                    raise RuntimeError("interruption")
                return original(evaluator, values, sample)

            with patch.object(ModelEvaluator, "update", interrupted):
                self.assertEqual(
                    1,
                    run(**args, output=root / "compact", storage="predictions")[
                        "failed"
                    ],
                )
            folder = root / "compact/resnet18__fp32"
            stamp = (folder / "000000.prediction.json").stat().st_mtime_ns
            self.assertEqual(
                1,
                run(**args, output=root / "compact", storage="predictions")[
                    "completed"
                ],
            )
            self.assertEqual(
                stamp, (folder / "000000.prediction.json").stat().st_mtime_ns
            )
            self.assertFalse(list(folder.glob("*.npz")))
            self.assertEqual(1, run(**args, output=root / "raw")["completed"])
            check = compare(
                actual=root / "compact", expected=root / "raw", output=root / "compare"
            )
            self.assertEqual({"pass": 1}, check["counts"])
            actual = unseal(folder / "result.json")
            other = copy.deepcopy(actual)
            other["samples"][0]["raw_digest"]["logits"]["sha256"] = "0" * 64
            self.assertEqual("pending", _pair(actual, other, folder, folder)["raw"])
            (folder / "000000.prediction.json").write_text("{}")
            self.assertEqual(
                1, run(**args, output=root / "compact", storage="predictions")["failed"]
            )

    def test_original_accuracy_is_independent_of_execution_completion(self):
        value = dict(stages=dict(full_accuracy="pending"))
        self.assertEqual("pending", original_accuracy(value, {}, Path("."))["state"])
        value.update(
            stages=dict(full_accuracy="executed"),
            measurement=dict(metrics={"AP": 0.5}, samples=5000),
            metrics={"AP": 0.4},
            sample_count=5000,
        )
        self.assertEqual("fail", original_accuracy(value, {}, Path("."))["state"])
        value["metrics"]["AP"] = 0.5
        self.assertEqual("pass", original_accuracy(value, {}, Path("."))["state"])

    def test_original_stream_encoding_and_missing_numeric_evidence(self):
        import json
        from opennpu_quant.reproduction.assets import file_record

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prediction = [dict(category_id=1, bbox=[1, 2, 3, 4], score=0.5)]
            atomic_json(root / "prediction.json", prediction)
            sample = dict(
                id="139", prediction=file_record(root / "prediction.json", root)
            )
            result = dict(
                model="posenet_101",
                task="keypoints",
                stages=dict(full_accuracy="executed"),
                measurement=dict(metrics={"AP": 0.5}, samples=1),
                metrics={"AP": 0.5},
                sample_count=1,
                samples=[sample],
            )
            record = dict(image_id=139, detections=prediction)
            digest = hashlib.sha256(
                (json.dumps(record, separators=(",", ":")) + "\n").encode()
            ).hexdigest()
            source = dict(prediction_evidence=dict(all_predictions_sha256=digest))
            self.assertEqual("pass", original_accuracy(result, source, root)["state"])
            source["prediction_evidence"]["all_predictions_sha256"] = "0" * 64
            answer = original_accuracy(result, source, root)
            self.assertEqual("pass", answer["metrics_state"])
            self.assertEqual("pending", answer["state"])
            result["task"] = "classification"
            self.assertEqual(
                "pending", original_accuracy(result, source, root)["prediction_stream"]
            )
