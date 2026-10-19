"""Fixed-graph replay contracts, independent of external research assets or CUDA."""

import copy
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
from PIL import Image

from opennpu_quant._io import atomic_json, read_json, sha256
from opennpu_quant.models.spec import ModelSpec
from opennpu_quant.reproduction.assets import unseal, file_record, catalog, Resolver
from opennpu_quant.reproduction.prepare import prepare
from opennpu_quant.reproduction.execution import run
from opennpu_quant.reproduction.compare import compare, _pair
from opennpu_quant.reproduction.environment import compare as compare_environment
from opennpu_quant.reproduction.checks import structured


def fixture(root):
    source = root / "assets"
    source.mkdir()
    h, t = onnx.helper, onnx.TensorProto
    model = h.make_model(
        h.make_graph(
            [
                h.make_node("GlobalAveragePool", ["images"], ["avg"]),
                h.make_node("Flatten", ["avg"], ["flat"], axis=1),
                h.make_node("MatMul", ["flat", "weight"], ["logits"]),
            ],
            "fixture",
            [h.make_tensor_value_info("images", t.FLOAT, [1, 3, 4, 4])],
            [h.make_tensor_value_info("logits", t.FLOAT, [1, 6])],
            [
                onnx.numpy_helper.from_array(
                    np.arange(18, dtype=np.float32).reshape(3, 6), "weight"
                )
            ],
        ),
        opset_imports=[h.make_opsetid("", 18)],
        ir_version=10,
    )
    onnx.save(model, source / "model.onnx")
    atomic_json(source / "source.json", {"origin": "fixture"})
    rows = []
    for i in range(3):
        Image.new("RGB", (8 + i, 8), (80 + i, 100, 120)).save(source / f"{i}.png")
        rows.append(dict(id=i, path=f"{i}.png", target=0))
    atomic_json(
        source / "data.json",
        dict(
            dataset="imagenet",
            root="unused",
            validation=rows,
            calibration=[],
            expected_samples=3,
        ),
    )

    def asset(name):
        return dict(id="artifact://assets/" + name, sha256=sha256(source / name))

    recipe = ModelSpec.load("resnet18").recipe
    recipe["input"]["shape"] = [1, 3, 4, 4]
    recipe["preprocess"].update(resize=4, crop=4)
    entry = dict(
        id="resnet18__fp32",
        model="resnet18",
        condition="fp32",
        recipe=recipe,
        graphs={"network": asset("model.onnx")},
        source_result=asset("source.json"),
        dataset=dict(manifest=asset("data.json"), root="artifact://assets/"),
        execution=dict(gpu_mem_limit=6442450944),
        measurement={},
        reference=None,
    )
    atomic_json(root / "selection.json", dict(entries=[entry]))
    atomic_json(root / "bindings.json", dict(assets=str(source)))
    return source


class GPUReproductionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = fixture(self.root)
        self.prepare = self.root / "prepared"
        prepare(
            bindings=self.root / "bindings.json",
            selection=self.root / "selection.json",
            output=self.prepare,
        )

    def execute(self, output="run"):
        return run(
            package=self.prepare,
            bindings=self.root / "bindings.json",
            output=self.root / output,
        )

    def test_catalog_is_38_times_seven_and_preserves_selected_policy(self):
        entries = catalog()["entries"]
        self.assertEqual(266, len(entries))
        self.assertEqual(38, len({e["model"] for e in entries}))
        self.assertFalse(
            any(
                e["model"] in ("quicksrnet_m", "arcface_mobilefacenet_webface600k")
                for e in entries
            )
        )
        y9 = {e["condition"]: e for e in entries if e["model"] == "yolov9_t"}
        self.assertEqual(99.9, y9["asym_pc_pot_rtn"]["percentile"])
        self.assertEqual(99.99, y9["asym_pc_float_rtn"]["percentile"])
        self.assertEqual(
            {6 * 1024**3, 8 * 1024**3},
            {e["execution"]["gpu_mem_limit"] for e in entries if e["model"] == "vgg16"},
        )

    def test_end_to_end_and_verified_completed_reuse(self):
        self.assertEqual(1, self.execute()["completed"])
        result = self.root / "run/resnet18__fp32/result.json"
        stamp = result.stat().st_mtime_ns
        self.assertEqual(1, self.execute()["completed"])
        self.assertEqual(stamp, result.stat().st_mtime_ns)
        c = compare(
            actual=self.root / "run",
            expected=self.root / "run",
            output=self.root / "compare",
        )
        self.assertEqual({"pass": 1}, c["counts"])
        self.assertEqual({"pending": 1}, c["gpu_comparison_counts"])

    def test_input_and_model_tamper_prevent_completed_reuse(self):
        self.execute()
        p = self.source / "0.png"
        original = p.read_bytes()
        Image.new("RGB", (8, 8), (0, 0, 0)).save(p)
        self.assertEqual(1, self.execute()["failed"])
        p.write_bytes(original)
        with (self.source / "model.onnx").open("ab") as stream:
            stream.write(b"changed")
        self.assertEqual(1, self.execute()["failed"])

    def test_profile_or_manifest_change_cannot_resume(self):
        self.execute()
        p = self.prepare / "manifest.json"
        info = unseal(p)
        info["conditions"][0]["entry"]["recipe"]["preprocess"]["mean"][0] = 0
        atomic_json(p, info)
        with self.assertRaisesRegex(ValueError, "changed"):
            self.execute()

    def test_interruption_resumes_completed_samples(self):
        from opennpu_quant.models.adapter import ModelEvaluator

        update = ModelEvaluator.update
        calls = []

        def interrupted(evaluator, values, sample):
            calls.append(sample.sample_id)
            if len(calls) == 2:
                raise RuntimeError("simulated interruption")
            return update(evaluator, values, sample)

        with patch.object(ModelEvaluator, "update", interrupted):
            self.assertEqual(1, self.execute()["failed"])
        saved = self.root / "run/resnet18__fp32/000000.npz"
        stamp = saved.stat().st_mtime_ns
        self.assertEqual(1, self.execute()["completed"])
        self.assertEqual(stamp, saved.stat().st_mtime_ns)

    def test_relocation_only_needs_new_bindings(self):
        self.execute()
        moved = self.root / "moved-assets"
        shutil.move(str(self.source), moved)
        atomic_json(self.root / "bindings.json", dict(assets=str(moved)))
        self.assertEqual(1, self.execute()["completed"])

    def test_missing_artifact_and_missing_reference_remain_pending(self):
        self.assertEqual(
            "pending",
            unseal(self.prepare / "manifest.json")["conditions"][0]["gpu_reference"],
        )
        (self.source / "model.onnx").unlink()
        result = prepare(
            bindings=self.root / "bindings.json",
            selection=self.root / "selection.json",
            output=self.root / "missing",
        )
        self.assertEqual({"pending": 1}, result["counts"])

    def test_tolerant_nonbitwise_alert_does_not_hide_prediction_failure(self):
        self.execute()
        folder = self.root / "run/resnet18__fp32"
        expected = unseal(folder / "result.json")
        actual = copy.deepcopy(expected)
        other = self.root / "other"
        shutil.copytree(folder, other)
        p = other / "000000.npz"
        with np.load(p) as z:
            values = z["logits"].copy()
        values.flat[0] = np.nextafter(values.flat[0], np.float32(np.inf))
        np.savez(p, logits=values)
        actual["samples"][0]["raw"] = file_record(p, other)
        check = _pair(actual, expected, other, folder)
        self.assertEqual("pass", check["state"])
        self.assertEqual(1, len(check["bitwise_alerts"]))
        p = other / "000000.prediction.json"
        prediction = read_json(p)
        prediction["top5"].reverse()
        atomic_json(p, prediction)
        actual["samples"][0]["prediction"] = file_record(p, other)
        check = _pair(actual, expected, other, folder)
        self.assertEqual("pass", check["raw"])
        self.assertEqual("fail", check["state"])

    def test_library_not_loaded_is_distinct_from_content_changed(self):
        env = dict(
            device="cuda:0",
            packages={},
            execution={},
            loaded_libraries={"libraries": [dict(name="libcudart.so.12", sha256="a")]},
        )
        other = copy.deepcopy(env)
        other["loaded_libraries"]["libraries"][0]["sha256"] = "b"
        self.assertEqual("fail", compare_environment(env, other)["state"])
        other["loaded_libraries"]["libraries"] = []
        self.assertEqual("pending", compare_environment(env, other)["state"])

    def test_discrete_predictions_and_paths_are_exact(self):
        self.assertFalse(structured({"labels": [[1, 2]]}, {"labels": [[1, 3]]}))
        self.assertTrue(structured({"score": 0.5}, {"score": 0.50000001}))
        self.assertFalse(structured({"score": float("nan")}, {"score": float("nan")}))
        with self.assertRaisesRegex(ValueError, "escapes"):
            Resolver(dict(assets=str(self.source))).path("artifact://assets/../outside")

    def test_wheel_library_resolution_uses_active_distribution(self):
        from opennpu_quant.ort.cuda_libs import wheel_library
        from types import SimpleNamespace

        path = self.root / "external/nvidia/cuda_runtime/lib/libcudart.so.12"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"fixture")
        distribution = SimpleNamespace(
            locate_file=lambda name: self.root / "external" / name
        )
        with patch("importlib.metadata.distribution", return_value=distribution):
            self.assertEqual(
                path,
                wheel_library("cuda_runtime/lib/libcudart.so.12", self.root / "unused"),
            )

    def test_missing_optional_gpu_raw_does_not_block_baseline_capture(self):
        selection = read_json(self.root / "selection.json")
        selection["entries"][0]["reference"] = {
            "manifest": {"id": "artifact://assets/missing.json", "sha256": "0" * 64},
            "root": "artifact://assets/",
        }
        atomic_json(self.root / "selection.json", selection)
        result = prepare(
            bindings=self.root / "bindings.json",
            selection=self.root / "selection.json",
            output=self.root / "no_reference",
        )
        self.assertEqual({"ready": 1}, result["counts"])
        self.assertEqual({"pending": 1}, result["gpu_references"])
        self.prepare = self.root / "no_reference"
        self.assertEqual(1, self.execute()["completed"])

    def test_resealed_stale_profile_identity_is_rejected(self):
        from opennpu_quant.reproduction.assets import seal

        p = self.prepare / "manifest.json"
        manifest = unseal(p)
        manifest["conditions"][0]["entry"]["recipe"]["preprocess"]["mean"][0] = 0
        seal(p, manifest)
        self.assertEqual(1, self.execute()["failed"])

    def test_repeat_bitwise_failure_cannot_be_overridden_by_tolerance(self):
        self.execute()
        folder = self.root / "run/resnet18__fp32"
        expected = unseal(folder / "result.json")
        actual = copy.deepcopy(expected)
        actual["repeat_bitwise"] = False
        result = _pair(actual, expected, folder, folder)
        self.assertEqual("pass", result["raw"])
        self.assertEqual("fail", result["state"])

    def test_failed_gpu_attempt_keeps_existing_run_index(self):
        self.execute()
        original = (self.root / "run/run.json").read_bytes()
        with patch(
            "opennpu_quant.reproduction.environment.probe_gpu",
            return_value={"available": False, "diagnostic": "unavailable"},
        ):
            result = run(
                package=self.prepare,
                bindings=self.root / "bindings.json",
                output=self.root / "run",
                device="cuda:0",
            )
        self.assertEqual("pending", result["state"])
        self.assertEqual(original, (self.root / "run/run.json").read_bytes())

    def test_missing_raw_reference_is_pending_and_input_binding_changes_fail(self):
        from opennpu_quant.reproduction.execution import reference_diagnostic

        descriptor = dict(
            root="artifact://assets/",
            samples=[dict(id="0", inputs="missing.npz", inputs_sha256="a")],
        )
        refs = dict(
            samples=[
                dict(id="0", inputs_sha256="a", outputs="out.npz", outputs_sha256="b")
            ],
            output_names=["logits"],
        )
        result = reference_diagnostic(
            (descriptor, refs),
            Resolver(self.root / "bindings.json"),
            "0",
            0,
            "digest",
            {},
            ["logits"],
        )
        self.assertEqual("pending", result["reference_state"])
        refs["samples"][0]["inputs_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "binding"):
            reference_diagnostic(
                (descriptor, refs),
                Resolver(self.root / "bindings.json"),
                "0",
                0,
                "digest",
                {},
                ["logits"],
            )

    def test_self_comparison_never_certifies_cross_gpu(self):
        self.execute()
        folder = self.root / "run/resnet18__fp32"
        value = unseal(folder / "result.json")
        value["environment"]["device"] = "cuda:0"
        value["environment"]["device_identity"] = {"uuid": "same-device"}
        result = _pair(value, value, folder, folder)
        self.assertEqual("pending", result["gpu_comparison"])

    def test_unidentified_callback_cannot_certify_completed_reuse(self):
        selection = read_json(self.root / "selection.json")
        selection["entries"][0]["recipe"]["preprocess"]["kind"] = "external:preprocess"
        atomic_json(self.root / "selection.json", selection)
        result = prepare(
            bindings=self.root / "bindings.json",
            selection=self.root / "selection.json",
            output=self.root / "callback",
        )
        self.assertEqual({"fail": 1}, result["counts"])

    def test_duplicate_library_names_cannot_hide_changed_content(self):
        from opennpu_quant.ort.cuda_libs import compare_libraries

        expected = {"libraries": [dict(name="libcublas.so.12", sha256="a")]}
        actual = {
            "libraries": [
                dict(name="libcublas.so.12", sha256="changed"),
                dict(name="libcublas.so.12", sha256="a"),
            ]
        }
        self.assertEqual(
            "content_changed", compare_libraries(actual, expected)["libcublas.so.12"]
        )
        self.assertEqual(
            "unverifiable", compare_libraries(actual, actual)["libcublas.so.12"]
        )

    def test_gpu_environment_is_sealed_after_postprocessing_library_load(self):
        from opennpu_quant.reproduction import execution

        initialized = []
        captures = []

        def capture(**kwargs):
            captures.append(bool(initialized))
            return {"version_mismatches": {}, "packages": {}}

        with (
            patch(
                "opennpu_quant.reproduction.environment.probe_gpu",
                return_value={"available": True},
            ),
            patch.object(
                execution,
                "initialize_postprocessing",
                side_effect=lambda: initialized.append(True),
            ),
            patch.object(execution, "capture", side_effect=capture),
            patch.object(
                execution, "_condition", side_effect=RuntimeError("fixture stop")
            ),
        ):
            run(
                package=self.prepare,
                bindings=self.root / "bindings.json",
                output=self.root / "gpu",
                device="cuda:0",
            )
        self.assertEqual([False, True], captures)

    def test_postprocessing_warmup_keeps_optional_dependency_optional(self):
        from opennpu_quant.reproduction.execution import initialize_postprocessing

        with patch("importlib.import_module", side_effect=ImportError):
            initialize_postprocessing()
        with patch(
            "importlib.import_module",
            side_effect=RuntimeError("incompatible native library"),
        ):
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                initialize_postprocessing()
