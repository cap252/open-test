"""User-facing help and input errors without changing valid execution contracts."""

import contextlib
import copy
import io
import json
from tests.library_fixtures import small_config

from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from opennpu_quant.cli import main
from opennpu_quant.configuration import validate


class CliUsabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def invoke(self, args):
        stdout, stderr = io.StringIO(), io.StringIO()
        code = 0
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                main(args)
            except SystemExit as error:
                code = error.code
        return code, stdout.getvalue(), stderr.getvalue()

    def config_file(self, config):
        path = self.root / "experiment.yaml"
        path.write_text(yaml.safe_dump(config))
        return str(path)

    def assert_error(self, args, *words):
        code, _, message = self.invoke(args)
        self.assertEqual(2, code, message)
        self.assertNotIn("Traceback", message)
        for word in words:
            self.assertIn(word, message)
        return message

    def test_demo_prints_only_json_and_validates_its_device(self):
        code, out, _ = self.invoke(["demo", "--device", "cpu"])
        self.assertEqual(0, code)
        conditions = [row["condition"] for row in json.loads(out)["results"]]
        self.assertEqual(["float_rtn", "pot_rtn"], conditions)
        self.assert_error(
            ["demo", "--device", "cuda"], "--device must be cpu or cuda:N"
        )

    def test_gpu_defaults_and_explicit_cpu_settings(self):
        from opennpu_quant.cli import build_parser

        parser = build_parser()
        self.assertEqual("cuda:0", parser.parse_args(["demo"]).device)
        replay = parser.parse_args(
            [
                "reproduce",
                "run",
                "--package",
                "package",
                "--bindings",
                "bindings.json",
                "--output",
                "run",
            ]
        )
        self.assertEqual("cuda:0", replay.device)
        for preset in ("core10", "cnn30"):
            code, out, err = self.invoke(["run", "-c", preset, "--dry-run"])
            self.assertEqual(0, code, err)
            self.assertEqual("cuda:0", json.loads(out)["config"]["runtime"]["device"])
        config = small_config()
        config["runtime"]["device"] = "cpu"
        for args in (["-c", self.config_file(config)], ["--device", "cpu"]):
            code, out, err = self.invoke(["run", *args, "--dry-run"])
            self.assertEqual(0, code, err)
            self.assertEqual("cpu", json.loads(out)["config"]["runtime"]["device"])

    def test_gpu_request_failure_is_not_retried_on_cpu(self):
        with patch(
            "opennpu_quant.demo.run", side_effect=RuntimeError("CUDA unavailable")
        ) as run:
            self.assert_error(["demo"], "CUDA unavailable")
        run.assert_called_once_with(adaround=False, device="cuda:0")

    def test_gpu_cache_alias_does_not_change_training_window(self):
        expected_window = small_config()["adaround"]["window_bytes"]
        outputs = []
        for flag in ("--gpu-cache-mib", "--gpu-window-mib"):
            code, out, err = self.invoke(
                ["run", "-c", "core10", flag, "32", "--dry-run"]
            )
            self.assertEqual(0, code, err)
            config = json.loads(out)["config"]
            self.assertEqual(32 * 1024**2, config["activation_cache"]["device_bytes"])
            self.assertEqual(expected_window, config["adaround"]["window_bytes"])
            outputs.append(config)
        self.assertEqual(*outputs)

    def test_run_displays_partial_accuracy_and_preserves_json_output(self):
        from opennpu_quant._io import read_json
        from tests.test_result_reliability import fixture

        paths, config = fixture(self.root / "assets")
        config["evaluation"]["limit"] = 1
        args = ["run", "-c", self.config_file(config), "--home", str(paths.home)]
        code, out, err = self.invoke(args)
        self.assertEqual(0, code, err)
        for label in (
            "toy",
            "pot_rtn",
            "Accuracy",
            "Top-5",
            "100.0000%",
            "1 / 2",
            "partial",
            "not aggregated",
            "summary.md",
        ):
            self.assertIn(label, out)
        saved = read_json(paths.runs / config["name"] / "summary.json")
        self.assertTrue(all(v is None for v in saved["mean_recovery"].values()))
        code, out, err = self.invoke([*args, "--json", "--force"])
        self.assertEqual(0, code, err)
        payload = json.loads(out)
        self.assertEqual("cpu", payload["device"])
        current = read_json(paths.runs / config["name"] / "summary.json")
        self.assertEqual(current["rows"], payload["rows"])
        self.assertEqual(2, len(saved["rows"]))
        for before, after in zip(saved["rows"], payload["rows"], strict=True):
            for key in (
                "metrics_json",
                "value_pct",
                "recovery_pct",
                "samples",
                "complete",
            ):
                self.assertEqual(before[key], after[key])
        self.assertEqual(saved["mean_recovery"], payload["mean_recovery"])

    def test_json_summary_sends_runner_progress_to_stderr(self):
        def measured(*args, **kwargs):
            print("Calibration progress")
            return {"rows": [], "mean_recovery": {"fp32": None}}

        with patch("opennpu_quant.runner.run", side_effect=measured):
            code, out, err = self.invoke(["run", "--json"])
        self.assertEqual(0, code, err)
        self.assertEqual([], json.loads(out)["rows"])
        self.assertIn("Calibration progress", err)

    def test_run_help_describes_conditions_devices_paths_and_force(self):
        code, message, _ = self.invoke(["run", "--help"])
        self.assertEqual(0, code)
        for word in (
            "fp32",
            "pot_rtn",
            "pot_adaround",
            "float_rtn",
            "float_adaround",
            "cpu",
            "cuda:N",
            "core10",
            "OPENNPU_QUANT_RUNS",
            "CLI",
            "environment",
            "checkpoint",
            "calibration",
            "reuse",
        ):
            self.assertIn(word, message)

    def test_other_commands_explain_inputs(self):
        for args, words in (
            ([], ["report", "info", "demo", "export", "reproduce"]),
            (["info"], ["diagnostics", "OPENNPU_QUANT_HOME"]),
            (["models"], ["model", "recipe"]),
            (["demo"], ["AdaRound", "cuda:N"]),
            (["export"], ["model", "checkpoint"]),
            (["data"], ["prepare", "status"]),
            (["data", "prepare"], ["root", "folder", "hf", "download"]),
            (["data", "status"], ["manifest"]),
            (["report"], ["run", "folder", "--stdout"]),
            (["results"], ["records.json", "folder", "--stdout"]),
        ):
            with self.subTest(command=args):
                code, message, _ = self.invoke([*args, "--help"])
                self.assertEqual(0, code)
                for word in words:
                    self.assertIn(word, message)

    def test_unused_output_is_hidden_but_still_accepted(self):
        for command in (
            ["export", "resnet18"],
            ["data", "prepare", "imagenet", "--root", str(self.root)],
            ["data", "status"],
        ):
            with self.subTest(command=command):
                self.assertNotIn("--output", self.invoke([*command, "--help"])[1])
                with patch("opennpu_quant.cli.dispatch", return_value=None) as dispatch:
                    self.assertEqual(
                        0, self.invoke([*command, "--output", "unused"])[0]
                    )
                self.assertEqual("unused", dispatch.call_args.args[0].output)

    def test_unknown_model_names_packaged_and_local_alternatives(self):
        self.assert_error(
            ["run", "--home", str(self.root), "--models", "resnet", "--dry-run"],
            "resnet",
            "Packaged models",
            "resnet18",
            str(self.root / "models/resnet/recipe.yaml"),
        )
        self.assertFalse((self.root / "runs").exists())

    def test_unknown_conditions_use_the_selected_config(self):
        config = small_config()
        config["conditions"]["my_rtn"] = config["conditions"].pop("pot_rtn")
        message = self.assert_error(
            [
                "run",
                "-c",
                self.config_file(config),
                "--conditions",
                "typo,also_wrong",
                "--dry-run",
            ],
            "typo",
            "also_wrong",
            "Available",
            "my_rtn",
            "float_rtn",
            "fp32",
        )
        self.assertNotIn("pot_adaround", message)

    def test_local_recipe_custom_condition_and_automatic_fp32(self):
        from opennpu_quant.models.spec import ModelSpec

        config = small_config()
        config["models"] = ["local_toy"]
        config["percentiles"]["local_toy"] = config["percentiles"]["resnet18"]
        config["conditions"]["custom"] = {"scheme": "pot"}
        model_dir = self.root / "models/local_toy"
        model_dir.mkdir(parents=True)
        recipe = copy.deepcopy(ModelSpec.load("resnet18").recipe)
        recipe["name"] = "local_toy"
        recipe["decode"]["profile"] = "current_v1"
        (model_dir / "recipe.yaml").write_text(yaml.safe_dump(recipe))
        code, output, error = self.invoke(
            [
                "run",
                "-c",
                self.config_file(config),
                "--home",
                str(self.root),
                "--conditions",
                "custom",
                "--device",
                "cuda:12",
                "--dry-run",
            ]
        )
        self.assertEqual(0, code, error)
        actual = json.loads(output)["config"]
        self.assertEqual(["fp32", "custom"], list(actual["conditions"]))
        self.assertEqual("cuda:12", actual["runtime"]["device"])
        self.assertFalse((self.root / "runs").exists())
        self.assertFalse((self.root / "cache").exists())

    def test_missing_percentiles_report_the_key_path(self):
        for missing in ("model", "family"):
            config = small_config()
            if missing == "model":
                del config["percentiles"]["resnet18"]
            else:
                del config["percentiles"]["resnet18"]["pot"]
            with self.subTest(missing=missing):
                self.assert_error(
                    ["run", "-c", self.config_file(config), "--dry-run"],
                    "percentiles.resnet18",
                    "pot" if missing == "family" else "percentiles",
                )

    def test_malformed_settings_report_paths_without_tracebacks(self):
        changes = (
            ("name", None),
            ("models", None),
            ("models", [{}]),
            ("conditions", []),
            ("conditions.fp32", None),
            ("conditions.pot_rtn.scheme", []),
            ("runtime", None),
            ("runtime.cuda", []),
            ("calibration", []),
            ("calibration.samples", None),
            ("calibration.histogram_bins", "many"),
            ("schemes", None),
            ("schemes.pot", []),
            ("schemes.pot.method", []),
            ("percentiles", []),
            ("percentiles.resnet18", []),
            ("percentiles.resnet18.pot", "many"),
            ("adaround", []),
            ("adaround.learning_rate", "fast"),
            ("activation_cache", []),
            ("evaluation", None),
            ("output", []),
            ("report", []),
        )
        for path, value in changes:
            config = small_config()
            keys = path.split(".")
            node = config
            for key in keys[:-1]:
                node = node[key]
            node[keys[-1]] = value
            with self.subTest(path=path):
                self.assert_error(
                    ["run", "-c", self.config_file(config), "--dry-run"],
                    keys[0],
                )
        for path in ("runtime.device", "calibration.device", "evaluation.limit"):
            config = small_config()
            section, field = path.split(".")
            del config[section][field]
            with self.subTest(missing=path):
                self.assert_error(
                    ["run", "-c", self.config_file(config), "--dry-run"], path
                )

    def test_invalid_runtime_devices_fail_dry_run(self):
        for device in ("gpu", "cuda", "cuda:-1", "cuda:x", "cuda:1.0"):
            with self.subTest(device=device):
                self.assert_error(
                    ["run", "--models", "resnet18", "--device", device, "--dry-run"],
                    "runtime.device",
                    "cpu",
                    "cuda:N",
                )

    def test_calibration_devices_and_non_string_devices(self):
        for section in ("runtime", "calibration"):
            for device in (None, 1, [], "gpu"):
                config = small_config()
                config[section]["device"] = device
                with self.subTest(section=section, device=device):
                    self.assert_error(
                        ["run", "-c", self.config_file(config), "--dry-run"],
                        section + ".device",
                    )
        for device in ("run", "cpu", "cuda:0", "cuda:12"):
            config = small_config()
            config["calibration"]["device"] = device
            validate(config)

    def test_missing_dataset_suggests_prepare_with_the_actual_data_directory(self):
        from tests.test_result_reliability import fixture

        paths, config = fixture(self.root / "space in home")
        manifest = paths.data / "imagenet/manifest.json"
        manifest.unlink()
        message = self.assert_error(
            ["run", "-c", self.config_file(config), "--home", str(paths.home)],
            "imagenet",
            str(manifest),
            "data prepare",
            "--root",
            "--data-dir",
        )
        command = shlex.split(message.split("Run: ", 1)[1].strip())
        self.assertEqual(str(paths.data), command[command.index("--data-dir") + 1])

    def test_metadata_commands_explain_missing_or_wrong_paths(self):
        file = self.root / "plain.json"
        file.write_text("{}")
        for command, expected in (
            ("report", "run directory"),
            ("results", "records.json"),
        ):
            paths = (
                (self.root / "missing", file)
                if command == "report"
                else (self.root / "missing", self.root)
            )
            for path in paths:
                with self.subTest(command=command, path=str(path)):
                    self.assert_error([command, str(path)], str(path), expected)

    def test_metadata_path_errors_do_not_import_numerical_runtimes(self):
        code = """
import sys, json, contextlib, io
from opennpu_quant.cli import main
codes = []
with contextlib.redirect_stdout(io.StringIO()):
    for command in ('report', 'results'):
        try:
            main([command, sys.argv[1]])
        except SystemExit as error:
            codes.append(error.code)
        else:
            codes.append(None)
print(json.dumps(dict(codes=codes, numerical_modules=sorted(
    {'numpy', 'onnx', 'onnxruntime', 'torch'} & set(sys.modules)
))))
"""
        result = subprocess.run(
            [sys.executable, "-O", "-c", code, str(self.root / "missing")],
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(
            dict(codes=[2, 2], numerical_modules=[]), json.loads(result.stdout)
        )

    def test_metadata_files_need_not_have_prescribed_names(self):
        from opennpu_quant._io import atomic_json
        from opennpu_quant.results import read_records

        records = self.root / "my-measurements.json"
        atomic_json(records, dict(schema_version=1, records=[]))
        self.assertEqual([], read_records(records)["records"])

    def test_feed_factory_errors_explain_replay_and_an_example(self):
        from opennpu_quant import calibrate, quantize, apply_adaround
        from tests.library_fixtures import toy_model, feeds

        model = toy_model()
        quantized = quantize(model, feeds)
        for bad in (list(feeds()), tuple(feeds()), iter(feeds())):
            with self.subTest(kind=type(bad).__name__):
                for call in (
                    lambda: calibrate(model, bad),
                    lambda: apply_adaround(model, quantized, bad),
                ):
                    with self.assertRaises(TypeError) as caught:
                        call()
                    message = str(caught.exception)
                    for word in (
                        "factory",
                        "zero-argument",
                        "iterator",
                        "dict",
                        "lambda: iter(feed_list)",
                    ):
                        self.assertIn(word, message)

    def test_input_cache_errors_identify_the_argument_before_writing(self):
        from opennpu_quant.data.feed_cache import prepare_feed_cache

        options = dict(
            factory=lambda: iter([]),
            verify_source=lambda: None,
            identity={},
            expected_samples=1,
            max_bytes=1024,
        )
        for field, bad in (
            ("factory", []),
            ("verify_source", None),
            ("expected_samples", 0),
            ("expected_samples", True),
            ("max_bytes", -1),
            ("max_bytes", "big"),
        ):
            with (
                self.subTest(field=field, bad=bad),
                self.assertRaisesRegex(ValueError, field),
            ):
                prepare_feed_cache(self.root / "cache", **{**options, field: bad})
        self.assertFalse((self.root / "cache").exists())


if __name__ == "__main__":
    unittest.main()
