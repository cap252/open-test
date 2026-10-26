"""Model-set defaults select new work without narrowing existing replay packages."""

import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from opennpu_quant._io import atomic_json, object_hash, read_json
from opennpu_quant.cli import build_parser, main
from opennpu_quant.configuration import load_config
from opennpu_quant.reproduction.assets import catalog, unseal
from opennpu_quant.reproduction.prepare import prepare
from opennpu_quant.reproduction.settings import configs
from opennpu_quant.reproduction.execution import run
from opennpu_quant.reproduction.compare import compare
from opennpu_quant.reproduction.report import report
from tests.test_gpu_reproduction import fixture


class ReproductionModelSetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_default_preparation_preserves_exact_selected_entries(self):
        original = catalog()
        before = object_hash(original)
        result = prepare(bindings={}, output=self.root / "prepared", limit="full")
        saved = unseal(self.root / "prepared/manifest.json")
        names = set(load_config("cnn30")["models"])
        expected = [e for e in original["entries"] if e["model"] in names]
        self.assertEqual(30, len(names))
        self.assertEqual(210, result["conditions"])
        self.assertEqual({"pending": 210}, result["counts"])
        self.assertEqual(expected, [r["entry"] for r in saved["conditions"]])
        self.assertEqual(before, saved["selection_identity"])
        self.assertEqual(before, object_hash(catalog()))
        self.assertNotIn("posenet_101", names)
        self.assertEqual(
            {
                "yolov3_full",
                "yolov4_full",
                "yolov7_full",
                "yolov9_s",
                "efficientnet_lite0",
                "yolov9_t",
                "posenet_101",
                "deeplabv3plus_r50_voc",
            },
            {e["model"] for e in original["entries"]} - names,
        )
        self.assertEqual(10, len(load_config("core10")["models"]))
        self.assertEqual(5, len(load_config("core10")["conditions"]))

    def test_default_and_explicit_cnn30_require_full_lfw_before_work(self):
        for group in (None, "cnn30"):
            with self.subTest(group=group):
                output = self.root / str(group)
                with (
                    patch("opennpu_quant.reproduction.assets.Resolver") as resolver,
                    self.assertRaisesRegex(ValueError, "LFW"),
                ):
                    prepare(bindings={}, output=output, model_set=group)
                resolver.assert_not_called()
                self.assertFalse(output.exists())

    def test_sets_agree_between_prepare_and_configs(self):
        for group, model_count, condition_count in (
            ("core10", 10, 70),
            ("cnn30", 30, 210),
            ("all", 38, 266),
        ):
            with self.subTest(group=group):
                p = self.root / (group + "-prepared")
                c = self.root / (group + "-configs")
                prepared = prepare(bindings={}, output=p, model_set=group, limit="full")
                generated = configs(output=c, model_set=group)
                rows = unseal(p / "manifest.json")["conditions"]
                settings = unseal(c / "manifest.json")["conditions"]
                self.assertEqual(condition_count, prepared["conditions"])
                self.assertEqual(condition_count, generated["conditions"])
                self.assertEqual(model_count, len({r["entry"]["model"] for r in rows}))
                self.assertEqual(
                    [r["entry"]["id"] for r in rows], [r["id"] for r in settings]
                )

    def test_default_configs_match_all_condition_files_byte_for_byte(self):
        default, all_models = self.root / "default", self.root / "all"
        self.assertEqual(210, configs(output=default)["conditions"])
        self.assertEqual(266, configs(output=all_models, model_set="all")["conditions"])
        rows = unseal(default / "manifest.json")["conditions"]
        for row in rows:
            name = row["config"]["path"]
            self.assertEqual(
                (all_models / name).read_bytes(), (default / name).read_bytes()
            )

    def test_explicit_models_and_condition_filters(self):
        for command in (prepare, configs):
            with self.subTest(command=command.__name__):
                output = self.root / command.__name__
                extra = {"bindings": {}} if command is prepare else {}
                result = command(
                    output=output,
                    models=["yolov9_s", "yolov9_t", "resnet18"],
                    conditions=["fp32", "asym_pc_pot_rtn"],
                    **extra,
                )
                self.assertEqual(6, result["conditions"])
                rows = unseal(output / "manifest.json")["conditions"]
                ids = {
                    r["entry"]["id"] if command is prepare else r["id"] for r in rows
                }
                self.assertEqual(
                    {
                        f"{m}__{c}"
                        for m in ("yolov9_s", "yolov9_t", "resnet18")
                        for c in ("fp32", "asym_pc_pot_rtn")
                    },
                    ids,
                )
        result = prepare(
            bindings={}, output=self.root / "fp32", conditions=["fp32"], limit="full"
        )
        self.assertEqual(30, result["conditions"])

    def test_custom_manifest_defaults_to_its_own_models(self):
        source = catalog()
        source["entries"] = [e for e in source["entries"] if e["model"] == "yolov9_s"]
        selection = self.root / "custom.json"
        atomic_json(selection, source)
        for command in (prepare, configs):
            extra = {"bindings": {}} if command is prepare else {}
            for group in (None, "all"):
                with self.subTest(command=command.__name__, group=group):
                    output = self.root / f"{command.__name__}-{group}"
                    self.assertEqual(
                        7,
                        command(
                            output=output, selection=selection, model_set=group, **extra
                        )["conditions"],
                    )
            for group in ("core10", "cnn30"):
                output = self.root / f"{command.__name__}-{group}"
                with self.assertRaisesRegex(ValueError, "Unknown model"):
                    command(
                        output=output, selection=selection, model_set=group, **extra
                    )
                self.assertFalse(output.exists())

    def test_invalid_api_selections_fail_before_output(self):
        for command in (prepare, configs):
            extra = {"bindings": {}} if command is prepare else {}
            for options, error in (
                ({"model_set": "core10", "models": ["resnet18"]}, "not both"),
                ({"model_set": "extended"}, "Unknown model set"),
                ({"model_set": "cnn30", "models": ["resnet18"]}, "not both"),
                ({"model_set": "unknown"}, "Unknown model set"),
                ({"models": []}, "at least one"),
                ({"models": ["missing"]}, "Unknown model"),
                ({"conditions": ["missing"]}, "Unknown condition"),
            ):
                with self.subTest(command=command.__name__, options=options):
                    output = self.root / "invalid"
                    with self.assertRaisesRegex(ValueError, error):
                        command(output=output, **options, **extra)
                    self.assertFalse(output.exists())

    def test_cli_routes_sets_and_rejects_conflicting_options(self):
        atomic_json(self.root / "bindings.json", {})
        for command in ("prepare", "configs"):
            base = ["reproduce", command]
            if command == "prepare":
                base += [
                    "--bindings",
                    str(self.root / "bindings.json"),
                    "--limit",
                    "full",
                ]
            for index, (options, count) in enumerate(
                (
                    ([], 210),
                    (["--model-set", "cnn30"], 210),
                    (["--model-set", "core10"], 70),
                    (["--model-set", "all", "--conditions", "fp32"], 38),
                    (["--models", "yolov9_s"], 7),
                )
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    main(
                        [
                            *base,
                            *options,
                            "--output",
                            str(self.root / f"{command}-{index}"),
                        ]
                    )
                saved = unseal(self.root / f"{command}-{index}/manifest.json")
                self.assertEqual(count, len(saved["conditions"]))
            for options in (
                ["--model-set", "bad"],
                ["--model-set", "extended"],
                ["--model-set", "core10", "--models", "resnet18"],
                ["--model-set", "cnn30", "--models", "resnet18"],
            ):
                with (
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as error,
                ):
                    build_parser().parse_args([*base, *options, "--output", "unused"])
                self.assertEqual(2, error.exception.code)

    def test_existing_package_scope_survives_run_resume_compare_and_report(self):
        fixture(self.root)
        selection = read_json(self.root / "selection.json")
        original = selection["entries"][0]
        original["measurement"] = {"samples": 3}
        additional = copy.deepcopy(original)
        additional.update(id="yolov9_s__fp32", model="yolov9_s")
        additional["recipe"].update(name="yolov9_s", title="YOLOv9-S")
        additional["recipe"]["decode"]["profile"] = "current_v1"
        selection["entries"].append(additional)
        former_extended = copy.deepcopy(additional)
        former_extended.update(id="yolov9_t__fp32", model="yolov9_t")
        former_extended["recipe"].update(name="yolov9_t", title="YOLOv9-T")
        selection["entries"].append(former_extended)
        atomic_json(self.root / "selection.json", selection)
        prepare(
            bindings=self.root / "bindings.json",
            selection=self.root / "selection.json",
            output=self.root / "prepared",
            limit="full",
        )
        args = dict(
            package=self.root / "prepared",
            bindings=self.root / "bindings.json",
            output=self.root / "run",
            device="cpu",
        )
        self.assertEqual(3, run(**args)["completed"])
        extra_result = self.root / "run/yolov9_s__fp32/result.json"
        stamp = extra_result.stat().st_mtime_ns
        # Neither the current default nor a future default may reselect prepared work.
        with patch(
            "opennpu_quant.reproduction.assets.select_catalog",
            side_effect=AssertionError("Reselected existing package"),
        ):
            self.assertEqual(3, run(**args)["completed"])
            self.assertEqual(stamp, extra_result.stat().st_mtime_ns)
            compared = compare(
                actual=self.root / "run",
                expected=self.root / "run",
                output=self.root / "comparison",
            )
            self.assertEqual({"pass": 3}, compared["counts"])
            table = report(
                run=self.root / "run",
                package=self.root / "prepared",
                output=self.root / "tables",
                table_profile="opennpu_raw",
            )
            self.assertEqual(3, table["full"])
            self.assertEqual(0, table["excluded"])
        self.assertIn(
            "YOLOv9-S",
            (self.root / "tables/raw_data.csv").read_text(encoding="utf-8-sig"),
        )
