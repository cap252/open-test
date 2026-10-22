"""Generated presets retain complete settings and the existing policy conversions."""

import ast
from contextlib import redirect_stdout
import hashlib
from importlib.resources import files
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from opennpu_quant import CalibrationConfig, QuantizationConfig, Scheme
from opennpu_quant.configuration import PRESETS, load_config, scheme_for
from tools.build_presets import main, presets, render, synchronize


def ordered_digest(config):
    return hashlib.sha256(
        json.dumps(
            config, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


class PresetGenerationTests(unittest.TestCase):
    def test_existing_settings_order_and_standalone_files(self):
        expected = json.loads(
            (Path(__file__).parent / "fixtures/preset_contracts.json").read_text()
        )
        generated = presets()
        self.assertEqual(set(PRESETS), set(generated))
        self.assertEqual(set(PRESETS), set(expected))
        packaged = files("opennpu_quant.resources.experiments")
        for name in PRESETS:
            with self.subTest(preset=name):
                config = load_config(name)
                self.assertEqual(
                    expected[name]["ordered_config_sha256"], ordered_digest(config)
                )
                self.assertEqual(config, generated[name])
                self.assertEqual(config, load_config(packaged / (name + ".yaml")))
                self.assertEqual(expected[name]["models"], len(config["models"]))
                self.assertEqual(
                    expected[name]["conditions"], list(config["conditions"])
                )
                self.assertEqual(
                    ordered_digest(config), ordered_digest(generated[name])
                )
                self.assertEqual(
                    render()[name + ".yaml"], (packaged / (name + ".yaml")).read_bytes()
                )
        self.assertEqual(("core10", "cnn30"), PRESETS)
        self.assertEqual(
            {"core10.yaml", "cnn30.yaml"},
            {p.name for p in packaged.iterdir() if p.name.endswith(".yaml")},
        )
        for removed in ("quick", "extended", "cnn38", "cnn38_smoke"):
            with self.assertRaisesRegex(ValueError, "not found"):
                load_config(removed)

    def test_each_generated_config_owns_its_nested_values(self):
        original = presets()
        changed = presets()
        changed["cnn30"]["adaround"]["steps"] = 1
        changed["cnn30"]["percentiles"]["resnet18"]["pot"] = 90.0
        changed["core10"]["schemes"]["pot"]["activation_scale"] = "float"
        self.assertEqual(original["core10"]["adaround"], changed["core10"]["adaround"])
        self.assertEqual(
            original["core10"]["percentiles"], changed["core10"]["percentiles"]
        )
        self.assertEqual(original["cnn30"]["schemes"], changed["cnn30"]["schemes"])
        self.assertEqual(99.99, changed["cnn30"]["percentiles"]["resnet50"]["pot"])
        self.assertEqual(original, presets())

    def test_check_is_read_only_and_write_is_repeatable(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "presets"
            self.assertEqual(2, len(synchronize(directory)))
            self.assertFalse(directory.exists())
            self.assertEqual(2, len(synchronize(directory, write=True)))
            before = {
                p.name: (p.read_bytes(), p.stat().st_mtime_ns)
                for p in directory.iterdir()
            }
            self.assertEqual([], synchronize(directory))
            self.assertEqual([], synchronize(directory, write=True))
            self.assertEqual(
                before,
                {
                    p.name: (p.read_bytes(), p.stat().st_mtime_ns)
                    for p in directory.iterdir()
                },
            )
            stale = directory / "core10.yaml"
            stale.write_bytes(stale.read_bytes().replace(b"limit: null", b"limit: 3"))
            changed = stale.read_bytes()
            self.assertEqual(["core10.yaml"], synchronize(directory))
            self.assertEqual(changed, stale.read_bytes())
            (directory / "cnn30.yaml").unlink()
            self.assertEqual(["core10.yaml", "cnn30.yaml"], synchronize(directory))
            self.assertEqual(
                ["core10.yaml", "cnn30.yaml"], synchronize(directory, write=True)
            )
            self.assertEqual([], synchronize(directory))
            self.assertEqual(
                {name: value[0] for name, value in before.items()}, render()
            )

    def test_cli_default_and_explicit_check_fail_on_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "presets"
            with (
                patch(
                    "tools.build_presets.synchronize",
                    side_effect=lambda **kwargs: synchronize(directory, **kwargs),
                ),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(1, main([]))
                self.assertEqual(1, main(["--check"]))
                self.assertFalse(directory.exists())
                self.assertEqual(0, main(["--write"]))
                self.assertEqual(0, main([]))
                self.assertEqual(0, main(["--check"]))

    def test_yaml_scheme_and_explicit_config_agree(self):
        for name, config in presets().items():
            for model in config["models"]:
                for family, row in config["schemes"].items():
                    with self.subTest(preset=name, model=model, family=family):
                        percentile = config["percentiles"][model][family]
                        explicit = QuantizationConfig(
                            scope=row["scope"],
                            activation_symmetric=row["activation_symmetric"],
                            weight_granularity=row["weight_granularity"],
                            calibration=CalibrationConfig.percentile(
                                percentile,
                                histogram_bins=config["calibration"]["histogram_bins"],
                            ),
                            activation_scale_policy=row["activation_scale"],
                            weight_scale_policy=row["weight_scale"],
                        )
                        scheme = Scheme(
                            **{
                                **row,
                                "percentile": percentile,
                                "histogram_bins": config["calibration"][
                                    "histogram_bins"
                                ],
                            }
                        )
                        self.assertEqual(explicit, scheme.config())
                        self.assertEqual(
                            explicit.scheme(),
                            scheme_for(config, model, family).config().scheme(),
                        )
                        self.assertEqual(explicit.name, scheme.name)

    def test_api_policy_defaults_and_other_methods_are_preserved(self):
        compact, explicit = Scheme().config(), QuantizationConfig()
        self.assertEqual(
            ("all", False, "per_channel", "Percentile", 99.99),
            (
                compact.scope,
                compact.activation_symmetric,
                compact.weight_granularity,
                compact.calibration.method,
                compact.calibration.percentile_value,
            ),
        )
        self.assertEqual(
            ("basic", True, "per_tensor", "MinMax", None),
            (
                explicit.scope,
                explicit.activation_symmetric,
                explicit.weight_granularity,
                explicit.calibration.method,
                explicit.calibration.percentile_value,
            ),
        )
        for method, calibration in (
            ("minmax", CalibrationConfig()),
            ("entropy", CalibrationConfig.entropy()),
        ):
            compact = Scheme(
                scope="basic",
                activation_symmetric=True,
                weight_granularity="per_tensor",
                method=method,
                percentile=None,
            )
            explicit = QuantizationConfig(calibration=calibration)
            self.assertEqual(explicit, compact.config())
            self.assertEqual(explicit.scheme(), compact.config().scheme())

    def test_runtime_has_no_maintainer_imports(self):
        root = Path(str(files("opennpu_quant")))
        self.assertFalse((root / "research.py").exists())
        for path in root.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    continue
                for module in modules:
                    with self.subTest(file=path, module=module):
                        self.assertNotIn(
                            module.split(".")[0],
                            {"tools", "research_records", "build_presets"},
                        )
