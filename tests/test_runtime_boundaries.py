"""Numerical environment identity, FP32-only input needs and GPU evidence scope."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from opennpu_quant._io import atomic_json, read_json
from opennpu_quant.ort.cpu_environment import cpu_environment
from opennpu_quant.ort.config import OrtConfig
from opennpu_quant.ort.session import environment
from opennpu_quant.ort.profile import summarize_profile, assert_cuda_observed
from opennpu_quant.reproduction.environment import compare
from opennpu_quant.runner import run
from tests.test_result_reliability import fixture


class RuntimeBoundaryTests(unittest.TestCase):
    def test_effective_numpy_features_and_flags_change_environment(self):
        before = environment(OrtConfig.cpu())
        enabled = before["cpu_numerics"]["features"] or {}
        candidates = [x for x in ("AVX2", "FMA3", "AVX512F") if enabled.get(x)]
        if not candidates:
            self.assertIsInstance(before["cpu_numerics"]["implementation"], dict)
            return
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json; from opennpu_quant.ort.session import environment; "
                "from opennpu_quant.ort.config import OrtConfig; "
                "print(json.dumps(environment(OrtConfig.cpu())))",
            ],
            env=dict(os.environ, NPY_DISABLE_CPU_FEATURES=",".join(candidates)),
            capture_output=True,
            text=True,
            check=True,
        )
        after = json.loads(child.stdout)
        self.assertNotEqual(before, after)
        for name in candidates:
            self.assertFalse(after["cpu_numerics"]["features"][name])

    def test_fp32_only_without_calibration_and_environment_reuse_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            paths, config = fixture(Path(temp))
            config["conditions"] = {"fp32": {}}
            manifest = paths.data / "imagenet/manifest.json"
            value = read_json(manifest)
            value["calibration"] = []
            atomic_json(manifest, value)
            with patch(
                "opennpu_quant.runner.calibrate",
                side_effect=AssertionError("calibration called"),
            ):
                run(config, paths)
                run(config, paths)
            status = read_json(paths.runs / config["name"] / "status.json")
            self.assertEqual(
                "reused", status["models"]["toy"]["conditions"]["fp32"]["state"]
            )
            with patch.dict(os.environ, {"NPY_DISABLE_CPU_FEATURES": "AVX2"}):
                with self.assertRaisesRegex(ValueError, "execution identity changed"):
                    run(config, paths)
            config["name"] = "mixed"
            config["conditions"]["pot_rtn"] = {
                "scheme": "asym_pc_pot",
                "adaround": False,
            }
            # Keep a valid quantization policy from the preset.
            from opennpu_quant.configuration import load_config

            config["conditions"]["pot_rtn"] = load_config()["conditions"]["pot_rtn"]
            with self.assertRaisesRegex(ValueError, "Calibration sample count"):
                run(config, paths)

    def test_environment_missing_is_pending_and_driver_changes_fail(self):
        base = dict(device="cpu", cpu_numerics=cpu_environment())
        self.assertEqual("pass", compare(base, base)["state"])
        self.assertEqual(
            "pending", compare({"device": "cpu"}, {"device": "cpu"})["state"]
        )
        other = copy.deepcopy(base)
        other["cpu_numerics"]["features"] = {"changed": True}
        self.assertEqual("fail", compare(base, other)["state"])
        base.update(
            device="cuda:0",
            kernel_driver="driver one",
            driver_libraries={"libraries": [dict(name="libcuda.so.1", sha256="a")]},
            loaded_libraries={
                "libraries": [
                    dict(name=n, sha256="x")
                    for n in (
                        "libcudart.so",
                        "libcublas.so",
                        "libcudnn.so",
                        "libonnxruntime_providers_cuda.so",
                    )
                ]
            },
        )
        self.assertEqual("pass", compare(base, base)["state"])
        for field in ("kernel_driver", "driver_libraries"):
            other = copy.deepcopy(base)
            if field == "kernel_driver":
                other[field] = "driver two"
            else:
                other[field]["libraries"][0]["sha256"] = "b"
            self.assertEqual("fail", compare(base, other)["state"])
            other.pop(field)
            self.assertEqual("pending", compare(base, other)["state"])

    def test_cuda_observation_does_not_attest_full_placement(self):
        events = [
            dict(cat="Node", args=dict(provider=p, op_name="Conv"))
            for p in ["CUDAExecutionProvider"] + ["CPUExecutionProvider"] * 99
        ]
        result = summarize_profile(events)
        assert_cuda_observed(["CUDAExecutionProvider"], result)
        self.assertEqual(
            "cpu_compute_observed", result["sampled_main_compute_placement"]
        )
        self.assertEqual(99, result["cpu_main_compute"][0]["calls"])
        self.assertEqual("not_verified", result["full_network_placement"])
