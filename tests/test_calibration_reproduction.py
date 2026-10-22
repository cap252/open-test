import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx

from opennpu_quant import (
    CalibrationConfig,
    OrtConfig,
    QuantizationConfig,
    calibrate,
    quantize,
)
from opennpu_quant._io import atomic_json, sha256
from opennpu_quant.reproduction.assets import Resolver
from tests.library_fixtures import toy_model, feeds

spec = importlib.util.spec_from_file_location(
    "calibration_check",
    Path(__file__).resolve().parents[1] / "tools/check_calibration_reproduction.py",
)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class CalibrationReproductionTests(unittest.TestCase):
    def test_bitwise_comparison_rejects_ulp_signed_zero_and_dtype_changes(self):
        original = {"scale": np.array([0.0, 1.0], np.float32)}
        for changed in (
            np.array([-0.0, 1.0], np.float32),
            np.array([0.0, np.nextafter(np.float32(1), np.float32(2))]),
            original["scale"].astype(np.float64),
            original["scale"].reshape(1, 2),
        ):
            self.assertEqual(
                "fail", tool.compare_arrays(original, {"scale": changed})["state"]
            )
        self.assertEqual("pass", tool.compare_arrays(original, original)["state"])

    def test_missing_extra_and_nonfinite_tensors_fail(self):
        original = {"a": np.array([1.0], np.float32)}
        self.assertEqual(["a"], tool.compare_arrays(original, {})["missing"])
        self.assertEqual(["a"], tool.compare_arrays({}, original)["extra"])
        for value in (np.nan, np.inf, -np.inf):
            payload = {"a": np.array([value], np.float32)}
            self.assertEqual("fail", tool.compare_arrays(payload, payload)["state"])

    def test_fresh_statistics_match_but_histogram_count_drift_fails(self):
        cfg = CalibrationConfig.percentile(99.9)
        ort = OrtConfig.cpu()
        first = calibrate(toy_model(), feeds, config=cfg, ort=ort)
        second = calibrate(toy_model(), feeds, config=cfg, ort=ort)
        self.assertEqual(
            "pass",
            tool.compare_arrays(
                tool.statistics_payload(first), tool.statistics_payload(second)
            )["state"],
        )
        second.histograms.rows["x"]["signed"][0] += 1
        comparison = tool.compare_arrays(
            tool.statistics_payload(first), tool.statistics_payload(second)
        )
        self.assertEqual("fail", comparison["state"])
        self.assertEqual(
            ["x/histogram/signed"], [r["name"] for r in comparison["differences"]]
        )

    def test_encodings_compare_actual_qdq_and_detect_changed_statistics(self):
        calibration = CalibrationConfig.percentile(99.9)
        ort = OrtConfig.cpu()
        stats = calibrate(toy_model(), feeds, config=calibration, ort=ort)
        quantized = quantize(
            toy_model(),
            stats,
            config=QuantizationConfig(scope="all", calibration=calibration),
            ort=ort,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model.onnx"
            audit_path = root / "audit.json"
            onnx.save(quantized.model, model_path)
            atomic_json(audit_path, quantized.audit)
            entry = dict(
                condition="sym_pt_float_rtn",
                percentile=99.9,
                audit=dict(id="artifact://test/audit.json", sha256=sha256(audit_path)),
                graphs=dict(
                    network=dict(
                        id="artifact://test/model.onnx", sha256=sha256(model_path)
                    )
                ),
            )
            resolver = Resolver({"test": tmp})
            self.assertEqual(
                "pass",
                tool.compare_encodings([entry], resolver, stats, stats)[0]["state"],
            )
            changed = copy.deepcopy(stats)
            for row in changed.histograms.rows.values():
                row["signed_edges"] *= 2
                row["absolute_edges"] *= 2
                row["low"] *= 2
                row["high"] *= 2
            result = tool.compare_encodings([entry], resolver, stats, changed)[0]
            self.assertEqual("pass", result["reference_to_original_qdq"]["state"])
            self.assertEqual("fail", result["fresh_to_original_qdq"]["state"])


if __name__ == "__main__":
    unittest.main()
