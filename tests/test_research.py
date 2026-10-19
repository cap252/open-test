import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from opennpu_quant.models.spec import model_names
from opennpu_quant.research import (
    FAMILIES,
    Portable,
    validate_selection,
    selected_config,
    export_tables,
    import_snapshot,
)


def cohort():
    rows = []
    for mid in [*model_names(), "quicksrnet_m"]:
        psnr = mid == "quicksrnet_m"
        for family, method in [
            ("", "fp32"),
            *((f, m) for f in FAMILIES for m in ("rtn", "adaround")),
        ]:
            score = (
                30.0
                if psnr
                else 80.0
                if method == "fp32"
                else 76.0
                if method == "rtn"
                else 77.6
            )
            rows.append(
                dict(
                    model_id=mid,
                    family=family,
                    method=method,
                    state="completed",
                    fp32_accuracy=30.0 if psnr else 80.0,
                    accuracy=score,
                    unit="dB" if psnr else "%",
                    delta_db=0 if psnr else None,
                    recovery_percent=None if psnr else score / 80 * 100,
                    drop_pp=None if psnr else 80 - score,
                    percentile=99.9 if method == "rtn" else 99.999,
                    evidence_sha256="a" * 64,
                    condition=family + "_" + method,
                    selection_basis="fixture",
                )
            )
    return rows


class ResearchTests(unittest.TestCase):
    def test_cohort_units_missing_results_and_posthoc_are_distinct(self):
        rows = cohort()
        means = validate_selection(rows)
        self.assertEqual(9, len(means))
        self.assertTrue(all(x["completed_models"] == 38 for x in means))
        for r in means:
            self.assertAlmostEqual(
                95 if r["method"] == "rtn" else 97, r["mean_recovery_percent"]
            )
        item = next(r for r in rows if r["unit"] == "%" and r["method"] == "rtn")
        item["state"] = "failed"
        affected = next(
            r
            for r in validate_selection(rows)
            if r["family"] == item["family"] and r["method"] == "rtn"
        )
        self.assertEqual(37, affected["completed_models"])
        self.assertIsNone(affected["mean_recovery_percent"])
        self.assertAlmostEqual(95, affected["observed_partial_mean_recovery_percent"])
        rows[-1]["recovery_percent"] = 100
        with self.assertRaisesRegex(ValueError, "PSNR"):
            validate_selection(rows)

    def test_selected_percentiles_do_not_replace_adaround_with_rtn(self):
        rows = cohort()
        before = copy.deepcopy(rows)
        cfg, evidence = selected_config(rows)
        self.assertEqual(38, len(cfg["models"]))
        self.assertNotIn("quicksrnet_m", cfg["models"])
        self.assertEqual(38 * 6, len(evidence))
        for mapping in cfg["percentiles"].values():
            for family in FAMILIES:
                self.assertEqual(99.9, mapping[family + "_rtn"])
                self.assertEqual(99.999, mapping[family + "_adaround"])
        self.assertEqual(before, rows)

    def test_json_configuration_preserves_scientific_notation_tolerances(self):
        from opennpu_quant.configuration import load_config

        cfg, _ = selected_config(cohort())
        cfg["adaround"].update(atol=1e-5, rtol=1e-4)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "selected.json"
            path.write_text(json.dumps(cfg))
            self.assertEqual(cfg, load_config(path))

    def test_duplicate_or_wrong_recovery_is_rejected(self):
        rows = cohort()
        rows[-1] = copy.deepcopy(rows[-2])
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_selection(rows)
        rows = cohort()
        rows[1]["recovery_percent"] = 100
        with self.assertRaisesRegex(ValueError, "recovery"):
            validate_selection(rows)

    def test_json_csv_tsv_keep_units_nulls_and_numbers(self):
        tables = {
            "Scores": [
                dict(model="분류", metric=0.123456789, unit="%", missing=None),
                dict(model="SR", metric=32.123456, unit="dB", missing=None),
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            counts = export_tables(tables, root)
            self.assertEqual({"Scores": 2}, counts)
            self.assertEqual(
                tables, json.loads((root / "records.json").read_text())["tables"]
            )
            for suffix, sep in [("csv", ","), ("tsv", "\t")]:
                with (root / ("Scores." + suffix)).open() as stream:
                    rows = list(csv.DictReader(stream, delimiter=sep))
                self.assertEqual(["%", "dB"], [r["unit"] for r in rows])
                self.assertEqual(["", ""], [r["missing"] for r in rows])
                self.assertEqual(
                    [r["metric"] for r in tables["Scores"]],
                    [float(r["metric"]) for r in rows],
                )

    def test_portable_paths_keep_original_unchanged(self):
        original = dict(
            evidence="/mnt/data/midap_quantization/runs/r/a.json",
            nested=json.dumps({"source": "/mnt/ssd/midap_cnn_quant/src/a.py"}),
            url="https://example.com/a",
        )
        before = copy.deepcopy(original)
        converter = Portable()
        result = converter(original)
        self.assertNotIn("/mnt/", json.dumps(result))
        self.assertIn("artifact://research-runs/r/a.json", result["evidence"])
        self.assertEqual(original, before)
        self.assertEqual(original["url"], result["url"])
        self.assertEqual(2, len(converter.bindings))

    def test_failed_import_never_publishes_partial_snapshot_or_bindings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binding = root / "bindings.json"
            binding.write_text('{"existing":true}')

            def partial(source, output, **kwargs):
                output.mkdir()
                (output / "Selected.csv").write_text("partial")
                raise ValueError("Changed payload")

            with patch("opennpu_quant.research._import_snapshot", side_effect=partial):
                with self.assertRaisesRegex(ValueError, "Changed payload"):
                    import_snapshot(
                        root / "source", root / "published", bindings_output=binding
                    )
            self.assertFalse((root / "published").exists())
            self.assertEqual({"existing": True}, json.loads(binding.read_text()))
            self.assertEqual([binding], list(root.iterdir()))
