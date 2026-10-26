"""Failed repeatability measurements stay inspectable without entering recovery."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from opennpu_quant._io import read_json, sha256
from opennpu_quant.reproduction.assets import file_record, seal, unseal
from opennpu_quant.reproduction.report import report
from opennpu_quant.results import export_records, recovery_percent, RAW_FIELDS
from tests.test_raw_report import replay


class ReplayDiagnosticTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.package, self.run, _, self.saved = replay(self.root, full=True)
        self.output = self.root / "tables"

    def failed_result(self, index=1):
        state = self.saved["conditions"][index]
        path = self.run / state["result"]["path"]
        result = unseal(path)
        # Old runs did not record an explicit failure_reason.
        result.update(state="fail", repeat_bitwise=False)
        seal(path, result)
        state.update(
            state="fail", repeat_bitwise=False, result=file_record(path, self.run)
        )
        seal(self.run / "run.json", self.saved)
        return state, path, result

    def export(self, **kwargs):
        return report(
            run=self.run,
            package=self.package,
            output=self.output,
            table_profile="opennpu_raw",
            **kwargs,
        )

    def csv_rows(self, name):
        with (self.output / name).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_failed_metrics_are_diagnostic_only_and_survive_portable_export(self):
        self.failed_result()
        counts = self.export()
        self.assertEqual(
            (1, 0, 1), tuple(counts[k] for k in ("full", "partial", "excluded"))
        )
        rows = self.csv_rows("raw_data.csv")
        self.assertEqual(1, len(rows))
        self.assertEqual("FP32", rows[0]["scheme"])
        diagnostic = self.csv_rows("diagnostics.csv")
        self.assertEqual(1, len(diagnostic))
        self.assertEqual("0.78", diagnostic[0]["value"])
        self.assertEqual("fraction", diagnostic[0]["unit"])
        self.assertEqual("fail", diagnostic[0]["state"])
        self.assertEqual("repeat_bitwise_mismatch", diagnostic[0]["failure_reason"])
        self.assertNotIn("recovery_percent", diagnostic[0])
        records = read_json(self.output / "records.json")["records"]
        failed = next(r for r in records if r["state"] == "fail")
        self.assertEqual({}, failed["metrics"])
        self.assertFalse(failed["complete"])
        self.assertFalse(failed["full_evaluation"])
        self.assertTrue(failed["evidence"]["result_hash_verified"])
        self.assertEqual(
            0.78, failed["diagnostic_measurement"]["metrics"]["top1"]["value"]
        )
        summary = read_json(self.output / "summary.json")
        self.assertIsNone(summary["mean_recovery"]["asym_pc_pot_rtn"])
        metadata = read_json(self.output / "raw_data.metadata.json")
        excluded = next(r for r in metadata["records"] if r["exported_to"] is None)
        self.assertEqual("repeat_bitwise_mismatch", excluded["reason"])
        copy = self.root / "portable"
        export_records(records, copy, table_profile="opennpu_raw")
        self.assertEqual(
            (self.output / "diagnostics.csv").read_bytes(),
            (copy / "diagnostics.csv").read_bytes(),
        )
        self.assertEqual(
            (self.output / "raw_data.csv").read_bytes(),
            (copy / "raw_data.csv").read_bytes(),
        )

    def test_failed_baseline_does_not_supply_a_recovery_denominator(self):
        self.failed_result(index=0)
        self.export(formats="csv")
        rows = self.csv_rows("raw_data.csv")
        self.assertEqual(1, len(rows))
        self.assertEqual("78.0", rows[0]["int8_accuracy"])
        self.assertEqual("", rows[0]["fp32_accuracy"])
        self.assertEqual("", rows[0]["recovery_percent"])
        self.assertEqual("0.8", self.csv_rows("diagnostics.csv")[0]["value"])
        self.assertFalse((self.output / "records.json").exists())

    def test_json_only_and_filtered_csv_do_not_keep_stale_diagnostic_rows(self):
        self.failed_result()
        self.export(formats="json")
        self.assertFalse((self.output / "diagnostics.csv").exists())
        self.assertIn(
            "diagnostic_measurement",
            read_json(self.output / "records.json")["records"][1],
        )
        self.export(formats="csv")
        self.assertEqual(1, len(self.csv_rows("diagnostics.csv")))
        self.export(formats="csv", conditions=["fp32"])
        self.assertEqual([], self.csv_rows("diagnostics.csv"))

    def test_failure_before_final_result_has_no_diagnostic_measurement(self):
        self.saved["conditions"][1] = dict(
            id=self.saved["conditions"][1]["id"],
            state="fail",
            error="inference failed",
        )
        seal(self.run / "run.json", self.saved)
        self.export()
        self.assertFalse((self.output / "diagnostics.csv").exists())
        failed = read_json(self.output / "records.json")["records"][1]
        self.assertNotIn("diagnostic_measurement", failed)
        self.assertEqual("inference failed", failed["failure_reason"])

    def test_original_accuracy_failure_does_not_hide_a_completed_measurement(self):
        state = self.saved["conditions"][1]
        path = self.run / state["result"]["path"]
        result = unseal(path)
        result["original_accuracy"] = dict(state="fail", metrics_state="fail")
        seal(path, result)
        state["result"] = file_record(path, self.run)
        seal(self.run / "run.json", self.saved)
        self.assertEqual(2, self.export()["full"])
        self.assertFalse((self.output / "diagnostics.csv").exists())
        self.assertEqual("78.0", self.csv_rows("raw_data.csv")[1]["int8_accuracy"])

    def test_tampered_failed_result_is_rejected_before_output(self):
        _, path, _ = self.failed_result()
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_resealed_invalid_failed_result_is_rejected_before_output(self):
        state, path, original = self.failed_result()
        for field, value in (
            ("input_identity", "changed"),
            ("prepared_identity", "changed"),
            ("sample_count", 1),
            ("repeat_bitwise", True),
            ("state", "completed"),
        ):
            with self.subTest(field=field):
                seal(path, dict(original, **{field: value}))
                state["result"] = file_record(path, self.run)
                seal(self.run / "run.json", self.saved)
                with self.assertRaises(ValueError):
                    self.export()
                self.assertFalse(self.output.exists())


class DocumentationExampleTests(unittest.TestCase):
    def test_csv_examples_match_current_recovery_and_recorded_hashes(self):
        folder = Path(__file__).resolve().parents[1] / "docs/examples/raw-data"
        provenance = read_json(folder / "provenance.json")
        self.assertFalse(list(folder.glob("*.tsv")))
        for item in provenance.values():
            for filename, expected in item["files"].items():
                with self.subTest(file=filename):
                    path = folder / filename
                    self.assertEqual(".csv", path.suffix)
                    self.assertEqual(expected, sha256(path))
                    with path.open(encoding="utf-8-sig", newline="") as stream:
                        reader = csv.DictReader(stream)
                        self.assertEqual(RAW_FIELDS, reader.fieldnames)
                        rows = list(reader)
                    baseline = next(r for r in rows if r["scheme"] == "FP32")
                    base = json.loads(baseline["metrics_json"])[baseline["metric"]]
                    for row in rows:
                        metric = json.loads(row["metrics_json"])[row["metric"]]
                        expected = recovery_percent(
                            metric, base, is_baseline=row is baseline
                        )
                        self.assertEqual(repr(expected), row["recovery_percent"])
