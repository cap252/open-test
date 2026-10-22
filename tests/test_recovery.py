"""Recovery arithmetic is shared without changing measurement or pairing rules."""

import copy
import csv
import json
from pathlib import Path
import struct
import tempfile
import unittest

from opennpu_quant.reproduction.assets import file_record, seal, unseal
from opennpu_quant.reproduction.execution import reevaluation_metrics
from opennpu_quant.reproduction.report import report
from opennpu_quant.results import recovery_percent, tables
from tests.test_raw_report import replay
from tests.test_results import record


def metric(value, unit="fraction"):
    return dict(value=value, unit=unit)


class RecoveryTests(unittest.TestCase):
    def assertBits(self, expected, actual):
        self.assertEqual(struct.pack(">d", expected), struct.pack(">d", actual))

    def test_operation_order_and_fp32_self_comparison(self):
        base = 0.3750636660794323
        value = 0.3546653217459048
        expected = 94.56136486193054
        self.assertNotEqual(expected, 100 * (100 * value) / (100 * base))
        self.assertBits(expected, recovery_percent(metric(value), metric(base)))
        self.assertBits(
            100.0, recovery_percent(metric(base), metric(base), is_baseline=True)
        )
        # Equal INT8 accuracy is still a measured ratio, not an FP32 identity.
        self.assertBits(99.99999999999999, recovery_percent(metric(base), metric(base)))
        rows, _ = tables([record(value=base), record("q", value)])
        self.assertBits(100.0, rows[0]["recovery_pct"])
        self.assertBits(expected, rows[1]["recovery_pct"])
        self.assertEqual(value * 100, rows[1]["value_pct"])
        self.assertEqual(value, json.loads(rows[1]["metrics_json"])["top1"]["value"])

    def test_same_and_mixed_units_preserve_inputs(self):
        base, value = 0.3750636660794323, 0.3546653217459048
        cases = [
            (metric(value), metric(base), 100 * value / base),
            (
                metric(value * 100, "percent"),
                metric(base * 100, "percent"),
                100 * (value * 100) / (base * 100),
            ),
            (
                metric(value * 100, "percent"),
                metric(base),
                100 * (value * 100 / 100) / base,
            ),
            (
                metric(value),
                metric(base * 100, "percent"),
                100 * value / (base * 100 / 100),
            ),
        ]
        for numerator, denominator, expected in cases:
            with self.subTest(numerator=numerator, denominator=denominator):
                before = copy.deepcopy((numerator, denominator))
                self.assertBits(expected, recovery_percent(numerator, denominator))
                self.assertEqual(before, (numerator, denominator))

    def test_missing_unsupported_or_zero_baseline_is_blank(self):
        for numerator, denominator in (
            (None, metric(0.8)),
            (metric(None), metric(0.8)),
            (metric(0.8), None),
            (metric(0.8), metric(None)),
            (metric(0.8), metric(0)),
            (metric(0.8), metric(-1)),
            (metric(30, "dB"), metric(31, "dB")),
            (metric(0.8, "unknown"), metric(0.9)),
            (metric(0.8), metric(0.9, "unknown")),
        ):
            for baseline in (False, True):
                with self.subTest(numerator=numerator, denominator=denominator):
                    self.assertIsNone(
                        recovery_percent(numerator, denominator, is_baseline=baseline)
                    )
        rows, _ = tables([record(value=0), record("q", 0.5)])
        self.assertTrue(all(r["recovery_pct"] is None for r in rows))

    def test_invalid_numbers_are_rejected(self):
        for invalid in (float("nan"), float("inf"), -float("inf"), True, "0.8"):
            for numerator, denominator in (
                (metric(invalid), metric(0.8)),
                (metric(0.8), metric(invalid)),
            ):
                with self.subTest(invalid=invalid):
                    with self.assertRaisesRegex(ValueError, "Nonfinite or nonnumeric"):
                        recovery_percent(numerator, denominator)

    def test_failed_or_unmatched_fp32_is_not_filled(self):
        for changes in (
            dict(evaluation_identity="different"),
            dict(state="failed"),
            dict(evidence={}),
        ):
            rows, _ = tables([record(**changes), record("q", 0.78)])
            self.assertIsNone(rows[1]["recovery_pct"])
        rows, _ = tables([record("q", 0.78)])
        self.assertIsNone(rows[0]["recovery_pct"])
        rows, _ = tables([record(evidence={})])
        self.assertIsNone(rows[0]["recovery_pct"])

    def test_replay_summary_and_csv_tsv_agree_bitwise(self):
        base, value = 0.3750636660794323, 0.3546653217459048
        with tempfile.TemporaryDirectory() as temp:
            for full in (False, True):
                root = Path(temp) / str(full)
                package, run, _, saved = replay(root, full=full)
                for state in saved["conditions"]:
                    path = run / state["result"]["path"]
                    result = unseal(path)
                    result["metrics"]["top1"] = (
                        base if result["condition"] == "fp32" else value
                    )
                    seal(path, result)
                    state["result"] = file_record(path, run)
                seal(run / "run.json", saved)
                summary = reevaluation_metrics(run, saved["conditions"])
                expected = [100.0, 94.56136486193054]
                self.assertEqual([r["recovery_percent"] for r in summary], expected)
                output = root / "report"
                counts = report(run=run, package=package, output=output)
                self.assertEqual(2, counts["full" if full else "partial"])
                self.assertEqual(0, counts["excluded"])
                for suffix, separator in (("csv", ","), ("tsv", "\t")):
                    path = output / f"raw_data{'' if full else '.partial'}.{suffix}"
                    with path.open(encoding="utf-8-sig", newline="") as stream:
                        rows = list(csv.DictReader(stream, delimiter=separator))
                    for row, accuracy, recovery in zip(rows, (base, value), expected):
                        self.assertBits(recovery, float(row["recovery_percent"]))
                        self.assertBits(
                            accuracy, json.loads(row["metrics_json"])["top1"]["value"]
                        )
