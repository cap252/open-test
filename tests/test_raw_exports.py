"""Raw CSV/TSV contracts of run exports, the results command and report rebuilds."""

import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from opennpu_quant._io import atomic_json, object_hash, sha256
from opennpu_quant.results import (
    export_records,
    normalize_scheme,
    raw_export_row,
    tables,
)


HEADERS = """model task dataset metric fp32_accuracy int8_accuracy recovery_percent
scheme percentile model_source_url model_source_revision top5_percent AP50_percent AP75_percent
metrics_json""".split()
SCHEME = (
    "scope=all | activation=asymmetric | weight=per_channel | "
    "activation_scale=pot_ceil | weight_scale=pot_ceil | "
    "calibration=percentile | rounding=AdaRound"
)


def fixture():
    catalog = dict(
        schema_version=1,
        models=[
            dict(
                id="toy",
                name="테스트 모델",
                task="object_detection",
                dataset="coco",
                metric="AP",
                expected_samples=100,
                source=dict(repo="https://example.test/model", commit="revision-1"),
            )
        ],
    )
    base = dict(
        record_id="fp32",
        model_id="toy",
        run_id="sweep",
        cohort_id="sweep",
        condition="fp32",
        state="evaluated",
        verified=True,
        full_validation=True,
        split="validation",
        primary_metric="AP",
        metrics={"AP": {"value": 0.8, "unit": "fraction"}},
        samples=100,
        expected_samples=100,
        dataset="coco",
        quantization=None,
        provenance=dict(
            model_identity="source",
            code_identity="code",
            preprocessing_identity="pre",
            data_identity="data",
            environment_identity="env",
        ),
        pair_identity="same-evaluation",
        baseline_record_id="fp32",
        core_compatible=True,
        is_fp32=True,
    )
    quantized = copy.deepcopy(base)
    quantized.update(
        record_id="int8",
        condition="pot_adaround",
        is_fp32=False,
        quantization=normalize_scheme(
            dict(
                scope="all",
                activation_symmetric=False,
                weight_granularity="per_channel",
                activation_scale="pot_ceil",
                weight_scale="pot_ceil",
                method="percentile",
                percentile=99.99,
                adaround={"steps": 10000},
            )
        ),
        metrics={
            "AP": {"value": 78, "unit": "percent"},
            "top5": {"value": 0.95, "unit": "fraction"},
            "AP50": {"value": 90, "unit": "percent"},
            "AP75": {"value": 0.7, "unit": "fraction"},
        },
    )
    return catalog, [base, quantized]


class RawExportTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.catalog, self.records = fixture()

    def run_rows(self):
        return [raw_export_row(row) for row in tables(self.run_records())[0]]

    def test_columns_units_and_source_values(self):
        base, quantized = self.run_rows()
        self.assertEqual(HEADERS, list(quantized))
        self.assertEqual("테스트 모델", quantized["model"])
        self.assertEqual("object_detection", quantized["task"])
        self.assertEqual("coco", quantized["dataset"])
        self.assertEqual("AP", quantized["metric"])
        self.assertEqual(80, quantized["fp32_accuracy"])
        self.assertEqual(78, quantized["int8_accuracy"])
        self.assertEqual(97.5, quantized["recovery_percent"])
        self.assertEqual([95, 90, 70], [quantized[k] for k in HEADERS[11:14]])
        self.assertEqual(99.99, quantized["percentile"])
        self.assertEqual("https://example.test/model", quantized["model_source_url"])
        self.assertEqual("revision-1", quantized["model_source_revision"])
        self.assertEqual(
            self.records[1]["metrics"], json.loads(quantized["metrics_json"])
        )
        self.assertEqual("FP32", base["scheme"])
        self.assertIsNone(base["percentile"])
        self.assertEqual(SCHEME, quantized["scheme"])

    def run_records(self):
        return [
            dict(
                record,
                model="toy",
                model_name="테스트 모델",
                origin="standalone_run",
                complete=True,
                evaluation_identity="same-evaluation",
                model_metadata=self.catalog["models"][0],
                model_source=self.catalog["models"][0]["source"],
                evidence=dict(model_identity="source", preprocessing_identity="pre"),
            )
            for record in self.records
        ]

    def test_run_exports_and_cli_stdout_use_the_same_raw_schema(self):
        records = self.run_records()
        export_records(records, self.root / "run")
        exported = []
        for name in ("accuracy", "results"):
            for suffix, delimiter in (("csv", ","), ("tsv", "\t")):
                with (self.root / f"run/{name}.{suffix}").open(
                    encoding="utf-8-sig", newline=""
                ) as stream:
                    reader = csv.DictReader(stream, delimiter=delimiter)
                    self.assertEqual(HEADERS, reader.fieldnames)
                    exported.append(list(reader))
        self.assertTrue(all(rows == exported[0] for rows in exported))
        row = exported[0][1]
        self.assertEqual("object_detection", row["task"])
        self.assertEqual("테스트 모델", row["model"])
        self.assertEqual(97.5, float(row["recovery_percent"]))
        self.assertEqual(SCHEME, row["scheme"])
        self.assertEqual(99.99, float(row["percentile"]))
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "opennpu_quant",
                "results",
                str(self.root / "run/records.json"),
                "--stdout",
                "tsv",
            ],
            cwd=self.root,
            text=True,
            capture_output=True,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(
            HEADERS, next(csv.reader(result.stdout.splitlines(), delimiter="\t"))
        )
        self.assertEqual((self.root / "run/results.tsv").read_text(), result.stdout)
        saved = json.loads((self.root / "run/records.json").read_text())
        self.assertEqual(records, saved["records"])

    def test_report_rebuild_and_stdout_use_raw_columns(self):
        policy = dict(
            scope="all",
            activation_symmetric=False,
            weight_granularity="per_channel",
            method="percentile",
            percentile=99.99,
            activation_scale="pot_ceil",
            weight_scale="pot_ceil",
        )
        config = dict(
            name="sweep",
            models=["toy"],
            conditions={
                "fp32": {},
                "pot_adaround": {"scheme": "pot", "adaround": True},
            },
            schemes={"pot": policy},
            adaround={"steps": 10000},
            evaluation={"split": "validation"},
        )
        directory = self.root / "report"
        atomic_json(directory / "effective_config.json", config)
        for record in self.records:
            path = directory / "toy" / record["condition"] / "result.json"
            measured = dict(
                config_identity=object_hash(config),
                model_metadata=self.catalog["models"][0],
                title="테스트 모델",
                metric="AP",
                metrics={k: v["value"] for k, v in record["metrics"].items()},
                metric_units={k: v["unit"] for k, v in record["metrics"].items()},
                samples=100,
                expected_samples=100,
                complete=True,
                model_identity="source",
                preprocessing_identity="pre",
                evaluation_identity="same-evaluation",
                scheme=policy if record["quantization"] else None,
            )
            atomic_json(path, measured)
            atomic_json(path.parent / "complete.json", dict(result_sha256=sha256(path)))
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "opennpu_quant",
                "report",
                str(directory),
                "--format",
                "csv,tsv,json",
                "--stdout",
                "tsv",
            ],
            cwd=self.root,
            text=True,
            capture_output=True,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        rows = list(csv.DictReader(result.stdout.splitlines(), delimiter="\t"))
        self.assertEqual(HEADERS, list(rows[0]))
        self.assertEqual("object_detection", rows[1]["task"])
        self.assertEqual(97.5, float(rows[1]["recovery_percent"]))
        self.assertEqual(99.99, float(rows[1]["percentile"]))
        self.assertEqual((directory / "accuracy.tsv").read_text(), result.stdout)
        self.assertEqual((directory / "results.tsv").read_text(), result.stdout)

    def test_percentile_is_separate_and_other_calibration_methods_stay_distinct(self):
        labels = set()
        for percentile in (99.9, 99.99, 99.999):
            self.records[1]["quantization"]["calibration"]["percentile"] = percentile
            row = self.run_rows()[1]
            self.assertEqual(percentile, row["percentile"])
            self.assertIn("calibration=percentile", row["scheme"].split(" | "))
            self.assertNotIn("percentile=", row["scheme"])
            labels.add(row["scheme"])
        self.assertEqual(1, len(labels))
        for method in ("MinMax", "Entropy"):
            self.records[1]["quantization"]["calibration"]["method"] = method
            row = self.run_rows()[1]
            self.assertIsNone(row["percentile"])
            self.assertIn("calibration=" + method.lower(), row["scheme"])
            self.assertNotIn(row["scheme"], labels)


if __name__ == "__main__":
    unittest.main()
