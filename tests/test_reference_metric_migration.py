"""Reference name conversion preserves values and invalidates dependent identities."""

import copy
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from opennpu_quant._io import atomic_json, read_json, sha256
from opennpu_quant.reproduction.assets import catalog, Resolver, unseal
from opennpu_quant.reproduction.prepare import prepare
from opennpu_quant.reproduction.execution import run
from opennpu_quant.reproduction.report import report
from tools.migrate_reference_metrics import canonicalize, exact, migrate
from tests.test_gpu_reproduction import fixture


class ReferenceMetricMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = fixture(self.root)
        source = dict(
            result=dict(
                metrics=dict(top1=1.0, top5=1.0),
                metric_units=dict(top1="fraction", top5="fraction"),
                details=dict(qualcomm_reference=dict(angles=[-0.0, 74.18], pairs=2)),
            )
        )
        atomic_json(self.source / "source.json", source)
        self.selection = self.root / "selection.json"
        selected = read_json(self.selection)
        entry = selected["entries"][0]
        entry["source_result"]["sha256"] = sha256(self.source / "source.json")
        entry["measurement"] = dict(metrics=source["result"]["metrics"], samples=3)
        atomic_json(self.selection, selected)
        self.bindings = self.root / "bindings.json"
        self.output = self.root / "converted"

    def migrate(self, **kw):
        return migrate(
            selection=self.selection, bindings=self.bindings, output=self.output, **kw
        )

    def test_conversion_preserves_float_bits_units_and_exact_duplicates(self):
        value = dict(
            metrics={
                "qualcomm_fixed_angle_accuracy": 0.9906666666666666,
                "qualcomm_angle_accuracy": 0.9906666666666666,
            },
            metric_units={"qualcomm_fixed_angle_accuracy": "fraction"},
            details={
                "qualcomm_angle_threshold_degrees": 74.18,
                "qualcomm_reference": {"angles": [-0.0, 1.0]},
            },
        )
        before = copy.deepcopy(value)
        result = canonicalize(value)
        self.assertEqual(before, value)
        self.assertEqual(
            {"fixed_angle_accuracy": 0.9906666666666666}, result["metrics"]
        )
        self.assertEqual({"fixed_angle_accuracy": "fraction"}, result["metric_units"])
        self.assertEqual(
            np.float64(-0.0).tobytes(),
            np.float64(
                result["details"]["fixed_angle_reference"]["angles"][0]
            ).tobytes(),
        )
        self.assertEqual(exact(result), exact(canonicalize(result)))
        for first, second in (
            (0.99, np.nextafter(0.99, 1.0).item()),
            (0.0, -0.0),
            (1, 1.0),
        ):
            with self.subTest(first=first, second=second):
                with self.assertRaisesRegex(ValueError, "Conflicting"):
                    canonicalize(
                        {
                            "fixed_angle_accuracy": first,
                            "qualcomm_angle_accuracy": second,
                        }
                    )
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            canonicalize({"qualcomm_angle_accuracy": float("nan")})

    def test_hashes_original_preservation_and_idempotent_migration(self):
        original = (
            self.selection.read_bytes(),
            (self.source / "source.json").read_bytes(),
        )
        ledger = self.migrate()
        selected = unseal(self.output / "selected.json")
        entry = selected["entries"][0]
        resolver = Resolver(self.output / "bindings.json")
        value = resolver.json(entry["source_result"])
        self.assertEqual(1, selected["metric_schema_version"])
        self.assertEqual(1, len(ledger["migrated_assets"]))
        self.assertTrue(ledger["numeric_values_unchanged"])
        self.assertEqual(
            sha256(self.output / "selected.json"), ledger["converted_selection_sha256"]
        )
        self.assertEqual(
            exact(canonicalize(read_json(self.source / "source.json"))), exact(value)
        )
        self.assertEqual(
            original,
            (self.selection.read_bytes(), (self.source / "source.json").read_bytes()),
        )
        self.assertEqual(
            original[0], (self.output / "original-selection.json").read_bytes()
        )
        again = self.root / "again"
        second = migrate(
            selection=self.output / "selected.json",
            bindings=self.output / "bindings.json",
            output=again,
        )
        self.assertEqual([], second["migrated_assets"])
        self.assertEqual([], second["migrated_conditions"])
        self.assertEqual(
            (self.output / "selected.json").read_bytes(),
            (again / "selected.json").read_bytes(),
        )
        with self.assertRaises(FileExistsError):
            self.migrate()

    def test_tampering_conflicts_and_mismatched_values_publish_nothing(self):
        with (self.source / "source.json").open("a") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(ValueError, "Artifact changed"):
            self.migrate()
        self.assertFalse(self.output.exists())
        selected = read_json(self.selection)
        selected["entries"][0]["source_result"]["sha256"] = sha256(
            self.source / "source.json"
        )
        selected["entries"][0]["measurement"]["metrics"]["top1"] = 0.0
        atomic_json(self.selection, selected)
        with self.assertRaisesRegex(ValueError, "Selected metrics differ"):
            self.migrate()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".metric-migration-*")))

    def test_binding_collision_and_unsafe_id_are_rejected(self):
        roots = read_json(self.bindings)
        roots["reference-metrics-v1"] = str(self.root / "previous-migration")
        atomic_json(self.bindings, roots)
        with self.assertRaisesRegex(ValueError, "Binding already exists"):
            self.migrate()
        self.assertFalse(self.output.exists())
        roots.pop("reference-metrics-v1")
        atomic_json(self.bindings, roots)
        selection = read_json(self.selection)
        selection["entries"][0]["id"] = "../escape"
        atomic_json(self.selection, selection)
        with self.assertRaisesRegex(ValueError, "Unsafe condition"):
            self.migrate()
        self.assertFalse(self.output.exists())

    def test_new_package_runs_reports_resumes_and_relocates(self):
        before = self.root / "before"
        prepare(
            selection=self.selection,
            bindings=self.bindings,
            output=before,
            limit="full",
        )
        self.migrate()
        relocated = self.root / "relocated"
        shutil.move(self.output, relocated)
        roots = read_json(relocated / "bindings.json")
        roots["reference-metrics-v1"] = str(relocated)
        bindings = self.root / "relocated-bindings.json"
        atomic_json(bindings, roots)
        package = self.root / "after"
        prepare(
            selection=relocated / "selected.json",
            bindings=bindings,
            output=package,
            limit="full",
        )
        left = unseal(before / "manifest.json")["conditions"][0]
        right = unseal(package / "manifest.json")["conditions"][0]
        self.assertEqual(left["data_identity"], right["data_identity"])
        self.assertNotEqual(left["identity"], right["identity"])
        output = self.root / "run"
        args = dict(package=package, bindings=bindings, device="cpu", output=output)
        self.assertEqual(1, run(**args)["completed"])
        result = output / "resnet18__fp32/result.json"
        stamp = result.stat().st_mtime_ns
        self.assertEqual("pass", unseal(result)["original_accuracy"]["metrics_state"])
        self.assertEqual(1, run(**args)["completed"])
        self.assertEqual(stamp, result.stat().st_mtime_ns)
        report(
            run=output,
            package=package,
            output=self.root / "report",
            table_profile="opennpu_raw",
        )
        self.assertTrue((self.root / "report/raw_data.csv").exists())
        with self.assertRaisesRegex(ValueError, "identity|Identity"):
            report(
                run=output,
                package=before,
                output=self.root / "wrong-report",
                table_profile="opennpu_raw",
            )
        self.assertEqual(1, run(**dict(args, package=before))["failed"])
        source = Resolver(bindings).check(right["entry"]["source_result"])
        with source.open("a") as stream:
            stream.write(" ")
        self.assertEqual(1, run(**args)["failed"])

    def test_packaged_selection_uses_canonical_names(self):
        rows = [
            e
            for e in catalog()["entries"]
            if e["model"] == "mobilefacenet_relu_foamliu"
        ]
        self.assertEqual(7, len(rows))
        for row in rows:
            self.assertIn("fixed_angle_accuracy", row["measurement"]["metrics"])
            self.assertFalse(
                any(
                    key.startswith("qualcomm_") for key in row["measurement"]["metrics"]
                )
            )
