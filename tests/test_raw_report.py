import copy
import contextlib
import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from opennpu_quant._io import atomic_json, object_hash, read_json, sha256
from opennpu_quant.raw_report import build_report, write_report
from opennpu_quant.report import report as run_report
from opennpu_quant.results import RAW_FIELDS, export_records
from opennpu_quant.reproduction.assets import catalog, file_record, seal
from opennpu_quant.reproduction.report import report
from tests.test_results import record

FIXTURE = read_json(Path(__file__).parent / "fixtures/opennpu_raw_format.json")


def replay(root, *, full=False, pairs=False):
    package, run = root / "package", root / "run"
    entries = [
        copy.deepcopy(e)
        for e in catalog()["entries"]
        if e["model"] == "resnet18" and e["condition"] in {"fp32", "asym_pc_pot_rtn"}
    ]
    prepared = dict(
        schema_version=1, selection_identity="selection", limit=2, conditions=[]
    )
    saved = dict(schema_version=1, implementation="saved-code", conditions=[])
    for e in entries:
        e["measurement"]["samples"] = 2 if full else 50000
        if pairs:
            e["recipe"].update(
                dataset="lfw_mobilefacenet",
                task="face_verification",
                metric="verification_accuracy",
            )
        limit = 6000 if pairs and full else 2
        r = dict(entry=e, state="ready", data_identity="input", limit=limit)
        r["identity"] = object_hash([e, r["data_identity"], limit])
        prepared["conditions"].append(r)
        count = limit * 4 if pairs else limit
        primary = (
            "verification_accuracy"
            if pairs and full
            else "smoke_pair_accuracy"
            if pairs
            else "top1"
        )
        result = dict(
            id=e["id"],
            model=e["model"],
            condition=e["condition"],
            state="completed",
            identity="run-condition",
            prepared_identity=r["identity"],
            recipe_identity=object_hash(e["recipe"]),
            input_identity="input",
            source_graphs=e["graphs"],
            profile=e["recipe"]["decode"].get("profile", "current_v1"),
            percentile=e.get("percentile"),
            limit=limit,
            environment={"implementation": "saved-code"},
            sample_count=count,
            samples=[{}] * count,
            unit="pairs" if pairs else "images",
            primary_metric=primary,
            metrics={primary: 0.8 if e["condition"] == "fp32" else 0.78},
            metric_units={primary: "fraction"},
            stages={"full_accuracy": "executed" if full else "pending"},
        )
        path = run / e["id"] / "result.json"
        seal(path, result)
        saved["conditions"].append(
            dict(
                id=e["id"],
                state="completed",
                samples=count,
                result=file_record(path, run),
            )
        )
    seal(package / "manifest.json", prepared)
    seal(run / "run.json", saved)
    return package, run, prepared, saved


class RawReportTests(unittest.TestCase):
    def test_run_and_replay_share_formats_and_measurements(self):
        expected_formats = {
            "csv": {"results.csv", "metrics.csv"},
            "tsv": {"results.tsv", "metrics.tsv"},
            "json": {"records.json", "summary.json"},
            "md": {"summary.md"},
        }
        for full in (False, True):
            for profile in (None, "opennpu_raw"):
                for formats in (None, "csv", "tsv", "json", "md", "csv,tsv,json,md"):
                    with (
                        self.subTest(full=full, profile=profile, formats=formats),
                        tempfile.TemporaryDirectory() as temp,
                    ):
                        root = Path(temp)
                        package, run, prepared, _ = replay(root / "replay", full=full)
                        standard = root / "ordinary"
                        entries = [r["entry"] for r in prepared["conditions"]]
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
                            name="test",
                            models=["resnet18"],
                            conditions={
                                "fp32": {},
                                "asym_pc_pot_rtn": {"scheme": "pot"},
                            },
                            schemes={"pot": policy},
                            evaluation={"split": "validation"},
                        )
                        atomic_json(standard / "effective_config.json", config)
                        for entry in entries:
                            fp32 = entry["condition"] == "fp32"
                            path = (
                                standard
                                / entry["model"]
                                / entry["condition"]
                                / "result.json"
                            )
                            atomic_json(
                                path,
                                dict(
                                    title=entry["recipe"]["title"],
                                    metric="top1",
                                    metrics={"top1": 0.8 if fp32 else 0.78},
                                    samples=2,
                                    expected_samples=2 if full else 50000,
                                    complete=full,
                                    full_evaluation=full,
                                    reference=False,
                                    config_identity=object_hash(config),
                                    model_identity="model",
                                    preprocessing_identity="pre",
                                    evaluation_identity="eval",
                                    model_metadata=entry["recipe"],
                                    scheme=None if fp32 else policy,
                                ),
                            )
                            atomic_json(
                                path.parent / "complete.json",
                                dict(result_sha256=sha256(path)),
                            )
                        result = run_report(
                            standard, formats=formats, table_profile=profile
                        )
                        output = root / "tables"
                        counts = report(
                            run=run,
                            package=package,
                            output=output,
                            formats=formats,
                            table_profile=profile,
                        )
                        selected = (formats or "csv,json").split(",")
                        expected = set().union(*(expected_formats[f] for f in selected))
                        if profile:
                            expected.add("raw_data.metadata.json")
                            for suffix in ("csv", "tsv"):
                                if suffix in selected:
                                    expected.remove("results." + suffix)
                                    expected.update(
                                        {
                                            "raw_data." + suffix,
                                            "raw_data.partial." + suffix,
                                        }
                                    )
                        self.assertEqual(expected, {p.name for p in output.iterdir()})
                        self.assertEqual(
                            expected,
                            {p.name for p in standard.iterdir() if p.is_file()}
                            - {"effective_config.json"},
                        )
                        self.assertEqual(2, counts["full" if full else "partial"])
                        self.assertEqual(97.5, result["rows"][1]["recovery_pct"])
                        for filename in expected:
                            if filename.endswith(
                                (".csv", ".tsv")
                            ) and not filename.startswith("metrics."):
                                self.assertEqual(
                                    (standard / filename).read_bytes(),
                                    (output / filename).read_bytes(),
                                )
                        if "json" in selected:
                            replay_summary = read_json(output / "summary.json")
                            self.assertEqual(
                                result["mean_recovery"], replay_summary["mean_recovery"]
                            )
                        self.assertFalse(list(output.glob("accuracy.*")))
                        self.assertFalse(list(output.glob("*.xlsx")))

    def test_cli_formats_match_and_invalid_formats_do_not_publish(self):
        from opennpu_quant.cli import main, build_parser
        from tests.test_review_fixes import ReportPolicyTests

        parser = build_parser()
        for command in (
            ["run"],
            ["report", "run"],
            ["results", "records.json"],
            [
                "reproduce",
                "report",
                "--run",
                "run",
                "--package",
                "package",
                "--output",
                "tables",
            ],
        ):
            parsed = parser.parse_args(command)
            self.assertIsNone(parsed.format)
            self.assertIsNone(parsed.table_profile)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package, run, _, _ = replay(root / "replay")
            ordinary = root / "ordinary"
            ReportPolicyTests().fixture(ordinary)
            run_report(ordinary)
            for formats in ("csv", "md", "tsv,json"):
                with self.subTest(formats=formats):
                    replay_out = root / ("replay-" + formats)
                    portable_out = root / ("portable-" + formats)
                    commands = [
                        ["report", str(ordinary)],
                        [
                            "results",
                            str(ordinary / "records.json"),
                            "--output",
                            str(portable_out),
                        ],
                        [
                            "reproduce",
                            "report",
                            "--run",
                            str(run),
                            "--package",
                            str(package),
                            "--output",
                            str(replay_out),
                        ],
                    ]
                    for command in commands:
                        with contextlib.redirect_stdout(io.StringIO()):
                            main(
                                [
                                    *command,
                                    "--format",
                                    formats,
                                    "--table-profile",
                                    "opennpu_raw",
                                ]
                            )
                    self.assertEqual(
                        {p.name for p in portable_out.iterdir()},
                        {p.name for p in replay_out.iterdir()},
                    )
                    self.assertFalse((replay_out / "results.csv").exists())
                    self.assertEqual(
                        "tsv" in formats, (replay_out / "raw_data.tsv").exists()
                    )
            for formats in ("xlsx", "csv,", ""):
                out = root / "invalid"
                with self.assertRaisesRegex(ValueError, "Supported report formats"):
                    report(run=run, package=package, output=out, formats=formats)
                self.assertFalse(out.exists())

    def test_portable_summary_does_not_average_separate_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            export_records(
                [
                    record(run_id="first", model="a", full_evaluation=True),
                    record(run_id="second", model="b", full_evaluation=True),
                ],
                temp,
            )
            self.assertIsNone(
                read_json(Path(temp) / "summary.json")["mean_recovery"]["fp32"]
            )

    def test_workbook_format_all_models_and_schemes(self):
        self.assertEqual(RAW_FIELDS, FIXTURE["fields"])
        models = set()
        for e in catalog()["entries"]:
            recipe = e["recipe"]
            from opennpu_quant.reproduction.report import _quantization

            r = record(
                e["condition"],
                model=e["model"],
                model_name=recipe["title"],
                task=recipe["task"],
                dataset=recipe["dataset"],
                quantization=_quantization(e),
            )
            raw = build_report([r])["partial"][0]
            models.add((raw["model"], raw["task"], raw["dataset"]))
            self.assertIn(raw["scheme"], FIXTURE["schemes"])
            self.assertEqual(RAW_FIELDS, list(raw))
        self.assertEqual({tuple(m) for m in FIXTURE["models"]}, models)

    def test_precision_units_full_partial_and_default_unchanged(self):
        value = 0.6968412345678912
        base = record(value=value, full_evaluation=True)
        quantized = record("q", 0.67890123456789, full_evaluation=True)
        partial = record(model="partial", samples=2, full_evaluation=False)
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            export_records([base, quantized, partial], p)
            before = (p / "results.csv").read_bytes()
            self.assertFalse((p / "raw_data.csv").exists())
            export_records([base, quantized, partial], p, table_profile="opennpu_raw")
            self.assertEqual(before, (p / "results.csv").read_bytes())
            self.assertTrue(
                (p / "raw_data.csv").read_bytes().startswith(b"\xef\xbb\xbf")
            )
            rows = list(
                csv.DictReader(
                    io.StringIO((p / "raw_data.csv").read_text(encoding="utf-8-sig"))
                )
            )
            self.assertEqual(2, len(rows))
            self.assertEqual(value * 100, float(rows[0]["fp32_accuracy"]))
            self.assertEqual(
                value, json.loads(rows[0]["metrics_json"])["top1"]["value"]
            )
            self.assertEqual(
                100 * 0.67890123456789 / value,
                float(rows[1]["recovery_percent"]),
            )
            self.assertEqual(
                2, len((p / "raw_data.partial.csv").read_text().splitlines())
            )

    def test_missing_mismatched_zero_or_unverified_denominator(self):
        for case in (
            "missing",
            "identity",
            "profile",
            "scope",
            "zero",
            "proof",
            "reference",
            "sample_unit",
            "expected_samples",
            "view_count",
        ):
            with self.subTest(case=case):
                b, q = record(), record("q", 0.78)
                records = [b, q]
                if case == "missing":
                    records = [q]
                if case == "identity":
                    q["evaluation_identity"] = "changed"
                if case == "profile":
                    q["evaluation_profile"] = "changed"
                if case == "scope":
                    b["full_evaluation"] = True
                if case == "zero":
                    b["metrics"]["top1"]["value"] = 0
                if case == "proof":
                    b["evidence"]["result_hash_verified"] = False
                if case == "reference":
                    b["reference"] = True
                if case in {"sample_unit", "expected_samples", "view_count"}:
                    q[case] = "changed"
                result = build_report(records)
                row = result["partial"][-1]
                self.assertIsNone(row["recovery_percent"])
                self.assertEqual(78, row["int8_accuracy"])

    def test_failures_missing_metrics_and_original_values_are_not_exported(self):
        values = [
            record(state="fail"),
            record(state="pending"),
            record(metrics={}),
            record(reference=True),
            record(origin="custom"),
        ]
        for i, r in enumerate(values):
            r["model"] = str(i)
            r["measurement"] = {"metrics": {"top1": 0.99}}
        result = build_report(values)
        self.assertEqual([], result["full"] + result["partial"])
        self.assertEqual(5, result["metadata"]["counts"]["excluded"])
        self.assertTrue(all(r["reason"] for r in result["metadata"]["records"]))

    def test_standard_run_report_and_cli_are_metadata_only(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            config = dict(
                name="test",
                models=["m"],
                conditions={"fp32": {}},
                evaluation={"split": "validation"},
            )
            atomic_json(p / "effective_config.json", config)
            path = p / "m/fp32/result.json"
            atomic_json(
                path,
                dict(
                    title="M",
                    metrics={"top1": 0.7},
                    metric="top1",
                    samples=2,
                    expected_samples=2,
                    complete=True,
                    full_evaluation=True,
                    config_identity=object_hash(config),
                    model_identity="m",
                    preprocessing_identity="p",
                    evaluation_identity="e",
                    model_metadata={
                        "task": "classification",
                        "metric": "top1",
                        "dataset": "imagenet",
                    },
                ),
            )
            atomic_json(path.parent / "complete.json", dict(result_sha256=sha256(path)))
            result = run_report(p, table_profile="opennpu_raw")
            self.assertEqual(1, len(result["raw_data"]["full"]))
            for args in (
                ["report", str(p), "--table-profile", "opennpu_raw"],
                [
                    "results",
                    str(p / "records.json"),
                    "--output",
                    str(p / "tables"),
                    "--table-profile",
                    "opennpu_raw",
                ],
            ):
                self.metadata_cli(args)
            path.write_text("{}")
            self.assertEqual(
                1,
                run_report(p, table_profile="opennpu_raw")["raw_data"]["metadata"][
                    "counts"
                ]["excluded"],
            )

    def metadata_cli(self, args):
        code = """
import sys, json, contextlib, io
from opennpu_quant.cli import main
with contextlib.redirect_stdout(io.StringIO()):
    main(json.loads(sys.argv[1]))
print(json.dumps(sorted({'numpy','onnx','onnxruntime','torch','tensorflow','cv2'} & set(sys.modules))))
"""
        result = subprocess.run(
            [sys.executable, "-O", "-c", code, json.dumps(args)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual([], json.loads(result.stdout))

    def test_replay_full_partial_pairs_and_no_inference_cli(self):
        for full, pairs in ((False, False), (True, False), (False, True), (True, True)):
            with (
                self.subTest(full=full, pairs=pairs),
                tempfile.TemporaryDirectory() as temp,
            ):
                root = Path(temp)
                package, run, _, _ = replay(root, full=full, pairs=pairs)
                output = root / "tables"
                if pairs and not full:
                    with self.assertRaisesRegex(ValueError, "full 6000"):
                        report(run=run, package=package, output=output)
                    self.assertFalse(output.exists())
                    continue
                result = report(run=run, package=package, output=output)
                self.assertEqual(2, result["full" if full else "partial"])
                data = read_json(output / "records.json")["records"]
                self.assertEqual(
                    0.8, data[0]["metrics"][data[0]["primary_metric"]]["value"]
                )
                if pairs:
                    self.assertEqual(6000 if full else 2, data[0]["samples"])
                    self.assertEqual(24000 if full else 8, data[0]["view_count"])
                self.metadata_cli(
                    [
                        "reproduce",
                        "report",
                        "--run",
                        str(run),
                        "--package",
                        str(package),
                        "--output",
                        str(output),
                        "--table-profile",
                        "opennpu_raw",
                    ]
                )

    def test_completed_and_missing_models_report_without_inference(self):
        from opennpu_quant.reproduction.assets import unseal
        from opennpu_quant.reproduction.execution import run as execute
        from opennpu_quant.reproduction.prepare import prepare
        from tests.test_gpu_reproduction import fixture

        for missing_condition, filename in (
            ("fp32", "model.onnx"),
            ("asym_pc_pot_rtn", "qdq.onnx"),
        ):
            with (
                self.subTest(missing=missing_condition),
                tempfile.TemporaryDirectory() as temp,
            ):
                root = Path(temp)
                assets = fixture(root, int8=True)
                (assets / filename).unlink()
                package, run = root / "prepared", root / "run"
                prepared = prepare(
                    bindings=root / "bindings.json",
                    selection=root / "selection.json",
                    output=package,
                    limit="full",
                )
                self.assertEqual({"ready": 1, "pending": 1}, prepared["counts"])
                executed = execute(
                    package=package,
                    bindings=root / "bindings.json",
                    output=run,
                    device="cpu",
                )
                self.assertEqual(1, executed["completed"])
                self.assertEqual(1, executed["pending"])
                self.assertEqual(0, executed["failed"])
                states = unseal(run / "run.json")["conditions"]
                pending = next(r for r in states if r["state"] == "pending")
                completed = next(r for r in states if r["state"] == "completed")
                missing = pending["missing"]
                self.assertIsInstance(missing, list)
                self.assertTrue(any(filename in detail for detail in missing))
                measured = unseal(run / completed["result"]["path"])
                accuracy = measured["metrics"]["top1"] * 100
                originals = {
                    p: p.read_bytes()
                    for directory in (run, package)
                    for p in directory.rglob("*.json")
                }
                for profile in (None, "opennpu_raw"):
                    output = root / str(profile)
                    command = [
                        "reproduce",
                        "report",
                        "--run",
                        str(run),
                        "--package",
                        str(package),
                        "--output",
                        str(output),
                        "--format",
                        "csv,tsv,json,md",
                    ]
                    if profile:
                        command += ["--table-profile", profile]
                    self.metadata_cli(command)
                    records = read_json(output / "records.json")["records"]
                    self.assertEqual(2, len(records))
                    missing_record = next(r for r in records if r["state"] == "pending")
                    self.assertEqual("state:pending", missing_record["failure_reason"])
                    self.assertEqual(missing, missing_record["failure_details"])
                    summary = read_json(output / "summary.json")
                    by_condition = {r["condition"]: r for r in summary["rows"]}
                    missing_row = by_condition[missing_condition]
                    self.assertEqual(
                        "measurement:state:pending", missing_row["pairing_reason"]
                    )
                    self.assertEqual(missing, missing_row["failure_details"])
                    self.assertIsNone(missing_row["value_pct"])
                    self.assertFalse(missing_row["complete"])
                    valid = by_condition[completed["condition"]]
                    self.assertTrue(valid["complete"])
                    self.assertEqual(accuracy, valid["value_pct"])
                    if missing_condition == "fp32":
                        self.assertEqual(
                            "baseline:state:pending", valid["pairing_reason"]
                        )
                        self.assertIsNone(valid["recovery_pct"])
                    for suffix, delimiter in (("csv", ","), ("tsv", "\t")):
                        report_filename = (
                            ("raw_data" if profile else "results") + "." + suffix
                        )
                        with (output / report_filename).open(
                            encoding="utf-8-sig", newline=""
                        ) as stream:
                            rows = list(csv.DictReader(stream, delimiter=delimiter))
                        self.assertEqual(1 if profile else 2, len(rows))
                        cell = (
                            "int8_accuracy"
                            if missing_condition == "fp32"
                            else "fp32_accuracy"
                        )
                        self.assertTrue(
                            any(
                                r[cell] != "" and float(r[cell]) == accuracy
                                for r in rows
                            )
                        )
                    if profile:
                        metadata = read_json(output / "raw_data.metadata.json")
                        self.assertEqual(
                            {"full": 1, "partial": 0, "excluded": 1}, metadata["counts"]
                        )
                        excluded = next(
                            r
                            for r in metadata["records"]
                            if r["condition"] == missing_condition
                        )
                        self.assertEqual("state:pending", excluded["reason"])
                        self.assertEqual(missing, excluded["failure_details"])
                    portable = root / (str(profile) + "-portable")
                    args = [
                        "results",
                        str(output / "records.json"),
                        "--output",
                        str(portable),
                        "--format",
                        "csv,tsv,json,md",
                    ]
                    if profile:
                        args += ["--table-profile", profile]
                    self.metadata_cli(args)
                    self.assertEqual(
                        (output / "summary.json").read_bytes(),
                        (portable / "summary.json").read_bytes(),
                    )
                self.assertEqual(originals, {p: p.read_bytes() for p in originals})

    def test_replay_relocation_missing_and_failed_conditions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package, run, prepared, saved = replay(root)
            row = prepared["conditions"][1]
            row["entry"]["graphs"]["postprocess"] = {
                "id": "artifact://test/host.onnx",
                "sha256": "different-host",
            }
            row["identity"] = object_hash(
                [row["entry"], row["data_identity"], row["limit"]]
            )
            path = run / saved["conditions"][1]["result"]["path"]
            value = read_json(path)
            value.update(
                source_graphs=row["entry"]["graphs"], prepared_identity=row["identity"]
            )
            seal(path, value)
            saved["conditions"][1]["result"] = file_record(path, run)
            seal(package / "manifest.json", prepared)
            seal(run / "run.json", saved)
            report(
                run=run,
                package=package,
                output=root / "tables",
                table_profile="opennpu_raw",
            )
            rows = list(
                csv.DictReader(
                    (root / "tables/raw_data.partial.csv").open(encoding="utf-8-sig")
                )
            )
            self.assertEqual("", rows[1]["recovery_percent"])
            self.assertEqual("78.0", rows[1]["int8_accuracy"])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package, run, prepared, saved = replay(root)
            saved["conditions"][0] = dict(
                id=saved["conditions"][0]["id"],
                state="fail",
                error="failed measurement",
            )
            saved["conditions"].pop()
            seal(run / "run.json", saved)
            shutil.move(str(package), str(root / "moved_package"))
            shutil.move(str(run), str(root / "moved_run"))
            result = report(
                run=root / "moved_run",
                package=root / "moved_package",
                output=root / "tables",
            )
            self.assertEqual(2, result["excluded"])
            self.assertEqual(0, result["partial"])

    def test_replay_tampering_rejected_before_output(self):
        for target in ("run", "package", "result"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                package, run, _, saved = replay(root)
                path = {
                    "run": run / "run.json",
                    "package": package / "manifest.json",
                    "result": run / saved["conditions"][0]["result"]["path"],
                }[target]
                path.write_text(path.read_text() + " ")
                with self.assertRaisesRegex(ValueError, "changed"):
                    report(
                        run=run,
                        package=package,
                        output=root / "tables",
                        table_profile="opennpu_raw",
                    )
                self.assertFalse((root / "tables").exists())

    def test_resealed_identity_mismatches_rejected(self):
        for field in (
            "input_identity",
            "prepared_identity",
            "recipe_identity",
            "profile",
            "source_graphs",
            "limit",
            "sample_count",
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                package, run, _, saved = replay(root)
                path = run / saved["conditions"][0]["result"]["path"]
                value = read_json(path)
                value[field] = "changed"
                seal(path, value)
                saved["conditions"][0]["result"] = file_record(path, run)
                seal(run / "run.json", saved)
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    report(
                        run=run,
                        package=package,
                        output=root / "tables",
                        table_profile="opennpu_raw",
                    )
                self.assertFalse((root / "tables").exists())

    def test_empty_report_has_headers_and_metadata(self):
        face = record(
            model_metadata={
                "task": "face_verification",
                "decode": {"kind": "face_verification"},
            },
            metric_details={"pair_count": 6000},
        )
        meta = build_report([face])["metadata"]["records"][0]
        self.assertEqual("views", meta["sample_unit"])
        self.assertEqual(100, meta["view_count"])
        self.assertEqual(6000, meta["pair_count"])
        self.assertEqual("current_v1", meta["evaluation_profile"])
        with tempfile.TemporaryDirectory() as temp:
            write_report(build_report([]), temp, formats="csv,tsv")
            self.assertEqual(
                "\t".join(RAW_FIELDS) + "\n", (Path(temp) / "raw_data.tsv").read_text()
            )
