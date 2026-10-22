"""Full LFW pair order, folds, views and rejection before mixed-model work."""

import copy
import contextlib
import io
import tempfile
import unittest
from tests.library_fixtures import small_config

from pathlib import Path
from unittest.mock import patch

import numpy as np

from opennpu_quant._io import atomic_json, read_json, sha256
from opennpu_quant.configuration import load_config, validate
from opennpu_quant.data.lfw_protocol import require_full, reject_partial_result
from opennpu_quant.evaluation.face_verification import verification, read_pairs


def write_pairs(path):
    path.write_text(
        "10\t300\n"
        + "".join("A\t1\t2\n" if i % 600 < 300 else "A\t1\tB\t1\n" for i in range(6000))
    )
    return read_pairs(path)


def full_manifest(root):
    from PIL import Image

    pair_path = root / "pairs.txt"
    pairs = write_pairs(pair_path)
    keys = sorted({p[k] for p in pairs for k in ("first", "second")})
    Image.new("RGB", (112, 112)).save(root / "image.png")
    atomic_json(
        root / "alignment.json",
        dict(kind="lfw_mobilefacenet", smoke=False, failures=[]),
    )
    info = dict(
        dataset="lfw_mobilefacenet",
        root=str(root),
        smoke=False,
        expected_samples=6,
        annotations=dict(pairs="pairs.txt", alignment="alignment.json"),
        evaluation_pairs=pairs,
        calibration=[
            dict(id="cal", path="image.png", image_key="C/C_0001.jpg", flip=0)
        ],
        validation=[
            dict(id=k + str(f), path="image.png", image_key=k, flip=f)
            for k in keys
            for f in (0, 1)
        ],
    )
    path = root / "manifest.json"
    atomic_json(path, info)
    return path, sha256(pair_path)


class LfwProtocolTests(unittest.TestCase):
    def test_folds_flip_views_and_full_only_scoring(self):
        with tempfile.TemporaryDirectory() as temp:
            pairs = write_pairs(Path(temp) / "pairs.txt")
        first, second = np.eye(2, dtype=np.float32)
        vectors = {
            "A/A_0001.jpg": {0: first, 1: first},
            "A/A_0002.jpg": {0: first, 1: first},
            "B/B_0001.jpg": {0: second, 1: second},
        }
        before = copy.deepcopy(vectors)
        metrics, details = verification(vectors, pairs, dimensions=2)
        self.assertEqual(1.0, metrics["verification_accuracy"])
        self.assertEqual(10, len(details["folds"]))
        self.assertTrue(all(row["threshold"] == 0.01 for row in details["folds"]))
        for key in vectors:
            for flip in (0, 1):
                self.assertEqual(
                    vectors[key][flip].tobytes(), before[key][flip].tobytes()
                )
        with self.assertRaisesRegex(ValueError, "full 6000"):
            verification(vectors, [pairs[0], pairs[300]], dimensions=2)
        with self.assertRaisesRegex(ValueError, "Incomplete/reordered"):
            verification(vectors, list(reversed(pairs)), dimensions=2)
        vectors["A/A_0001.jpg"].pop(1)
        with self.assertRaisesRegex(ValueError, "original and horizontal"):
            verification(vectors, pairs, dimensions=2)

    def test_scope_boundaries_and_removed_options(self):
        from opennpu_quant.cli import build_parser

        for limit in (None, "full", 6000, 6001):
            require_full(limit=limit)
        for limit in (0, 1, 3, 5999, True):
            with (
                self.subTest(limit=limit),
                self.assertRaisesRegex(ValueError, "full 6000"),
            ):
                require_full(limit=limit)
        for args in (
            ["run", "--functional-check"],
            ["reproduce", "prepare", "--functional-check"],
            ["data", "prepare", "lfw_mobilefacenet", "--root", ".", "--smoke"],
        ):
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                build_parser().parse_args(args)
            self.assertEqual(2, error.exception.code)
        config = small_config()
        config["evaluation"]["functional_check"] = True
        with self.assertRaisesRegex(ValueError, "Unknown setting.*functional_check"):
            validate(config)

    def test_legacy_records_are_rejected_without_numpy_or_artifact_rewrite(self):
        from opennpu_quant.results import tables, read_records

        variants = [
            dict(primary_metric="smoke_pair_accuracy"),
            dict(metrics={"smoke_pair_accuracy": {"value": 1.0, "unit": "fraction"}}),
            dict(evaluation_mode="functional_check"),
            dict(task="face_verification", limit=4),
            dict(task="face_verification", metric_details=dict(pair_count=2)),
        ]
        for row in variants:
            with (
                self.subTest(row=row),
                self.assertRaisesRegex(ValueError, "full 6000"),
            ):
                tables([row])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "records.json"
            atomic_json(
                path, dict(schema_version=1, records=[], functional_checks=[{}])
            )
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "full 6000"):
                read_records(path)
            self.assertEqual(before, path.read_bytes())
        reject_partial_result(dict(task="face_verification", limit=6000))

    def test_mixed_run_rejects_before_lock_calibration_or_inference(self):
        from opennpu_quant.paths import Paths
        from opennpu_quant.runner import run

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = load_config("cnn30")
            config["models"] = ["resnet18", "mobilefacenet_relu_foamliu"]
            config["evaluation"]["limit"] = 3
            paths = Paths.resolve(home=root)
            with (
                patch("opennpu_quant.runner.RunLock") as lock,
                patch("opennpu_quant.runner._run_model") as work,
            ):
                with self.assertRaisesRegex(ValueError, "resnet18,yolov5s"):
                    run(config, paths)
            lock.assert_not_called()
            work.assert_not_called()
            self.assertEqual([], list(root.iterdir()))

    def test_mixed_prepare_and_old_package_reject_before_work(self):
        from tests.test_gpu_reproduction import fixture
        from opennpu_quant.reproduction.assets import seal
        from opennpu_quant.reproduction.prepare import prepare
        from opennpu_quant.reproduction.execution import run

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            selection = read_json(root / "selection.json")
            face = copy.deepcopy(selection["entries"][0])
            face.update(
                id="mobilefacenet_relu_foamliu__fp32",
                model="mobilefacenet_relu_foamliu",
            )
            face["recipe"].update(task="face_verification", dataset="lfw_mobilefacenet")
            selection["entries"].append(face)
            atomic_json(root / "selection.json", selection)
            with patch("opennpu_quant.reproduction.data.dataset_for") as dataset:
                with self.assertRaisesRegex(ValueError, "full 6000"):
                    prepare(
                        bindings=root / "bindings.json",
                        selection=root / "selection.json",
                        output=root / "prepared",
                    )
            dataset.assert_not_called()
            self.assertFalse((root / "prepared").exists())
            # A malformed full-scope face descriptor must fail before other models.
            with self.assertRaisesRegex(ValueError, "full 6000"):
                prepare(
                    bindings=root / "bindings.json",
                    selection=root / "selection.json",
                    output=root / "bad-full",
                    limit="full",
                )
            self.assertFalse((root / "bad-full").exists())
            package = root / "legacy"
            package.mkdir()
            seal(
                package / "manifest.json",
                dict(limit=3, conditions=[dict(entry=face, state="ready")]),
            )
            with patch("opennpu_quant.reproduction.environment.probe_gpu") as probe:
                with self.assertRaisesRegex(ValueError, "full 6000"):
                    run(
                        package=package,
                        bindings=root / "bindings.json",
                        output=root / "out",
                        device="cuda:0",
                    )
            probe.assert_not_called()
            self.assertFalse((root / "out").exists())

    def test_legacy_report_and_compare_rejected_including_missing_peer(self):
        from tests.test_raw_report import replay
        from opennpu_quant.reproduction.assets import seal
        from opennpu_quant.reproduction.report import report
        from opennpu_quant.reproduction.compare import compare

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package, run, _, _ = replay(root, pairs=True)
            before = {str(p): sha256(p) for p in root.rglob("*.json")}
            with self.assertRaisesRegex(ValueError, "full 6000"):
                report(run=run, package=package, output=root / "report")
            peer = root / "peer"
            peer.mkdir()
            seal(peer / "run.json", dict(conditions=[]))
            with self.assertRaisesRegex(ValueError, "full 6000"):
                compare(actual=run, expected=peer, output=root / "compare")
            self.assertFalse((root / "report").exists())
            self.assertFalse((root / "compare").exists())
            self.assertEqual(before, {p: sha256(p) for p in before})


class LfwManifestTests(unittest.TestCase):
    def test_full_dataset_order_views_identity_and_calibration(self):
        from opennpu_quant.data.samples import Dataset
        from opennpu_quant.reproduction.assets import Resolver
        from opennpu_quant.reproduction.data import dataset_for

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, digest = full_manifest(root)
            entry = dict(
                dataset=dict(
                    manifest=dict(
                        id="artifact://data/manifest.json", sha256=sha256(path)
                    ),
                    root="artifact://data/",
                )
            )
            with patch("opennpu_quant.data.lfw.PAIR_HASH", digest):
                dataset = Dataset(path)
                before = copy.deepcopy(dataset.info)
                cal = dataset.identity("calibration", 1)
                replay = dataset_for(entry, Resolver({"data": str(root)}), 6000)
                self.assertEqual(before, replay.info)
                for count in (6000, 6001, None):
                    self.assertEqual(
                        before["validation"], dataset.rows("validation", count)
                    )
                    self.assertEqual(
                        dataset.identity("validation"),
                        dataset.identity("validation", count),
                    )
                self.assertEqual(cal, replay.identity("calibration", 1))
                with self.assertRaisesRegex(ValueError, "full 6000"):
                    dataset.rows("validation", 3)
                self.assertEqual(before, dataset.info)

    def test_full_package_execution_report_and_compact_resume(self):
        import onnx
        from tests.test_gpu_reproduction import fixture
        from opennpu_quant.models.spec import ModelSpec
        from opennpu_quant.reproduction.assets import unseal
        from opennpu_quant.reproduction.prepare import prepare
        from opennpu_quant.reproduction.execution import run
        from opennpu_quant.reproduction.report import report

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            assets = fixture(root)
            manifest, pair_hash = full_manifest(assets)
            selection = read_json(root / "selection.json")
            entry = selection["entries"][0]
            name = "mobilefacenet_relu_foamliu"
            recipe = ModelSpec.load(name).recipe
            recipe["input"]["shape"] = [1, 3, 4, 4]
            recipe["preprocess"] = dict(
                kind="center_crop",
                interpolation="bilinear",
                resize=4,
                crop=4,
                mean=[0.1, 0.1, 0.1],
                std=[1.0, 1.0, 1.0],
            )
            # A six-view graph exercises the complete 6000-pair scoring protocol.
            model = onnx.load(assets / "model.onnx")
            model.graph.output[0].name = "embedding"
            model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 128
            model.graph.node[-1].output[0] = "embedding"
            model.graph.initializer[0].CopyFrom(
                onnx.numpy_helper.from_array(
                    np.arange(384, dtype=np.float32).reshape(3, 128), "weight"
                )
            )
            onnx.save(model, assets / "model.onnx")
            entry.update(
                id=name + "__fp32",
                model=name,
                recipe=recipe,
                measurement=dict(samples=6000),
            )
            entry["graphs"]["network"]["sha256"] = sha256(assets / "model.onnx")
            entry["dataset"]["manifest"] = dict(
                id="artifact://assets/manifest.json", sha256=sha256(manifest)
            )
            atomic_json(root / "selection.json", selection)
            kwargs = dict(
                bindings=root / "bindings.json", selection=root / "selection.json"
            )
            with patch("opennpu_quant.data.lfw.PAIR_HASH", pair_hash):
                prepare(**kwargs, output=root / "package", limit="full")
                prepare(**kwargs, output=root / "large", limit=6001)
                prepared = unseal(root / "package/manifest.json")
                self.assertEqual(
                    prepared["conditions"],
                    unseal(root / "large/manifest.json")["conditions"],
                )
                args = dict(
                    package=root / "package",
                    bindings=root / "bindings.json",
                    output=root / "run",
                    device="cpu",
                    storage="predictions",
                )
                outcome = run(**args)
                self.assertEqual(
                    1, outcome["completed"], unseal(root / "run/run.json")["conditions"]
                )
                folder = root / "run" / entry["id"]
                result = unseal(folder / "result.json")
                self.assertEqual(6000, result["metric_details"]["pair_count"])
                self.assertEqual(6, result["sample_count"])
                self.assertEqual("executed", result["stages"]["full_accuracy"])
                before = (folder / "result.json").read_bytes()
                self.assertEqual(1, run(**args)["completed"])
                self.assertEqual(before, (folder / "result.json").read_bytes())
                (folder / "result.json").unlink()
                self.assertEqual(1, run(**args)["completed"])
                self.assertEqual(
                    result["metrics"], unseal(folder / "result.json")["metrics"]
                )
                table = report(
                    run=root / "run", package=root / "package", output=root / "report"
                )
                self.assertEqual(
                    (1, 0, 0), (table["full"], table["partial"], table["excluded"])
                )

    def test_smoke_and_pair_or_view_mutation_rejected(self):
        from opennpu_quant.data.samples import Dataset

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, digest = full_manifest(root)
            original = read_json(path)
            for mutation, message in (
                (lambda r: r.update(smoke=True), "full 6000"),
                (
                    lambda r: r.update(evaluation_pairs=r["evaluation_pairs"][:2]),
                    "full 6000",
                ),
                (
                    lambda r: r["evaluation_pairs"][0].update(same=False),
                    "pairs/fold order",
                ),
                (lambda r: r["validation"].pop(), "image view"),
            ):
                row = copy.deepcopy(original)
                mutation(row)
                atomic_json(path, row)
                with (
                    patch("opennpu_quant.data.lfw.PAIR_HASH", digest),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    Dataset(path)


class LfwAlignmentDependencyTests(unittest.TestCase):
    def test_missing_alignment_dependency_fails_before_download_or_output(self):
        import builtins
        from opennpu_quant.data.lfw import prepare_lfw

        original = builtins.__import__

        def without_skimage(name, *args, **kwargs):
            if name == "skimage" or name.startswith("skimage."):
                raise ModuleNotFoundError("No module named skimage")
            return original(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with (
                patch("builtins.__import__", side_effect=without_skimage),
                patch("opennpu_quant.data.lfw.download_inputs") as download,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "scikit-image==0.25.2.*uv pip install"
                ):
                    prepare_lfw(
                        root / "source", root / "data/manifest.json", download=True
                    )
            download.assert_not_called()
            self.assertEqual([], list(root.iterdir()))

    def test_identity_landmarks_preserve_pixels_with_the_installed_dependency(self):
        from opennpu_quant.data.lfw import align_mobilefacenet

        image = np.random.default_rng(1).integers(0, 256, (112, 112, 3), dtype=np.uint8)
        points = np.array(
            [
                [38.29459953, 51.69630051],
                [73.53179932, 51.50139999],
                [56.02519989, 71.73660278],
                [41.54930115, 92.3655014],
                [70.72990036, 92.20410156],
            ],
            dtype=np.float32,
        )
        actual = align_mobilefacenet(image, points.T.reshape(-1))
        self.assertEqual(image.tobytes(), actual.tobytes())
