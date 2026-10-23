"""Local LFW cache construction is pinned to the selected reference bytes."""

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from opennpu_quant._io import atomic_json, read_json, sha256
from opennpu_quant._feeds import feed_digest
from opennpu_quant.data.feed_cache import prepare_feed_cache
from opennpu_quant.models.spec import ModelSpec
from opennpu_quant.reproduction.assets import Resolver, catalog
from opennpu_quant.reproduction.data import FrozenLFW
from opennpu_quant.reproduction.lfw import prepare_lfw
from tests.test_lfw import write_pairs


def fixture(root):
    """Three originals, six views and the full 6000-pair scoring protocol."""
    import cv2
    from opennpu_quant.data.lfw import align_mobilefacenet
    from opennpu_quant.models.extended_preprocess import preprocess

    source = root / "originals"
    source.mkdir()
    pairs = write_pairs(source / "pairs.txt")
    keys = sorted({p[k] for p in pairs for k in ("first", "second")})
    points = [[38.3, 73.5, 56.0, 41.5, 70.7, 51.7, 51.5, 71.7, 92.3, 92.2]]
    landmarks, rows, feeds = [], [], []
    recipe = ModelSpec.load("mobilefacenet_relu_foamliu").recipe
    for index, key in enumerate(keys):
        path = source / "data/lfw_funneled" / key
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = (np.arange(120 * 120 * 3).reshape(120, 120, 3) + index * 19).astype(
            np.uint8
        )
        Image.fromarray(pixels).save(path)
        landmarks.append(dict(full_path="lfw_funneled/" + key, landmarks=points))
        crop = align_mobilefacenet(cv2.imread(str(path)), points)
        for flip in (0, 1):
            # The recorded paths are deliberately unusable after relocation.
            rows.append(
                dict(
                    image_key=key,
                    image_id=key + f"|flip={flip}",
                    path="/unavailable/original/server/" + key,
                    sha256=sha256(path),
                    landmarks=points,
                    flip=flip,
                )
            )
            rgb = crop[..., ::-1]
            if flip:
                rgb = rgb[:, ::-1]
            tensor, _ = preprocess(Image.fromarray(rgb.copy()), recipe["preprocess"])
            feeds.append({"images": tensor})
    atomic_json(source / "lfw_funneled.json", dict(samples=landmarks))
    pins = {name: sha256(source / name) for name in ("pairs.txt", "lfw_funneled.json")}
    contract = dict(
        input_size=[112, 112],
        source_color="BGR",
        network_color="RGB",
        normalization={k: recipe["preprocess"][k] for k in ("divide", "mean", "std")},
        dataset_files=pins,
    )
    identity = dict(role="validation", contract=contract, rows=rows)
    samples = dict(
        pairs=pairs,
        pairs_sha256=pins["pairs.txt"],
        validation=rows,
        input_contract=contract,
        cache_identity=dict(validation=identity),
    )
    atomic_json(root / "samples.json", samples)
    cache = prepare_feed_cache(
        root / "reference",
        lambda: iter(feeds),
        identity=identity,
        verify_source=lambda: None,
        expected_samples=6,
        max_bytes=2**20,
    )
    manifest = cache.manifest
    reference = dict(
        schema_version=1,
        samples_sha256=sha256(root / "samples.json"),
        cache_sha256=sha256(root / "reference/manifest.json"),
        **{
            k: manifest[k]
            for k in ("sample_count", "tensors", "feed_digests", "samples_identity")
        },
    )
    descriptor = dict(
        kind="frozen_lfw",
        samples=dict(
            id="artifact://bundle/samples.json", sha256=reference["samples_sha256"]
        ),
        cache=dict(
            id="artifact://lfw-cache/manifest.json", sha256=reference["cache_sha256"]
        ),
    )
    atomic_json(
        root / "selection.json",
        dict(
            entries=[
                dict(
                    id="mobilefacenet_relu_foamliu__fp32",
                    model="mobilefacenet_relu_foamliu",
                    condition="fp32",
                    recipe=recipe,
                    dataset=descriptor,
                )
            ]
        ),
    )
    atomic_json(
        root / "bindings.json",
        {"bundle": str(root), "lfw-cache": str(root / "generated")},
    )
    return reference, pins


class LfwCacheRegenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference, self.pins = fixture(self.root)
        for target, value in (
            ("opennpu_quant.reproduction.lfw._reference", self.reference),
            ("opennpu_quant.data.lfw.PAIR_HASH", self.pins["pairs.txt"]),
            ("opennpu_quant.data.lfw.LANDMARK_HASH", self.pins["lfw_funneled.json"]),
        ):
            mock = (
                patch(target, return_value=value)
                if target.endswith("_reference")
                else patch(target, value)
            )
            mock.start()
            self.addCleanup(mock.stop)

    def generate(self, **kwargs):
        return prepare_lfw(
            selection=self.root / "selection.json",
            bindings=self.root / "bindings.json",
            root=self.root / "originals",
            output=self.root / "generated",
            **kwargs,
        )

    def test_exact_cache_reuse_and_frozen_evaluator_connection_without_inference(self):
        with patch(
            "onnxruntime.InferenceSession", side_effect=AssertionError("No inference")
        ):
            result = self.generate()
            self.assertFalse(result["reused"])
            for name in ("manifest.json", "tensor_0.npy"):
                self.assertEqual(
                    (self.root / "reference" / name).read_bytes(),
                    (self.root / "generated" / name).read_bytes(),
                )
            proof = read_json(self.root / "generated/verification.json")
            self.assertTrue(all(proof["checks"].values()))
            self.assertEqual(6000, proof["pair_count"])
            descriptor = read_json(self.root / "selection.json")["entries"][0][
                "dataset"
            ]
            dataset = FrozenLFW(
                descriptor, Resolver(self.root / "bindings.json"), "full"
            )
            samples = list(dataset.samples(None))
            self.assertEqual(6, len(samples))
            self.assertEqual(
                self.reference["feed_digests"], [feed_digest(s.feeds) for s in samples]
            )
            from opennpu_quant.evaluation.face_verification import verification

            embeddings = {
                k: {f: np.array([1, 0], np.float32) for f in (0, 1)}
                for k in {p["first"] for p in dataset.info["evaluation_pairs"]}
                | {p["second"] for p in dataset.info["evaluation_pairs"]}
            }
            metrics, details = verification(
                embeddings, dataset.info["evaluation_pairs"], dimensions=2
            )
            self.assertIn("verification_accuracy", metrics)
            self.assertEqual(10, len(details["folds"]))
            before = sha256(self.root / "generated/verification.json")
            with patch(
                "opennpu_quant.reproduction.lfw._feeds",
                side_effect=AssertionError("Do not rebuild"),
            ):
                self.assertTrue(self.generate()["reused"])
            self.assertEqual(before, sha256(self.root / "generated/verification.json"))

    def test_output_binding_mismatch_is_rejected_before_download(self):
        atomic_json(
            self.root / "bindings.json",
            {"bundle": str(self.root), "lfw-cache": str(self.root / "elsewhere")},
        )
        with patch("opennpu_quant.data.lfw.download_inputs") as download:
            with self.assertRaisesRegex(ValueError, "--output must match"):
                self.generate(download=True)
            download.assert_not_called()
        self.assertFalse((self.root / "generated").exists())

    def test_local_source_changes_are_rejected_without_publishing(self):
        for name in (
            "pairs.txt",
            "lfw_funneled.json",
            "data/lfw_funneled/A/A_0001.jpg",
        ):
            path = self.root / "originals" / name
            before = path.read_bytes()
            with self.subTest(name=name):
                path.write_bytes(before + b"changed")
                with self.assertRaisesRegex(ValueError, "source file changed"):
                    self.generate()
                self.assertFalse((self.root / "generated").exists())
                path.write_bytes(before)

    def test_missing_local_image_is_not_downloaded_implicitly(self):
        (self.root / "originals/data/lfw_funneled/A/A_0001.jpg").unlink()
        with patch("opennpu_quant.data.lfw.download_inputs") as download:
            with self.assertRaises(FileNotFoundError):
                self.generate()
            download.assert_not_called()

    def test_reference_tamper_is_detected_by_selected_manifest_hash(self):
        self.reference["feed_digests"][0] = "0" * 64
        with self.assertRaisesRegex(ValueError, "selected cache hash"):
            self.generate()
        self.assertFalse((self.root / "generated").exists())

    def test_reordered_pairs_and_missing_flip_cannot_be_authorized_by_samples_hash_only(
        self,
    ):
        original = read_json(self.root / "samples.json")
        for mutation in ("pairs", "views"):
            changed = copy.deepcopy(original)
            if mutation == "pairs":
                changed["pairs"].reverse()
            else:
                changed["validation"].pop()
            atomic_json(self.root / "samples.json", changed)
            selection = read_json(self.root / "selection.json")
            digest = sha256(self.root / "samples.json")
            selection["entries"][0]["dataset"]["samples"]["sha256"] = digest
            self.reference["samples_sha256"] = digest
            atomic_json(self.root / "selection.json", selection)
            with (
                self.subTest(mutation=mutation),
                self.assertRaisesRegex(ValueError, "order|views"),
            ):
                self.generate()
            self.assertFalse((self.root / "generated").exists())

    def test_input_mismatch_and_interruption_never_publish(self):
        from opennpu_quant.models.extended_preprocess import preprocess

        def wrong(image, cfg):
            tensor, meta = preprocess(image, cfg)
            tensor.flat[0] += np.float32(1)
            return tensor, meta

        for replacement in (wrong, KeyboardInterrupt()):
            with (
                self.subTest(error=type(replacement).__name__),
                patch(
                    "opennpu_quant.models.extended_preprocess.preprocess",
                    side_effect=replacement,
                ),
            ):
                with self.assertRaises((ValueError, KeyboardInterrupt)):
                    self.generate()
            self.assertFalse((self.root / "generated").exists())
            self.assertEqual([], list(self.root.glob("generated.building-*")))

    def test_corrupt_existing_cache_is_not_overwritten(self):
        self.generate()
        path = self.root / "generated/tensor_0.npy"
        path.chmod(0o644)
        with path.open("r+b") as stream:
            stream.seek(-4, 2)
            stream.write(b"oops")
        before = sha256(path)
        with self.assertRaisesRegex(ValueError, "payload changed"):
            self.generate()
        self.assertEqual(before, sha256(path))

    def test_download_is_explicit_and_local_source_root_can_move(self):
        moved = self.root / "moved"
        shutil.move(self.root / "originals", moved)
        with patch(
            "opennpu_quant.data.lfw.download_inputs",
            side_effect=lambda root: shutil.move(moved, root),
        ) as download:
            self.generate(download=True)
            download.assert_called_once_with(self.root / "originals")
        renamed = self.root / "cache-elsewhere"
        shutil.move(self.root / "generated", renamed)
        atomic_json(
            self.root / "bindings.json",
            {"bundle": str(self.root), "lfw-cache": str(renamed)},
        )
        result = prepare_lfw(
            selection=self.root / "selection.json",
            bindings=self.root / "bindings.json",
            root=self.root / "originals",
            output=renamed,
        )
        self.assertTrue(result["reused"])

    def test_cli_dispatch(self):
        from opennpu_quant.cli import build_parser
        from opennpu_quant.reproduction.cli import handle

        args = build_parser().parse_args(
            [
                "reproduce",
                "prepare-lfw",
                "--selection",
                str(self.root / "selection.json"),
                "--bindings",
                str(self.root / "bindings.json"),
                "--root",
                str(self.root / "originals"),
                "--output",
                str(self.root / "generated"),
            ]
        )
        self.assertEqual("verified", handle(args)["state"])


class PackagedLfwReferenceTests(unittest.TestCase):
    def test_packaged_reference_contains_no_images_landmarks_or_paths_and_covers_seven_conditions(
        self,
    ):
        from opennpu_quant.reproduction.lfw import _reference
        from importlib.resources import files

        reference = _reference()
        text = json.dumps(reference)
        self.assertNotIn("landmarks", text)
        self.assertNotIn("/", text)
        self.assertNotIn("pixels", text)
        self.assertEqual(15402, reference["sample_count"])
        self.assertEqual(15402, len(reference["feed_digests"]))
        entries = [
            e for e in catalog()["entries"] if e["dataset"].get("kind") == "frozen_lfw"
        ]
        self.assertEqual(7, len(entries))
        for e in entries:
            self.assertEqual(reference["cache_sha256"], e["dataset"]["cache"]["sha256"])
            self.assertEqual(
                reference["samples_sha256"], e["dataset"]["samples"]["sha256"]
            )
        self.assertLess(
            (files("opennpu_quant.resources.reproduction") / "lfw-input-reference.json")
            .stat()
            .st_size,
            1200000,
        )
