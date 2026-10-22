"""F03: custom code has no closed dependency boundary; never reuse its feeds."""

import sys
import tempfile
from pathlib import Path
from types import ModuleType
import unittest
from unittest.mock import patch
import numpy as np

from tests.test_result_reliability import fixture
from opennpu_quant.data.samples import Dataset
from opennpu_quant.models.bundle import ModelBundle
from opennpu_quant.models.spec import ModelSpec
from opennpu_quant._feeds import feed_digest


class PreprocessingReliabilityTests(unittest.TestCase):
    def test_f03_same_name_body_helper_and_global_changes_refuse_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths, _ = fixture(root)
            spec = ModelSpec.load(paths.models / "toy/recipe.yaml")
            spec.recipe["preprocess"] = dict(kind="custom_reliability:transform")
            bundle = ModelBundle.from_directory(spec, paths.models / "toy")
            data = Dataset(paths.data / "imagenet/manifest.json")
            module = ModuleType("custom_reliability")
            module.np = np
            exec(
                "offset = 0\ndef helper(): return offset\ndef transform(image, settings, layout):\n return np.full((1,3,4,4), helper(), np.float32), {}",
                module.__dict__,
            )
            with patch.dict(sys.modules, custom_reliability=module):
                for change in ("initial", "body", "helper", "global"):
                    with self.subTest(change=change):
                        if change == "body":
                            exec(
                                "def transform(image, settings, layout):\n return np.full((1,3,4,4), helper()+1, np.float32), {}",
                                module.__dict__,
                            )
                        elif change == "helper":
                            exec("def helper(): return offset + 2", module.__dict__)
                        elif change == "global":
                            module.offset = 4
                        feed = next(data.feeds(bundle, 2)())["images"]
                        self.assertEqual(
                            {"initial": 0, "body": 1, "helper": 3, "global": 7}[change],
                            float(feed.mean()),
                        )
                        with self.assertRaisesRegex(ValueError, "custom preprocessing"):
                            data.cached_feeds(
                                bundle, 2, root / "cache", max_bytes=100000
                            )
                self.assertFalse((root / "cache").exists())

    def test_builtin_cache_on_off_feeds_match_and_dependency_change_invalidates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths, _ = fixture(root)
            bundle = ModelBundle.from_directory(
                ModelSpec.load(paths.models / "toy/recipe.yaml"), paths.models / "toy"
            )
            data = Dataset(paths.data / "imagenet/manifest.json")
            direct = [feed_digest(f) for f in data.feeds(bundle, 2)()]
            cached = data.cached_feeds(bundle, 2, root / "cache", max_bytes=100000)
            self.assertEqual(direct, [feed_digest(f) for f in cached()])
            reused = data.cached_feeds(bundle, 2, root / "cache", max_bytes=100000)
            self.assertTrue(reused.reused)
            import cv2

            with patch.object(cv2, "__version__", "changed-build"):
                changed = data.cached_feeds(bundle, 2, root / "cache", max_bytes=100000)
                self.assertFalse(changed.reused)
                self.assertNotEqual(
                    cached.manifest["identity"], changed.manifest["identity"]
                )
                self.assertEqual(direct, [feed_digest(f) for f in changed()])

    def test_cache_replay_rejects_changed_settings_and_dependencies(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths, _ = fixture(root)
            bundle = ModelBundle.from_directory(
                ModelSpec.load(paths.models / "toy/recipe.yaml"), paths.models / "toy"
            )
            data = Dataset(paths.data / "imagenet/manifest.json")
            cached = data.cached_feeds(bundle, 2, root / "cache", max_bytes=100000)
            import cv2

            with patch.object(cv2, "__version__", "changed-build"):
                with self.assertRaisesRegex(ValueError, "dependencies changed"):
                    list(cached())
            bundle.spec.recipe["preprocess"]["mean"][0] += 1
            with self.assertRaisesRegex(ValueError, "settings/input mapping changed"):
                list(cached())

    def test_ort_code_change_invalidates_only_caches_with_a_cpu_graph(self):
        from opennpu_quant.data import samples

        real = samples.source_hashes

        def changed(*paths):
            return {**real(*paths), "ort/fetch.py": "changed"}

        for host in (None, "preprocess"):
            for shape in ([1, 3, 4, 4], [1, 3, "height", "width"]):
                with self.subTest(host=host, shape=shape):
                    with tempfile.TemporaryDirectory() as temp:
                        root = Path(temp)
                        paths, _ = fixture(root, host=host)
                        bundle = ModelBundle.from_directory(
                            ModelSpec.load(paths.models / "toy/recipe.yaml"),
                            paths.models / "toy",
                        )
                        bundle.spec.recipe["input"]["shape"] = shape
                        data = Dataset(paths.data / "imagenet/manifest.json")

                        def cached():
                            return data.cached_feeds(
                                bundle, 2, root / "cache", max_bytes=100000
                            )

                        first = cached()
                        kind = "Variable" if isinstance(shape[2], str) else ""
                        self.assertEqual(kind + "FeedCache", type(first).__name__)
                        direct = [feed_digest(f) for f in data.feeds(bundle, 2)()]
                        self.assertEqual(direct, [feed_digest(f) for f in first()])
                        with patch.object(samples, "source_hashes", changed):
                            again = cached()
                            if host:
                                self.assertFalse(again.reused)
                                self.assertNotEqual(
                                    first.manifest["identity"],
                                    again.manifest["identity"],
                                )
                                with self.assertRaisesRegex(
                                    ValueError, "runtime code changed"
                                ):
                                    list(first())
                            else:
                                self.assertTrue(again.reused)
                                self.assertEqual(
                                    first.manifest["identity"],
                                    again.manifest["identity"],
                                )
                                self.assertEqual(
                                    direct, [feed_digest(f) for f in first()]
                                )
