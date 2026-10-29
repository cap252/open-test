import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from opennpu_quant._io import sha256

TOOLS = Path(__file__).resolve().parents[1] / "tools"
with patch.object(sys, "path", [str(TOOLS), *sys.path]):
    spec = importlib.util.spec_from_file_location(
        "frozen_check", TOOLS / "check_reproduction.py"
    )
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)


class ReproductionCheckTests(unittest.TestCase):
    def test_cosine_cannot_override_elementwise_failure(self):
        out = tool.arrays(np.ones(5, np.float32) * 1.01, np.ones(5, np.float32))
        self.assertAlmostEqual(1, out["cosine_similarity"])
        self.assertFalse(out["numeric_passed"])
        self.assertEqual(5, out["outside_tolerance"])
        for value in (
            np.ones(4, np.float32),
            np.ones(5, np.float64),
            np.full(5, np.nan, np.float32),
        ):
            with self.assertRaises(ValueError):
                tool.arrays(value, np.ones(5, np.float32))

    def test_large_integer_differences_are_not_lost(self):
        out = tool.arrays(np.array([2**53 + 1], np.int64), np.array([2**53], np.int64))
        self.assertFalse(out["bitwise_equal"])
        self.assertEqual(1, out["differing_elements"])
        self.assertEqual(1, out["outside_tolerance"])
        self.assertFalse(out["numeric_passed"])

    def test_prediction_order_is_distinct_from_set(self):
        a = [
            dict(category_id=1, bbox=[0, 0, 1, 1], score=0.8),
            dict(category_id=2, bbox=[5, 5, 1, 1], score=0.7),
        ]
        out = tool.predictions(a[::-1], a)
        self.assertTrue(out["same_exact_set"])
        self.assertFalse(out["ordered_exact"])
        self.assertFalse(out["passed"])
        self.assertFalse(tool.predictions([], a)["passed"])

    def test_changed_input_and_unsafe_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            p = root / "feeds.npz"
            np.savez(p, images=np.ones((1, 3, 5, 7), np.float32))
            digest = sha256(p)
            self.assertEqual(p, tool.checked(root, "feeds.npz", digest))
            np.savez(p, images=np.zeros((1, 3, 5, 7), np.float32))
            with self.assertRaisesRegex(ValueError, "changed"):
                tool.checked(root, "feeds.npz", digest)
            with self.assertRaisesRegex(ValueError, "unsafe"):
                tool.checked(root, "../feeds.npz", digest)
