"""Metric definitions: Top-K ranking, WIDER FACE AP edge cases and COCO undefined values."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from opennpu_quant import EvaluationSample, TopKClassification
from opennpu_quant.evaluation.dataset_metrics import wider_face_ap


def sample(target=None, sample_id="1"):
    return EvaluationSample(sample_id, {}, target)


class TopKTests(unittest.TestCase):
    def test_integer_logits_preserve_exact_order_and_values(self):
        cases = [
            (np.uint8, [0, 1, 2, 3, 4, 5]),
            (np.int8, [-128, 1, 2, 3, 4, 5]),
            (np.int64, [-(2**63), -1, 0, 1, 2**63 - 2, 2**63 - 1]),
            (np.uint64, [0, 1, 2, 2**53, 2**64 - 2, 2**64 - 1]),
            (np.int64, [2**53 + i for i in range(1, 7)]),
            (np.uint64, [2**63 + i for i in range(1, 7)]),
        ]
        for dtype, values in cases:
            with self.subTest(dtype=dtype, values=values):
                logits = np.array([values], dtype=dtype)
                before = logits.copy()
                logits.flags.writeable = False
                metric = TopKClassification()
                metric.update({"logits": logits}, sample(target=5))
                self.assertEqual({"top1": 1.0, "top5": 1.0}, metric.finalize())
                self.assertEqual([5, 4, 3, 2, 1], metric.prediction["top5"])
                self.assertEqual(values[:0:-1], metric.prediction["top5_logits"])
                np.testing.assert_array_equal(before, logits)
                self.assertEqual(np.dtype(dtype), logits.dtype)

    def test_ties_at_top1_and_top5_keep_lower_class_indices(self):
        for dtype in (np.float32, np.int8, np.uint8):
            for values, target, expected in (
                ([1, 1, 1, 1, 1, 1], 0, {"top1": 1.0, "top5": 1.0}),
                ([1, 1, 1, 1, 1, 1], 5, {"top1": 0.0, "top5": 0.0}),
                ([4, 3, 2, 1, 1, 1], 4, {"top1": 0.0, "top5": 1.0}),
                ([4, 3, 2, 1, 1, 1], 5, {"top1": 0.0, "top5": 0.0}),
            ):
                with self.subTest(dtype=dtype, values=values, target=target):
                    logits = np.array([values], dtype=dtype)
                    before = logits.copy()
                    metric = TopKClassification()
                    metric.update({"logits": logits}, sample(target=target))
                    self.assertEqual(expected, metric.finalize())
                    self.assertEqual([0, 1, 2, 3, 4], metric.prediction["top5"])
                    self.assertEqual(values[:5], metric.prediction["top5_logits"])
                    np.testing.assert_array_equal(before, logits)

    def test_tied_logits_rank_the_lower_class_first(self):
        metric = TopKClassification(topk=(1,))
        logits = np.asarray([[1, 3, 3, 0, 0, 0]], np.float32)
        metric.update({"logits": logits}, sample(target=1))
        self.assertEqual({"top1": 1.0}, metric.finalize())
        self.assertEqual([1, 2, 0, 3, 4], metric.prediction["top5"])

    def test_topk_must_be_distinct_positive_integers(self):
        for topk in ((), (1, 1, 5), (0,), (1.5,), (True,)):
            with self.subTest(topk=topk):
                with self.assertRaisesRegex(ValueError, "distinct positive"):
                    TopKClassification(topk=topk)
        self.assertEqual((1, 5), TopKClassification(topk=iter((1, 5))).topk)


class WiderFaceApTests(unittest.TestCase):
    def truth(self, easy):
        return [
            dict(
                image_id=1,
                boxes=[[10, 10, 20, 20]],
                keep=dict(easy=easy, medium=[0], hard=[0]),
            )
        ]

    def test_setting_without_faces_is_omitted_instead_of_nan(self):
        detections = {1: [[10, 10, 20, 20, 0.9], [60, 60, 10, 10, 0.1]]}
        result = wider_face_ap(detections, self.truth(easy=[]))
        self.assertEqual({"AP_medium": 1.0, "AP_hard": 1.0}, result)

    def test_no_detections_score_zero(self):
        self.assertEqual(
            {"AP_easy": 0.0, "AP_medium": 0.0, "AP_hard": 0.0},
            wider_face_ap({1: []}, self.truth(easy=[0])),
        )


@unittest.skipUnless(
    importlib.util.find_spec("pycocotools"), "Optional COCO dependency"
)
class CocoTests(unittest.TestCase):
    def test_undefined_ranges_are_omitted_and_stdout_stays_clean(self):
        from opennpu_quant.evaluation.coco import CocoMetric

        annotations = dict(
            info={},
            images=[dict(id=1, width=200, height=200)],
            categories=[dict(id=1, name="thing")],
            annotations=[
                dict(
                    id=1,
                    image_id=1,
                    category_id=1,
                    bbox=[20, 20, 60, 60],
                    area=3600,
                    iscrowd=0,
                )
            ],
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "instances.json"
            path.write_text(json.dumps(annotations))
            metric = CocoMetric(path)
            metric.update(
                [dict(category_id=1, bbox=[20, 20, 60, 60], score=0.9)], sample()
            )
            with (
                contextlib.redirect_stdout(io.StringIO()) as out,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                metrics = metric.finalize()
        self.assertEqual("", out.getvalue())
        self.assertAlmostEqual(1.0, metrics["AP"])
        self.assertNotIn("AP_small", metrics)
        self.assertTrue(all(value >= 0 for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
