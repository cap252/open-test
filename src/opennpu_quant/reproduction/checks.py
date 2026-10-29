"""Fixed-tensor diagnostics adapted from the pinned research comparison tool."""

from collections import Counter
import json
import numpy as np


def arrays(actual, expected, atol=1e-5, rtol=1e-4):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("Output shape/dtype mismatch")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Nonfinite output")
    n = actual.size
    maximum = total = squares = dot = aa = bb = 0.0
    outside = differing = 0
    integer = actual.dtype.kind in "biu"
    a, b = actual.reshape(-1), expected.reshape(-1)
    for start in range(0, n, 262144):
        exact = a[start : start + 262144] != b[start : start + 262144]
        x = a[start : start + 262144].astype(np.float64)
        y = b[start : start + 262144].astype(np.float64)
        delta = np.abs(x - y)
        maximum = max(maximum, float(delta.max(initial=0)))
        total += float(delta.sum())
        squares += float(np.dot(delta, delta))
        # Integers above 2**53 can collide in float64, so count them exactly.
        outside += int(
            np.count_nonzero(exact if integer else delta > atol + rtol * np.abs(y))
        )
        differing += int(np.count_nonzero(exact))
        dot += float(np.dot(x, y))
        aa += float(np.dot(x, x))
        bb += float(np.dot(y, y))
    cosine = dot / np.sqrt(aa * bb) if aa and bb else (1.0 if aa == bb else 0.0)
    return dict(
        shape=list(actual.shape),
        dtype=str(actual.dtype),
        elements=n,
        bitwise_equal=actual.tobytes() == expected.tobytes(),
        differing_elements=differing,
        max_absolute_error=maximum,
        mean_absolute_error=total / max(n, 1),
        rmse=float(np.sqrt(squares / max(n, 1))),
        cosine_similarity=float(np.clip(cosine, -1, 1)),
        outside_tolerance=outside,
        numeric_passed=outside == 0,
        atol=0 if integer else atol,
        rtol=0 if integer else rtol,
    )


def predictions(actual, expected):
    result = dict(
        count_equal=len(actual) == len(expected),
        ordered_categories_equal=True,
        numeric_passed=True,
        masks_equal=True,
        max_box_error=0.0,
        max_score_error=0.0,
    )

    def key(row):
        return json.dumps(row, sort_keys=True, separators=(",", ":"))

    result["same_exact_set"] = Counter(map(key, actual)) == Counter(map(key, expected))
    result["ordered_exact"] = actual == expected
    if not result["count_equal"]:
        result.update(
            ordered_categories_equal=False, numeric_passed=False, masks_equal=False
        )
        result["passed"] = False
        return result
    for a, b in zip(actual, expected):
        if isinstance(a, dict):
            result["ordered_categories_equal"] &= a["category_id"] == b["category_id"]
            box_a, box_b, score_a, score_b = (
                a["bbox"],
                b["bbox"],
                a["score"],
                b["score"],
            )
            if "segmentation" in a or "segmentation" in b:
                from pycocotools import mask as api

                result["masks_equal"] &= np.array_equal(
                    api.decode(a["segmentation"]), api.decode(b["segmentation"])
                )
        else:
            box_a, box_b, score_a, score_b = a[:4], b[:4], a[4], b[4]
        box_a, box_b = np.asarray(box_a, np.float64), np.asarray(box_b, np.float64)
        result["max_box_error"] = max(
            result["max_box_error"], float(np.max(np.abs(box_a - box_b), initial=0))
        )
        result["max_score_error"] = max(
            result["max_score_error"], abs(float(score_a) - float(score_b))
        )
        result["numeric_passed"] &= bool(
            np.allclose(box_a, box_b, atol=1e-5, rtol=1e-4)
            and np.isclose(score_a, score_b, atol=1e-5, rtol=1e-4)
        )
    result["passed"] = all(
        result[k]
        for k in (
            "count_equal",
            "ordered_categories_equal",
            "numeric_passed",
            "masks_equal",
        )
    )
    return result


def structured(actual, expected, *, exact=False):
    """Ordered predictions: discrete identities/masks exact, floating values tolerant."""
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            return False
        if "counts" in actual and "size" in actual:
            from pycocotools import mask as api

            return bool(np.array_equal(api.decode(actual), api.decode(expected)))
        return all(
            structured(
                actual[k],
                expected[k],
                exact=exact
                or k
                in (
                    "category_id",
                    "image_id",
                    "top5",
                    "labels",
                    "label",
                    "flip",
                    "image_key",
                ),
            )
            for k in actual
        )
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(
            structured(a, b, exact=exact) for a, b in zip(actual, expected)
        )
    if isinstance(actual, float):
        return bool(
            np.isfinite(actual)
            and np.isfinite(expected)
            and (
                actual == expected
                if exact
                else np.isclose(actual, expected, atol=1e-5, rtol=1e-4)
            )
        )
    return actual == expected
