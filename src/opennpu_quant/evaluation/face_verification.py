"""LFW protocol: ordered ten-fold threshold selection and separate smoke scoring."""

from pathlib import Path
import numpy as np


def read_pairs(path):
    lines = Path(path).read_text().splitlines()
    if not lines or lines[0].split() != ["10", "300"] or len(lines) != 6001:
        raise ValueError("Expected official 6000-pair, ten-fold LFW protocol")

    def image(person, index):
        if (
            not person
            or person in (".", "..")
            or Path(person).name != person
            or not index.isdecimal()
            or int(index) < 1
        ):
            raise ValueError("Invalid LFW image identifier")
        return f"{person}/{person}_{int(index):04d}.jpg"

    rows = []
    for i, line in enumerate(lines[1:]):
        fields = line.split()
        same = len(fields) == 3
        if len(fields) not in (3, 4) or same != (i % 600 < 300):
            raise ValueError("LFW pair order/fold composition changed")
        first = image(fields[0], fields[1])
        second = image(
            fields[0] if same else fields[2], fields[2] if same else fields[3]
        )
        if first == second:
            raise ValueError("Self pair in LFW protocol")
        rows.append(
            dict(pair_id=i, fold=i // 600, first=first, second=second, same=same)
        )
    return rows


def _fixed_angle_verification(first, second, labels):
    # The reference normalizes rows in FP32, then compares stored angles in FP64.
    first = first / np.linalg.norm(first, axis=1, keepdims=True)
    second = second / np.linalg.norm(second, axis=1, keepdims=True)
    angles = np.degrees(np.arccos(np.clip((first * second).sum(axis=1), -1.0, 1.0)))
    correct = int(np.count_nonzero((angles.astype(np.float64) <= 74.18) == labels))
    # Preserve the reference's percentage-to-fraction roundtrip, including its ULP.
    accuracy = correct / len(labels) * 100.0 / 100.0
    return accuracy, angles


def verification(embeddings, pairs, *, dimensions, smoke=False):
    required = {p[k] for p in pairs for k in ("first", "second")}
    if not pairs or set(embeddings) != required:
        raise ValueError(
            "Missing/extra LFW embeddings; limited runs need a --smoke manifest"
        )
    combined, summed = {}, {}
    for key, views in embeddings.items():
        if set(views) != {0, 1}:
            raise ValueError("Expected original and horizontal-flip views")
        a, b = (np.asarray(views[i]) for i in (0, 1))
        if (
            a.shape != (dimensions,)
            or b.shape != a.shape
            or a.dtype != np.float32
            or b.dtype != np.float32
        ):
            raise ValueError("Invalid recognition embedding shape/dtype")
        value = a + b
        norm = np.linalg.norm(value)
        if not np.isfinite(value).all() or not np.isfinite(norm) or norm <= 0:
            raise ValueError("Nonfinite or zero face embedding")
        summed[key] = value
        combined[key] = value / norm
    first = np.stack([combined[p["first"]] for p in pairs])
    second = np.stack([combined[p["second"]] for p in pairs])
    distance = np.sum(np.square(first - second), axis=1)
    labels = np.asarray([p["same"] for p in pairs], dtype=bool)
    details = dict(
        pair_count=len(pairs),
        full_protocol=not smoke,
        flip="original + horizontal flip before FP32 L2 normalization",
    )
    if smoke:
        # No threshold fitting on two pairs and no claim of ten-fold accuracy.
        details.update(
            protocol="functional subset; fixed squared-L2 threshold 1.0", threshold=1.0
        )
        return {
            "smoke_pair_accuracy": float(np.mean((distance < 1.0) == labels))
        }, details
    if len(pairs) != 6000 or [p["pair_id"] for p in pairs] != list(range(6000)):
        raise ValueError("Incomplete/reordered LFW protocol")
    folds = np.asarray([p["fold"] for p in pairs])
    if not np.array_equal(folds, np.repeat(np.arange(10), 600)):
        raise ValueError("Changed LFW fold membership")
    thresholds = np.arange(0, 4, 0.01)
    correctness = (distance[None, :] < thresholds[:, None]) == labels[None, :]
    results = []
    for fold in range(10):
        test = folds == fold
        if labels[test].sum() != 300:
            raise ValueError("Invalid LFW fold composition")
        best = int(np.argmax(np.mean(correctness[:, ~test], axis=1)))
        results.append(
            dict(
                fold=fold,
                threshold=float(thresholds[best]),
                accuracy=float(np.mean(correctness[best, test])),
            )
        )
    scores = np.asarray([r["accuracy"] for r in results])
    if dimensions == 128:
        angle_accuracy, _ = _fixed_angle_verification(
            np.stack([summed[p["first"]] for p in pairs]),
            np.stack([summed[p["second"]] for p in pairs]),
            labels,
        )
    else:
        angle_accuracy = None
    details.update(
        folds=results, threshold_grid="0 <= t < 4, step 0.01; distance < threshold"
    )
    metrics = dict(
        verification_accuracy=float(scores.mean()), verification_std=float(scores.std())
    )
    if angle_accuracy is not None:
        metrics["fixed_angle_accuracy"] = angle_accuracy
        details["fixed_angle_threshold_degrees"] = 74.18
    return metrics, details


class FaceVerificationEvaluator:
    def __init__(self, bundle, dataset):
        self.bundle, self.dataset = bundle, dataset
        self.cfg = bundle.spec.recipe["decode"]
        self.reset()

    def reset(self):
        self.embeddings = {}
        self.prediction = None
        self.metric_details = {}

    def update(self, outputs, sample):
        row = sample.metadata["dataset_row"]
        key, flip = row["image_key"], row["flip"]
        views = self.embeddings.setdefault(key, {})
        if flip in views:
            raise ValueError("Duplicate recognition image view")
        value = outputs[self.cfg["output"]]
        if value.shape != (1, self.cfg["dimensions"]):
            raise ValueError("Unexpected recognition batch/output")
        views[flip] = value[0].copy()
        self.prediction = {
            "image_key": key,
            "flip": flip,
            "embedding": value[0].tolist(),
        }

    def finalize(self):
        metrics, self.metric_details = verification(
            self.embeddings,
            self.dataset.info["evaluation_pairs"],
            dimensions=self.cfg["dimensions"],
            smoke=self.dataset.info.get("smoke", False),
        )
        return metrics
