"""Assess new full evaluations against the hash-bound original measurements."""

import hashlib
import json
import numpy as np

from .._io import read_json
from .assets import checked_file
from .checks import arrays


def _metric_values(metrics, details=None):
    values, paths, conflicts = {}, {}, set()
    fields = [("metrics." + key, key, value) for key, value in metrics.items()]
    # Only this documented diagnostic is shared with the segmentation details.
    # Arbitrary details must not fill a missing metric or override a measured one.
    if details is not None and "pixel_accuracy" in details:
        fields.append(
            (
                "metric_details.pixel_accuracy",
                "pixel_accuracy",
                details["pixel_accuracy"],
            )
        )
    for path, key, value in fields:
        if (
            key in values
            and np.float64(values[key]).tobytes() != np.float64(value).tobytes()
        ):
            conflicts.add(key)
        else:
            values[key] = value
        paths.setdefault(key, []).append(path)
    return values, paths, sorted(conflicts)


def original_accuracy(result, source, folder):
    if result["stages"]["full_accuracy"] != "executed":
        return dict(
            state="pending",
            reason="Limited evaluation; original scores are full-dataset measurements",
        )
    expected = result["measurement"].get("metrics", {})
    if not expected:
        return dict(state="pending", reason="No original metrics")
    expected, expected_paths, expected_conflicts = _metric_values(expected)
    actual, actual_paths, actual_conflicts = _metric_values(
        result["metrics"], result.get("metric_details")
    )
    missing = sorted(set(expected) - actual.keys())
    checks = {
        key: dict(
            arrays(np.array([actual[key]], np.float64), np.array([value], np.float64)),
            actual_paths=actual_paths[key],
            expected_paths=["measurement." + path for path in expected_paths[key]],
        )
        for key, value in expected.items()
        if key in actual
    }
    complete = not (missing or actual_conflicts or expected_conflicts)
    answer = dict(
        state="fail"
        if not complete or not all(c["numeric_passed"] for c in checks.values())
        else "pass",
        metrics=checks,
        metrics_bitwise_equal=complete
        and all(c["bitwise_equal"] for c in checks.values()),
        missing_metrics=missing,
        conflicting_metrics=dict(actual=actual_conflicts, expected=expected_conflicts),
        prediction_stream="unavailable",
    )
    expected_samples = result["measurement"].get("samples")
    if expected_samples is not None and expected_samples != result["sample_count"]:
        answer.update(state="fail", reason="Original evaluation sample count differs")
    answer["metrics_state"] = answer["state"]
    evidence = source.get("prediction_evidence", {})
    expected_hash = evidence.get("all_predictions_sha256")
    if expected_hash:
        task = result.get("task")
        string_ids = result.get("model") in {
            "yolov9_s",
            "yolov9_t",
            "yolox",
            "yolov7_face",
            "yolov5s_seg",
        }
        if task not in {"detection", "keypoints", "instance_segmentation", "face"}:
            answer.update(
                prediction_stream="pending",
                reason="Original prediction serialization includes task-specific hashes or diagnostics; compare retained records separately",
            )
            if answer["state"] == "pass":
                answer["state"] = "pending"
            return answer
        digest = hashlib.sha256()
        for sample in result["samples"]:
            value = read_json(checked_file(folder, sample["prediction"]))
            image_id = sample["id"] if string_ids else int(sample["id"])
            record = dict(image_id=image_id)
            record["faces" if task == "face" else "detections"] = value
            digest.update(
                (
                    json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n"
                ).encode()
            )
        equal = digest.hexdigest() == expected_hash
        answer.update(
            prediction_stream="pass" if equal else "pending",
            actual_predictions_sha256=digest.hexdigest(),
            original_predictions_sha256=expected_hash,
        )
        if not equal:
            answer["reason"] = (
                "Prediction stream differs; the hash alone cannot establish numeric tolerance"
            )
            if answer["state"] == "pass":
                answer["state"] = "pending"
    return answer
