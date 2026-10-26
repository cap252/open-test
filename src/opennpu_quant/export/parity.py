"""Framework-to-raw-ONNX diagnostics, independent of graph integrity checks."""

import numpy as np

ATOL = 1e-5
RTOL = 1e-4


def _valid_outputs(expected, actual):
    if not expected or set(expected) != set(actual):
        raise ValueError("Framework output names changed or are empty")
    for name, value in expected.items():
        other = actual[name]
        if value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError("Framework output shape/dtype changed: " + name)
        if value.dtype.kind not in "buif":
            raise ValueError("Unsupported framework output dtype: " + name)
        if not np.isfinite(value).all() or not np.isfinite(other).all():
            raise ValueError("Nonfinite framework output: " + name)


def _finite(value):
    return float(value) if np.isfinite(value) else None


def _statistics(a, b, atol=ATOL, rtol=RTOL, *, absolute_bound=None):
    # Keep numpy's original comparison direction and dtype rounding. Metrics use
    # float64, but neither inputs nor the allclose decision are converted.
    close = np.isclose(a, b, atol=atol, rtol=rtol)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
        relative = np.divide(delta, np.abs(b), out=np.zeros_like(delta), where=b != 0)
        maximum = delta.max(initial=0)
        rmse = np.sqrt(np.mean(delta * delta)) if a.size else 0.0
    failed = int(np.count_nonzero(~close))
    bound_failed = (
        int(np.count_nonzero(delta > absolute_bound))
        if absolute_bound is not None
        else 0
    )
    return dict(
        shape=list(a.shape),
        dtype=str(a.dtype),
        elements=int(a.size),
        bitwise_equal=a.tobytes() == b.tobytes(),
        max_abs=_finite(maximum),
        rmse=_finite(rmse),
        max_relative_nonzero=_finite(relative.max(initial=0)),
        zero_reference_mismatches=int(np.count_nonzero((b == 0) & (a != b))),
        tolerance_reference="onnx",
        atol=atol,
        rtol=rtol,
        above_tolerance=failed,
        above_tolerance_fraction=failed / a.size if a.size else 0.0,
        absolute_bound=absolute_bound,
        above_absolute_bound=bound_failed,
        passed=failed == 0 and bound_failed == 0,
    )


def _quality(expected, actual, recipe):
    """Decisions available in raw outputs only, never a dataset accuracy verdict."""
    task = recipe["task"]
    policy = recipe["export"].get("framework_parity", {})
    checks = {}
    if task in ("classification", "segmentation"):
        if set(expected) != {"logits"}:
            raise ValueError("Framework quality check requires logits")
        a, b = expected["logits"], actual["logits"]
        if task == "classification":
            if a.ndim != 2 or a.shape[-1] < 5 or not a.shape[0]:
                raise ValueError("Classification parity requires at least five logits")
            different = np.any(
                np.argsort(a, axis=-1)[..., -5:] != np.argsort(b, axis=-1)[..., -5:],
                axis=-1,
            )
            checks["top5_order"] = dict(
                different=int(different.sum()),
                total=int(different.size),
                passed=not bool(different.any()),
            )
        else:
            if a.ndim != 4 or not all(a.shape):
                raise ValueError("Segmentation parity requires NCHW logits")
            different = a.argmax(1) != b.argmax(1)
            fraction = float(different.mean())
            limit = policy.get("label_disagreement_fraction_limit", 0.0)
            checks["raw_argmax"] = dict(
                different=int(different.sum()),
                total=int(different.size),
                fraction=fraction,
                limit=limit,
                passed=fraction <= limit,
            )
    if recipe["export"]["family"] == "tf_ssd":
        for name in ("num_detections", "detection_classes", "detection_anchor_indices"):
            if name not in expected:
                raise ValueError("SSD diagnostic output missing: " + name)
            different = expected[name] != actual[name]
            checks[name] = dict(
                different=int(different.sum()),
                total=int(different.size),
                passed=not bool(different.any()),
            )
    return dict(
        shape_dtype_finite_passed=True,
        scope="raw_outputs_only",
        decisions=checks,
        decisions_passed=all(c["passed"] for c in checks.values()) if checks else None,
        accuracy="not_evaluated",
    )


def framework_compare(expected, actual, recipe):
    """Record finite differences without rejecting export, including in strict mode."""
    _valid_outputs(expected, actual)
    common = {name: _statistics(a, actual[name]) for name, a in expected.items()}
    policy = recipe["export"].get("framework_parity", {})
    family = recipe["export"]["family"]
    raw_names = list(expected)
    if family == "tf_ssd":
        raw_names = ["raw_detection_boxes", "raw_detection_scores"]
        if any(name not in expected for name in raw_names):
            raise ValueError("SSD raw diagnostic outputs missing")
    effective = {name: common[name] for name in raw_names}
    exception = None
    if recipe["task"] in ("classification", "segmentation") and "logits" in expected:
        bound = policy.get("logits_absolute_bound", ATOL)
        if not common["logits"]["passed"]:
            effective["logits"] = _statistics(
                expected["logits"], actual["logits"], bound, absolute_bound=bound
            )
    elif recipe["task"] == "face":
        for name in expected:
            if not common[name]["passed"]:
                bound = policy.get("output_absolute_bounds", {}).get(name, ATOL)
                effective[name] = _statistics(
                    expected[name], actual[name], bound, absolute_bound=bound
                )
    elif family == "yolov5":
        a, b = expected["predictions"], actual["predictions"]
        if a.ndim != 3 or a.shape[-1] < 5:
            raise ValueError("YOLO parity requires decoded xywh and scores")
        coords = _statistics(a[..., :4], b[..., :4])
        if not coords["passed"]:
            bound = policy.get("decoded_xywh_absolute_bound", ATOL)
            coords = _statistics(a[..., :4], b[..., :4], bound, absolute_bound=bound)
        effective = dict(
            decoded_xywh=coords, scores=_statistics(a[..., 4:], b[..., 4:])
        )
    elif recipe.get("name") == "nanodet_plus_m_320" and "predictions" in expected:
        a, b = expected["predictions"], actual["predictions"]
        if set(expected) != {"predictions"} or a.ndim != 3 or a.shape[-1] != 112:
            raise ValueError("NanoDet parity requires 80 probabilities and 32 logits")
        exception = "nanodet_plus_m_320_raw_fp32_v1"
        effective = dict(
            class_probabilities=_statistics(a[..., :80], b[..., :80]),
            regression_logits=_statistics(a[..., 80:], b[..., 80:], 2e-4, 0.0),
        )
    quality = _quality(expected, actual, recipe)
    common_passed = all(c["passed"] for c in common.values())
    raw_passed = all(c["passed"] for c in effective.values())
    quality_passed = quality["decisions_passed"] is not False
    return dict(
        schema_version=2,
        state="agreement" if common_passed and quality_passed else "differences",
        strict_passed=common_passed and quality_passed,
        common_passed=common_passed,
        effective_passed=raw_passed and quality_passed,
        raw_numerical_passed=raw_passed,
        errors={name: c["max_abs"] for name, c in common.items()},
        outputs=common,
        raw_outputs=raw_names,
        output_policies=effective,
        policy=policy,
        output_exception=exception,
        quality=quality,
        numerical_action="record_only",
    )


def validate_framework(expected, actual, recipe, mode):
    if mode not in ("strict", "functional"):
        raise ValueError("Unknown export validation mode")
    result = framework_compare(expected, actual, recipe)
    if mode == "functional":
        result["functional_passed"] = True
    return result


def diagnostic_lines(model, validation):
    for index, check in enumerate(validation["checks"]):
        result = check["framework_raw"]
        if result["state"] != "differences":
            continue
        details = ", ".join(
            f"{name}: max_abs={value['max_abs']}, "
            f"above_tolerance={value['above_tolerance']}/{value['elements']}"
            for name, value in result["outputs"].items()
        )
        yield (
            f"Export diagnostic {model} fixture {index}: {details}. "
            f"Raw policy passed={result['raw_numerical_passed']}, "
            f"raw decision checks passed={result['quality']['decisions_passed']}. "
            "Export continued. Details: model.json validation.checks"
        )
