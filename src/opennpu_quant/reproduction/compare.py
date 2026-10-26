"""Independent raw, prediction, metric, repeat and environment verdicts."""

from pathlib import Path
from collections import Counter
from .._io import read_json
from .assets import unseal, checked_file, seal
from .execution import load_arrays, verify_complete, raw_digest
from .checks import arrays, structured
from .environment import compare as compare_environment


def _pair(actual, expected, actual_dir, expected_dir):
    result = dict(
        id=actual["id"],
        model=actual["model"],
        condition=actual["condition"],
        state="pending",
        raw="pending",
        predictions="pending",
        metrics="pending",
        bitwise_alerts=[],
        prediction_mismatches=[],
        raw_failures=[],
    )
    if actual["prepared_identity"] != expected["prepared_identity"]:
        return dict(
            result,
            state="fail",
            reason="Model/input/profile/selection identity differs",
        )
    if actual["output_names"] != expected["output_names"]:
        return dict(result, state="fail", reason="Output names/order differ")
    left, right = actual["samples"], expected["samples"]
    if [s["id"] for s in left] != [s["id"] for s in right] or not left:
        return dict(result, state="fail", reason="Sample count/order differs")
    raw_pass = predictions_pass = True
    raw_pending = False
    for a, b in zip(left, right):
        if a["input_digest"] != b["input_digest"]:
            return dict(result, state="fail", reason="Input tensor identity differs")
        if a.get("raw") and b.get("raw"):
            av = load_arrays(checked_file(actual_dir, a["raw"]))
            bv = load_arrays(checked_file(expected_dir, b["raw"]))
            if set(av) != set(bv) or set(av) != set(actual["output_names"]):
                return dict(result, state="fail", reason="Stored output keys differ")
            for name in actual["output_names"]:
                check = arrays(av[name], bv[name])
                detail = dict(sample=a["id"], tensor=name, **check)
                raw_pass &= check["numeric_passed"]
                if not check["numeric_passed"]:
                    result["raw_failures"].append(detail)
                elif not check["bitwise_equal"]:
                    result["bitwise_alerts"].append(detail)
        else:
            ad = (
                raw_digest(load_arrays(checked_file(actual_dir, a["raw"])))
                if a.get("raw")
                else a["raw_digest"]
            )
            bd = (
                raw_digest(load_arrays(checked_file(expected_dir, b["raw"])))
                if b.get("raw")
                else b["raw_digest"]
            )
            if set(ad) != set(bd) or set(ad) != set(actual["output_names"]):
                return dict(result, state="fail", reason="Stored output keys differ")
            for name in ad:
                if any(ad[name][key] != bd[name][key] for key in ("shape", "dtype")):
                    return dict(
                        result, state="fail", reason="Output shape/dtype mismatch"
                    )
            if ad != bd:
                raw_pending = True
                result.setdefault("missing_raw_samples", []).append(a["id"])
        p = structured(
            read_json(checked_file(actual_dir, a["prediction"])),
            read_json(checked_file(expected_dir, b["prediction"])),
        )
        predictions_pass &= p
        if not p:
            result["prediction_mismatches"].append(a["id"])
    metrics_pass = actual["metric_units"] == expected["metric_units"] and structured(
        actual["metrics"], expected["metrics"]
    )
    result.update(
        raw="fail" if not raw_pass else "pending" if raw_pending else "pass",
        predictions="pass" if predictions_pass else "fail",
        metrics="pass" if metrics_pass else "fail",
        actual_metrics=actual["metrics"],
        expected_metrics=expected["metrics"],
        repeats="pass"
        if actual["repeat_bitwise"] and expected["repeat_bitwise"]
        else "fail",
        environment=compare_environment(actual["environment"], expected["environment"]),
    )
    result["state"] = (
        "pass"
        if all(
            result[k] == "pass" for k in ("raw", "predictions", "metrics", "repeats")
        )
        else "fail"
    )
    if (
        raw_pending
        and raw_pass
        and predictions_pass
        and metrics_pass
        and result["repeats"] == "pass"
    ):
        result.update(
            state="pending",
            reason="Raw hashes differ; retain tensors to assess numerical tolerance",
        )
    if result["environment"]["state"] == "fail":
        result["state"] = "fail"
    elif result["state"] == "pass" and result["environment"]["state"] == "pending":
        result["state"] = "pending"
    if actual["environment"]["device"].startswith("cuda") != expected["environment"][
        "device"
    ].startswith("cuda"):
        result.update(
            state="pending",
            reason="CPU/GPU comparison is diagnostic, outside GPU acceptance",
        )
    left_uuid = (actual["environment"].get("device_identity") or {}).get("uuid")
    right_uuid = (expected["environment"].get("device_identity") or {}).get("uuid")
    distinct = bool(left_uuid and right_uuid and left_uuid != right_uuid)
    result["scope"] = (
        "cross_gpu"
        if result["environment"]["gpu_comparison"] and distinct
        else "same_device_or_unverified_gpu"
    )
    result["gpu_comparison"] = (
        result["state"] if result["scope"] == "cross_gpu" else "pending"
    )
    return result


def compare(*, actual, expected, output, models=None, conditions=None):
    actual, expected, output = Path(actual), Path(expected), Path(output)
    if output.exists():
        raise FileExistsError(output)
    a, b = unseal(actual / "run.json"), unseal(expected / "run.json")
    left = {r["id"]: r for r in a["conditions"]}
    right = {r["id"]: r for r in b["conditions"]}
    from ..data.lfw_protocol import reject_partial_result

    for root, run in ((actual, a), (expected, b)):
        for state in run["conditions"]:
            model, condition = state["id"].split("__", 1)
            if (models and model not in models) or (
                conditions and condition not in conditions
            ):
                continue
            reject_partial_result(state)
            if not state.get("result"):
                from ..models.spec import ModelSpec, model_names
                from ..data.lfw_protocol import require_full

                if (
                    model in model_names()
                    and ModelSpec.load(model).recipe["task"] == "face_verification"
                ):
                    require_full(limit=run.get("limit"))
                continue
            try:
                result = unseal(checked_file(root, state["result"]))
            except (ValueError, OSError, KeyError):
                # The ordinary comparison records corrupt artifacts as failed conditions.
                continue
            reject_partial_result(result)
    rows = []
    for key in sorted(left.keys() | right.keys()):
        model, condition = key.split("__", 1)
        if (models and model not in models) or (
            conditions and condition not in conditions
        ):
            continue
        row = dict(
            id=key,
            model=model,
            condition=condition,
            state="pending",
            reason="Missing completed reference or candidate",
        )
        if (
            key in left
            and key in right
            and all(r.get("state") == "completed" for r in (left[key], right[key]))
        ):
            try:
                ap, bp = (
                    checked_file(actual, left[key]["result"]),
                    checked_file(expected, right[key]["result"]),
                )
                av, bv = verify_complete(ap.parent), verify_complete(bp.parent)
                row = _pair(av, bv, ap.parent, bp.parent)
            except (ValueError, OSError, KeyError) as error:
                row.update(state="fail", reason=str(error))
        if any(side.get(key, {}).get("state") == "fail" for side in (left, right)):
            row.update(state="fail", reason="Candidate or reference execution failed")
        rows.append(row)
    if not rows:
        raise ValueError("No matching comparisons")
    alerts = [
        dict(model=r["model"], condition=r["condition"], **alert)
        for r in rows
        for alert in r.get("bitwise_alerts", [])
    ]
    result = dict(
        schema_version=1,
        conditions=rows,
        counts=dict(Counter(r["state"] for r in rows)),
        within_tolerance_nonbitwise=alerts,
        gpu_comparison_counts=dict(
            Counter(r.get("gpu_comparison", "pending") for r in rows)
        ),
        retraining="not_run",
    )
    output.mkdir(parents=True)
    seal(output / "comparison.json", result)
    lines = [
        "| Model | Condition | Raw | Prediction | Metric | Overall | Cross-GPU | Non-bitwise tensors |",
        "|---|---|---|---|---|---|---|---:|",
    ]
    lines.extend(
        f"| {r['model']} | {r['condition']} | {r.get('raw', 'pending')} | {r.get('predictions', 'pending')} | {r.get('metrics', 'pending')} | {r['state']} | {r.get('gpu_comparison', 'pending')} | {len(r.get('bitwise_alerts', []))} |"
        for r in rows
    )
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    return dict(
        output=str(output),
        counts=result["counts"],
        within_tolerance_nonbitwise=len(alerts),
        gpu_comparison_counts=result["gpu_comparison_counts"],
    )
