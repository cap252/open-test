"""Rebuild a run folder's summary and tables from its verified result files.

Only results whose hashes match their complete.json (and the recorded run status) are
reported as measured; missing or failed measurements are never treated as zeros.
"""

from pathlib import Path
from ._io import read_json, sha256, object_hash
from .results import (
    export_records,
    metric_values,
    normalize_scheme,
    normalize_formats,
    report_policy,
    percent,
    summarize,
)

# Files that complete.json may attest next to result.json (model.onnx has its own hash).
ATTESTED_FILES = ("quantization.json", "predictions.jsonl.gz")


def verify_artifacts(directory, proof):
    """True when result.json, model.onnx and the extra files match complete.json."""
    directory = Path(directory)
    if not isinstance(proof, dict):
        return False
    result = directory / "result.json"
    if not result.is_file() or sha256(result) != proof.get("result_sha256"):
        return False
    files = proof.get("files", {})
    if not isinstance(files, dict) or set(files) - set(ATTESTED_FILES):
        return False
    for name, digest in files.items():
        if not (directory / name).is_file() or sha256(directory / name) != digest:
            return False
    model = directory / "model.onnx"
    return not proof.get("model_sha256") or (
        model.is_file() and sha256(model) == proof["model_sha256"]
    )


def report(directory, *, formats=None, table_profile=None, _refresh_existing=False):
    """Compare matched FP32 denominators without treating missing results as zero."""
    selected_formats = normalize_formats(formats)
    directory = Path(directory)
    if not (directory / "effective_config.json").is_file():
        raise ValueError(
            f"Not a run directory: {directory} (effective_config.json is missing). "
            "Pass <output>/<experiment name>, e.g. workspace/runs/core10"
        )
    config = read_json(directory / "effective_config.json")
    policy = report_policy(config.get("report"))
    status_path = directory / "status.json"
    status = read_json(status_path) if status_path.is_file() else {}
    records = [
        _record(directory, config, status, policy, model, condition)
        for model in config["models"]
        for condition in config["conditions"]
    ]
    result = summarize(records, policy=policy)
    export_records(
        records,
        directory,
        formats=selected_formats,
        table_profile=table_profile,
        metadata={
            "kind": "standalone_run",
            "config_identity": object_hash(config),
            "report_policy": policy,
        },
        summary=result,
        _refresh_existing=_refresh_existing,
    )
    if table_profile:
        from .raw_report import build_report

        result["raw_data"] = build_report(records, profile=table_profile)
    return result


def _condition_status(status, model, condition):
    return (
        status.get("models", {}).get(model, {}).get("conditions", {}).get(condition, {})
    )


def _verified(path, config, status):
    """True when result.json is the attested output of the recorded run."""
    proof_path = path.parent / "complete.json"
    if not path.is_file() or not proof_path.is_file():
        return False
    try:
        proof = read_json(proof_path)
        if not verify_artifacts(path.parent, proof):
            return False
        measured = read_json(path)
    except (ValueError, FileNotFoundError, IsADirectoryError):
        # Corrupt evidence must not prevent stale successful reports being replaced.
        return False
    if not isinstance(measured, dict):
        return False
    if measured.get("config_identity") != object_hash(config):
        return False
    if status.get("attempt_id"):
        current = _condition_status(status, path.parent.parent.name, path.parent.name)
        if (
            current.get("state") not in {"completed", "reused"}
            or current.get("result_sha256") != proof.get("result_sha256")
            or current.get("identity") != measured.get("identity")
            or proof.get("identity") != measured.get("identity")
            or current.get("measurement_attempt_id") != measured.get("attempt_id")
        ):
            return False
    return True


def _record(directory, config, status, policy, model, condition):
    """One portable record per model/condition; unverified results carry no measurement."""
    path = directory / model / condition / "result.json"
    ok = _verified(path, config, status)
    current = _condition_status(status, model, condition)
    row = read_json(path) if ok else {}
    from .data.lfw_protocol import reject_partial_result

    reject_partial_result(row)
    reject_partial_result(current)
    metadata = row.get("model_metadata", {})
    binding = "recorded_at_measurement" if metadata else "not_recorded"
    if not metadata:
        # A reference recipe is not measured provenance.
        try:
            from .models.spec import ModelSpec

            recipe = ModelSpec.load(model).recipe
            metadata = {
                k: recipe[k] for k in ("source", "dataset", "task", "input", "metric")
            }
            binding = "packaged_recipe_reference"
        except (FileNotFoundError, ValueError):
            pass
    condition_policy = config["conditions"][condition]
    scheme = row.get("scheme")
    if not ok and condition_policy:
        scheme = dict(config["schemes"][condition_policy["scheme"]])
        if scheme.get("percentile") == "per_model":
            scheme["percentile"] = config["percentiles"][model][
                condition_policy["scheme"]
            ]
    adaround = (row.get("adaround") or {}).get("config")
    if adaround is None and condition_policy.get("adaround"):
        adaround = config["adaround"]
    task = metadata.get("task")
    if task == "face_verification":
        from .data.lfw_protocol import require_full

        require_full(limit=config.get("evaluation", {}).get("limit"))
    if ok and task is None and policy["primary_metric"]:
        raise ValueError("Report primary_metric requires model task metadata: " + model)
    primary = policy["primary_metric"].get(
        task, row.get("metric", metadata.get("metric", "unknown"))
    )
    metrics = metric_values(row.get("metrics", {}), row.get("metric_units"))
    if ok and (primary not in metrics or percent(metrics[primary]) is None):
        raise ValueError(
            "Missing or incompatible report metric: " + model + "/" + primary
        )
    if ok:
        state = "completed"
    elif current.get("state") in {"completed", "reused"}:
        state = "unverified"
    else:
        state = current.get("state", "unverified" if path.exists() else "pending")
    return dict(
        run_id=config["name"],
        model=model,
        model_name=row.get("title", model),
        condition=condition,
        state=state,
        origin="standalone_run",
        primary_metric=primary,
        metrics=metrics,
        metric_details=row.get("metric_details", {}),
        functional_passed=row.get("functional_passed", False),
        export_validation=row.get("export_validation", {}),
        full_evaluation=row.get("full_evaluation", False),
        dataset=metadata.get("dataset"),
        split="validation",
        samples=row.get("samples"),
        expected_samples=row.get("expected_samples"),
        complete=row.get("complete", False),
        reference=row.get("reference", False),
        evaluation_identity=row.get("evaluation_identity"),
        quantization=normalize_scheme(scheme, adaround),
        model_source=metadata.get("source", {}),
        model_metadata=metadata,
        runtime=row.get("runtime", {}),
        finished_at=row.get("finished_at"),
        elapsed_s=row.get("elapsed_s"),
        evidence=dict(
            result_sha256=sha256(path) if ok else None,
            result_hash_verified=ok,
            source_binding=binding,
            model_identity=row.get("model_identity"),
            preprocessing_identity=row.get("preprocessing_identity"),
            identity=row.get("identity"),
            attempt_id=status.get("attempt_id"),
            measurement_attempt_id=row.get("attempt_id"),
            current_attempt=current,
        ),
    )
