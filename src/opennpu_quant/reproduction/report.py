"""Read sealed replay evaluations without loading models or numerical runtimes."""

from pathlib import Path

from .._io import object_hash, sha256
from ..results import export_records, metric_values, normalize_scheme, normalize_formats
from ..raw_report import build_report, validate_profile
from .assets import checked_file, selected, unseal


def _quantization(entry):
    condition = entry["condition"]
    if condition == "fp32":
        return None
    family, rounding = condition.rsplit("_", 1)
    if family not in {"asym_pc_float", "asym_pc_pot", "sym_pt_pot"} or rounding not in {
        "rtn",
        "adaround",
    }:
        raise ValueError("Unknown selected quantization condition: " + condition)
    if rounding == "adaround" and not entry.get("adaround"):
        raise ValueError("Missing selected AdaRound settings: " + entry["id"])
    scale = "float" if family.endswith("float") else "pot_ceil"
    return normalize_scheme(
        dict(
            scope="all",
            activation_symmetric=family == "sym_pt_pot",
            weight_granularity="per_tensor"
            if family == "sym_pt_pot"
            else "per_channel",
            method="percentile",
            percentile=entry["percentile"],
            activation_scale=scale,
            weight_scale=scale,
        ),
        entry.get("adaround") if rounding == "adaround" else None,
    )


def _record(row, state, prepared, saved, run, run_id):
    entry = row["entry"]
    recipe = entry["recipe"]
    failure = state.get("error") or state.get("missing") or "not_executed"
    record = dict(
        run_id=run_id,
        model=entry["model"],
        model_name=recipe["title"],
        condition=entry["condition"],
        state=state.get("state", "pending"),
        origin="fixed_replay",
        task=recipe["task"],
        dataset=recipe["dataset"],
        split="validation",
        primary_metric=recipe["metric"],
        metrics={},
        quantization=_quantization(entry),
        model_source=recipe.get("source", {}),
        complete=False,
        full_evaluation=False,
        reference=False,
        failure_reason=failure
        if isinstance(failure, str)
        else "state:" + str(state.get("state", "pending")),
        evidence=dict(
            result_hash_verified=False,
            source_result=entry["source_result"],
            recorded_condition=entry.get("recorded_condition"),
        ),
    )
    diagnostic = state.get("state") == "fail" and bool(state.get("result"))
    if state.get("state") != "completed" and not diagnostic:
        details = state.get("missing") or (
            failure if not isinstance(failure, str) else None
        )
        if details is not None:
            record["failure_details"] = details
        return record
    if not state.get("result"):
        raise ValueError("Completed condition has no result: " + entry["id"])
    path = checked_file(run, state["result"])
    result = unseal(path)
    from ..data.lfw_protocol import reject_partial_result

    reject_partial_result(result)
    limit = row.get("limit", prepared["limit"])
    expected = dict(
        id=entry["id"],
        model=entry["model"],
        condition=entry["condition"],
        state=state["state"],
        prepared_identity=row.get("identity"),
        recipe_identity=object_hash(recipe),
        input_identity=row.get("data_identity"),
        source_graphs=entry["graphs"],
        profile=recipe["decode"].get("profile", "current_v1"),
        percentile=entry.get("percentile"),
        limit=limit,
    )
    if row.get("state") != "ready" or row.get("identity") != object_hash(
        [entry, row.get("data_identity"), limit]
    ):
        raise ValueError("Prepared identity mismatch: " + entry["id"])
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError("Replay identity mismatch: " + entry["id"] + "/" + key)
    if not result.get("identity") or result.get("environment", {}).get(
        "implementation"
    ) != saved.get("implementation"):
        raise ValueError("Replay implementation identity mismatch: " + entry["id"])
    if diagnostic and result.get("repeat_bitwise") is not False:
        raise ValueError(
            "Failed replay result lacks repeatability evidence: " + entry["id"]
        )
    count = result.get("sample_count")
    if (
        type(count) is not int
        or count < 1
        or count != len(result.get("samples", []))
        or state.get("samples") != count
    ):
        raise ValueError("Replay sample count mismatch: " + entry["id"])
    unit = result.get("unit")
    if unit not in {"images", "pairs"} or type(limit) is not int or limit < 1:
        raise ValueError("Invalid replay sample unit or limit")
    if unit == "images" and count != limit:
        raise ValueError("Incomplete replay sample count: " + entry["id"])
    primary = result.get("primary_metric")
    allowed = {recipe["metric"]}
    if primary not in allowed:
        raise ValueError("Replay primary metric mismatch: " + entry["id"])
    values, units = result.get("metrics", {}), result.get("metric_units", {})
    if set(values) - set(units):
        raise ValueError("Missing replay metric units: " + entry["id"])
    # Only the count comes from the original measurement, never its metric values.
    expected_count = (
        6000 if unit == "pairs" else entry.get("measurement", {}).get("samples")
    )
    samples = limit if unit == "pairs" else count
    full = result.get("stages", {}).get("full_accuracy") == "executed"
    if full and (type(expected_count) is not int or samples != expected_count):
        raise ValueError("Full evaluation count mismatch: " + entry["id"])
    measured = metric_values(values, units)
    record.update(
        state=state["state"],
        metrics={} if diagnostic else measured,
        primary_metric=primary,
        metric_details=result.get("metric_details", {}),
        samples=samples,
        expected_samples=expected_count,
        sample_unit=unit,
        view_count=count if unit == "pairs" else None,
        complete=not diagnostic,
        full_evaluation=full and not diagnostic,
        failure_reason="repeat_bitwise_mismatch" if diagnostic else None,
        evaluation_profile=result["profile"],
        runtime=result["environment"],
        evaluation_identity=object_hash(
            [
                result["recipe_identity"],
                result["input_identity"],
                result["profile"],
                entry["graphs"].get("postprocess"),
                limit,
                unit,
                primary,
            ]
        ),
    )
    record["evidence"].update(
        result_hash_verified=True,
        result_sha256=sha256(path),
        identity=result["identity"],
        model_identity=object_hash(
            [prepared["selection_identity"], entry["model"], recipe.get("source")]
        ),
        preprocessing_identity=object_hash(
            [
                recipe["input"],
                recipe["preprocess"],
                recipe.get("network_input_map"),
                entry["graphs"].get("preprocess"),
            ]
        ),
        prepared_identity=row["identity"],
        source_graphs=result["source_graphs"],
        implementation=saved.get("implementation"),
    )
    if diagnostic:
        record["diagnostic_measurement"] = dict(
            metrics=measured,
            repeat_bitwise=False,
            original_accuracy=result.get("original_accuracy", {}),
        )
    return record


def report(
    *,
    run,
    package,
    output,
    formats=None,
    table_profile=None,
    models=None,
    conditions=None,
):
    formats = normalize_formats(formats)
    if table_profile is not None:
        validate_profile(table_profile)
    run, package = Path(run), Path(package)
    prepared = unseal(package / "manifest.json")
    saved = unseal(run / "run.json")
    if prepared.get("schema_version") != 1 or saved.get("schema_version", 1) != 1:
        raise ValueError("Unsupported replay report schema")
    entries = [r["entry"] for r in prepared["conditions"]]
    chosen = selected(entries, models, conditions)
    from .data import require_full_lfw_rows

    chosen_ids = {e["id"] for e in chosen}
    require_full_lfw_rows(
        [r for r in prepared["conditions"] if r["entry"]["id"] in chosen_ids],
        prepared["limit"],
    )
    prepared_rows = {r["entry"]["id"]: r for r in prepared["conditions"]}
    states = {r["id"]: r for r in saved["conditions"]}
    if len(prepared_rows) != len(entries) or len(states) != len(saved["conditions"]):
        raise ValueError("Duplicate replay condition")
    if set(states) - set(prepared_rows):
        raise ValueError("Run conditions are absent from the prepared package")
    run_id = "replay:" + sha256(run / "run.json")
    records = [
        _record(
            prepared_rows[e["id"]],
            states.get(e["id"], {}),
            prepared,
            saved,
            run,
            run_id,
        )
        for e in chosen
    ]
    source = dict(
        run_sha256=sha256(run / "run.json"),
        prepared_sha256=sha256(package / "manifest.json"),
        selection_identity=prepared["selection_identity"],
        verification="sealed evaluation metadata; no inference or external model reads",
    )
    export_records(
        records, output, metadata=source, formats=formats, table_profile=table_profile
    )
    return dict(
        output=str(output),
        **build_report(records, profile=table_profile or "opennpu_raw")["metadata"][
            "counts"
        ],
    )
