"""OpenNPU workbook tables from verified evaluations, without numerical runtimes."""

from pathlib import Path

from ._io import atomic_json
from .results import (
    RAW_FIELDS,
    SUCCESS,
    normalize_formats,
    raw_export_row,
    table_text,
    tables,
    _text,
)

TABLE_PROFILES = ("opennpu_raw",)
MODEL_NAMES = {
    "yolov3_full": "YOLOv3",
    "yolov4_full": "YOLOv4",
    "yolov7_full": "YOLOv7",
    "deeplabv3plus_r50_voc": "DeepLabV3+ R50-D8",
}
TASK_NAMES = {
    "classification": "image_classification",
    "detection": "object_detection",
    "face": "face_detection",
    "segmentation": "semantic_segmentation",
    "keypoints": "pose_estimation",
}
DATASET_NAMES = {"coco_keypoints": "coco", "lfw_mobilefacenet": "lfw"}


def validate_profile(profile):
    if profile not in TABLE_PROFILES:
        raise ValueError("Unknown table profile: " + str(profile))


def build_report(records, *, profile="opennpu_raw"):
    validate_profile(profile)
    rows, _ = tables(records)
    lookup = {(r["run_id"], r["model"], r["condition"]): r for r in records}
    groups = {"full": [], "partial": []}
    metadata = []

    def full(record):
        return bool(
            record.get("full_evaluation") is True
            and record.get("complete") is True
            and type(record.get("samples")) is int
            and type(record.get("expected_samples")) is int
            and record["samples"] > 0
            and record["samples"] == record.get("expected_samples")
        )

    for record, row in zip(records, rows):
        proof = record.get("evidence", {})
        item = {
            k: record.get(k)
            for k in (
                "run_id",
                "model",
                "condition",
                "state",
                "origin",
                "samples",
                "expected_samples",
                "sample_unit",
                "view_count",
                "full_evaluation",
                "evaluation_profile",
                "evaluation_identity",
                "quantization",
                "task",
                "dataset",
                "primary_metric",
            )
        }
        task = row.get("task")
        decode = record.get("model_metadata", {}).get("decode", {})
        item.update(
            task=task,
            sample_unit=record.get("sample_unit")
            or ("views" if task == "face_verification" else "images"),
            view_count=record.get(
                "view_count",
                record.get("samples") if task == "face_verification" else None,
            ),
            pair_count=record.get("metric_details", {}).get("pair_count"),
            evaluation_profile=record.get("evaluation_profile")
            or (decode.get("profile", "current_v1") if decode else None),
            evidence=proof,
            exported_to=None,
            reason=None,
        )
        if record.get("origin") not in {"standalone_run", "fixed_replay"} or record.get(
            "reference"
        ):
            item["reason"] = "not_this_execution"
        elif record["state"] not in SUCCESS:
            item["reason"] = record.get("failure_reason") or "state:" + record["state"]
        elif proof.get("result_hash_verified") is not True:
            item["reason"] = "unverified_result"
        elif row["value_pct"] is None:
            item["reason"] = "missing_or_incompatible_metric"
        if item["reason"]:
            metadata.append(item)
            continue
        group = "full" if full(record) else "partial"
        base = lookup.get((record["run_id"], record["model"], "fp32"))
        reason = row["pairing_reason"]
        if base and (
            full(base) != full(record)
            or base.get("expected_samples") != record.get("expected_samples")
            or base.get("sample_unit") != record.get("sample_unit")
            or base.get("view_count") != record.get("view_count")
            or base.get("evaluation_profile") != record.get("evaluation_profile")
            or base.get("evidence", {}).get("result_hash_verified") is not True
            or base.get("reference")
            or base.get("origin") != record.get("origin")
        ):
            row = dict(row, fp32_pct=None, recovery_pct=None)
            reason = "incompatible_baseline_scope_or_profile"
        elif row["fp32_pct"] == 0:
            reason = "zero_fp32_denominator"
        raw = raw_export_row(row)
        raw.update(
            model=MODEL_NAMES.get(record["model"], raw["model"]),
            task=TASK_NAMES.get(raw["task"], raw["task"]),
            dataset=DATASET_NAMES.get(raw["dataset"], raw["dataset"]),
        )
        groups[group].append(raw)
        item.update(
            exported_to="raw_data" + (".partial" if group == "partial" else ""),
            row=len(groups[group]) + 1,
            pairing_status="unmatched" if reason else "passed",
            pairing_reason=reason,
        )
        metadata.append(item)
    return dict(
        **groups,
        metadata=dict(
            schema_version=1,
            table_profile=profile,
            fields=RAW_FIELDS,
            origin="this_execution",
            records=metadata,
            counts=dict(
                full=len(groups["full"]),
                partial=len(groups["partial"]),
                excluded=sum(r["exported_to"] is None for r in metadata),
            ),
        ),
    )


def write_report(report, directory, *, source=None, formats=None):
    selected = normalize_formats(formats)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for group, name in (("full", "raw_data"), ("partial", "raw_data.partial")):
        if "csv" in selected:
            _text(
                directory / (name + ".csv"),
                table_text(report[group], delimiter=","),
                "utf-8-sig",
            )
        if "tsv" in selected:
            _text(directory / (name + ".tsv"), table_text(report[group]), "utf-8")
    metadata = dict(report["metadata"], source=source or {})
    atomic_json(directory / "raw_data.metadata.json", metadata)
    return metadata["counts"]
