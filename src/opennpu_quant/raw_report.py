"""OpenNPU workbook tables from verified evaluations, without numerical runtimes."""

from pathlib import Path

from ._io import atomic_json
from .results import (
    RAW_FIELDS,
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
    groups = {"full": [], "partial": []}
    metadata = []

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
        if "failure_details" in row:
            item["failure_details"] = row["failure_details"]
        if record.get("origin") not in {"standalone_run", "fixed_replay"}:
            item["reason"] = "not_this_execution"
        elif row["measurement_reason"]:
            item["reason"] = row["measurement_reason"]
        elif row["value_pct"] is None:
            item["reason"] = "missing_or_incompatible_metric"
        if item["reason"]:
            metadata.append(item)
            continue
        group = "full" if row["full_evaluation"] else "partial"
        reason = row["pairing_reason"]
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
    return _write_report(
        report, directory, source=source, selected=normalize_formats(formats)
    )


def _write_report(report, directory, *, source, selected, refresh_existing=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for group, name in (("full", "raw_data"), ("partial", "raw_data.partial")):
        if "csv" in selected or (
            refresh_existing and (directory / (name + ".csv")).is_file()
        ):
            _text(
                directory / (name + ".csv"),
                table_text(report[group], delimiter=","),
                "utf-8-sig",
            )
        if "tsv" in selected or (
            refresh_existing and (directory / (name + ".tsv")).is_file()
        ):
            _text(directory / (name + ".tsv"), table_text(report[group]), "utf-8")
    metadata = dict(report["metadata"], source=source or {})
    atomic_json(directory / "raw_data.metadata.json", metadata)
    return metadata["counts"]
