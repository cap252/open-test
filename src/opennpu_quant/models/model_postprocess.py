"""Adapters from host outputs to task predictions, with no metric accumulation."""

from functools import partial
import numpy as np
from . import decode, extended_decode


def _yolo(outputs, recipe, meta, categories):
    cfg = recipe["decode"]
    policy = dict(
        confidence_threshold=cfg["confidence_threshold"],
        multi_label=cfg["multi_label"],
        max_nms_candidates=cfg["max_nms_candidates"],
        nms_iou_threshold=cfg["nms"]["iou"],
        max_detections=cfg["max_detections"],
    )
    return decode.yolo(outputs[cfg["output"]], meta, categories, policy)


def _tf(outputs, recipe, meta, categories):
    return decode.tf_od_api(
        outputs, recipe["decode"], meta["original_size"], categories
    )


def _retinaface(outputs, recipe, meta, categories, *, priors):
    cfg = recipe["decode"]
    policy = dict(
        prior_box=dict(variance=cfg["variance"]),
        score_threshold=cfg["score_threshold"],
        nms_iou=cfg["nms_iou"],
        max_detections=cfg["max_detections"],
    )
    boxes, scores = decode.decode_retinaface(
        outputs["loc"][0],
        outputs["conf"][0],
        priors,
        policy,
        recipe["preprocess"]["size"],
        meta["original_size"],
    )
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("Nonfinite decoded face")
    return decode.wider_face_record(boxes, scores)


def _labels(outputs, recipe, meta, categories):
    return decode.segmentation_labels(
        outputs[recipe["decode"]["output"]], meta["valid_size"], meta["original_size"]
    )


def _resize(outputs, recipe, meta, categories):
    return extended_decode.segmentation(
        outputs[recipe["decode"]["output"]], meta, recipe["preprocess"]["kind"]
    )


def _extra(outputs, recipe, meta, categories):
    return extended_decode.yolo_extra(outputs, recipe["decode"], meta, categories)


def _pose(outputs, recipe, meta, categories):
    return extended_decode.pose(outputs, meta)


def _detection(outputs, recipe, meta, categories):
    return extended_decode.detection(outputs, recipe["decode"], meta, categories)


def for_recipe(recipe):
    kind = recipe["decode"]["kind"]
    if kind == "retinaface":
        cfg = recipe["decode"]
        w, h = recipe["preprocess"]["size"]
        priors = decode.retinaface_priors(
            h, w, cfg["priors"]["min_sizes"], cfg["priors"]["steps"]
        )
        return partial(_retinaface, priors=priors)
    return {
        "logits": None,
        "face_verification": None,
        "yolo": _yolo,
        "tf_od_api": _tf,
        "segmentation_labels": _labels,
        "segmentation_resize": _resize,
        "yolo_face": _extra,
        "yolo_seg": _extra,
        "posenet": _pose,
        "yolo_no_objectness": _detection,
        "retinanet": _detection,
        "nanodet": _detection,
        "yolov4": _detection,
    }[kind]
