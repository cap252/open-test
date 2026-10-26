"""Decoders of the extended model recipes, in original-image coordinates."""

import numpy as np
from .decode import class_nms, yolo


def restore_boxes(boxes, meta):
    boxes = boxes.copy()
    w, h = meta["original_size"]
    boxes[:, [0, 2]] = (
        (boxes[:, [0, 2]] - meta.get("pad", [0, 0])[0]) / meta["scale"][0]
    ).clip(0, w)
    boxes[:, [1, 3]] = (
        (boxes[:, [1, 3]] - meta.get("pad", [0, 0])[1]) / meta["scale"][1]
    ).clip(0, h)
    boxes[:, 2:] -= boxes[:, :2]
    return boxes


def detection(outputs, cfg, meta, categories):
    kind = cfg["kind"]
    if kind == "yolo_no_objectness":
        p = outputs[cfg["output"]].transpose(0, 2, 1)
        if p.shape[-1] != 84:
            raise ValueError("Expected 80 YOLO class scores without objectness")
        p = np.concatenate(
            (p[:, :, :4], np.ones_like(p[:, :, :1]), p[:, :, 4:]), axis=-1
        )
        policy = dict(
            confidence_threshold=cfg["confidence_threshold"],
            multi_label=cfg["multi_label"],
            max_nms_candidates=cfg["max_nms_candidates"],
            nms_iou_threshold=cfg["nms"]["iou"],
            max_detections=cfg["max_detections"],
        )
        return yolo(p, meta, categories, policy)
    if kind == "retinanet":
        boxes = restore_boxes(outputs["boxes"], meta)
        scores, labels = outputs["scores"], outputs["labels"]
        if not np.array_equal(labels, labels.astype(np.int64)) or np.any(
            (labels < 0) | (labels > 90)
        ):
            raise ValueError("Invalid RetinaNet class slot")
        # Original model already did TopK/NMS. Drop only the documented unused
        # COCO slots afterwards; do not refill the candidate list.
        keep = np.isin(labels, categories)
        return [
            dict(bbox=b.tolist(), score=float(s), category_id=int(c))
            for b, s, c in zip(boxes[keep], scores[keep], labels[keep])
            if b[2] > 0 and b[3] > 0
        ]
    if kind == "nanodet":
        p = outputs[cfg["output"]][0]
        scores = p[:, :80]
        reg = p[:, 80:].reshape(-1, 4, 8)
        exp = np.exp(reg - reg.max(axis=-1, keepdims=True))
        distance = (
            exp / exp.sum(axis=-1, keepdims=True) * np.arange(8, dtype=np.float32)
        ).sum(axis=-1)
        priors = []
        for stride in (8, 16, 32, 64):
            cells = int(np.ceil(320 / stride))
            yy, xx = np.meshgrid(np.arange(cells), np.arange(cells), indexing="ij")
            priors.append(
                np.stack(
                    (
                        xx.ravel() * stride,
                        yy.ravel() * stride,
                        np.full(xx.size, stride),
                    ),
                    axis=1,
                )
            )
        prior = np.concatenate(priors).astype(np.float32)
        distance *= prior[:, 2:]
        boxes = np.column_stack(
            (
                prior[:, 0] - distance[:, 0],
                prior[:, 1] - distance[:, 1],
                prior[:, 0] + distance[:, 2],
                prior[:, 1] + distance[:, 3],
            )
        ).clip(0, 320)
        threshold, limit = 0.05, 100
    elif kind == "yolov4":
        boxes = outputs["boxes"][0].reshape(-1, 4) * np.array([416, 416, 416, 416])
        scores = outputs["scores"][0]
        threshold, limit = 0.001, 300
    else:
        raise ValueError("Unknown detector: " + kind)
    candidates, labels = np.nonzero(scores > threshold)
    confidence = scores[candidates, labels]
    order = np.argsort(-confidence, kind="stable")[:30000]
    boxes, labels, confidence = (
        boxes[candidates[order]],
        labels[order],
        confidence[order],
    )
    keep = class_nms(boxes, confidence, labels, 0.6, limit)
    boxes = restore_boxes(boxes[keep], meta)
    return [
        dict(bbox=b.tolist(), category_id=categories[int(c)], score=float(s))
        for b, c, s in zip(boxes, labels[keep], confidence[keep])
        if b[2] > 0 and b[3] > 0
    ]


def yolo_extra(outputs, cfg, meta, categories):
    p = outputs[cfg["output"]][0]
    face = cfg["kind"] == "yolo_face"
    classes = 1 if face else 80
    required = 21 if face else 117
    if p.ndim != 2 or p.shape[1] != required:
        raise ValueError("Unexpected face/mask prediction shape")
    scores = p[:, 5 : 5 + classes] * p[:, 4:5]
    rows, labels = np.nonzero(scores > cfg["confidence_threshold"])
    confidence = scores[rows, labels]
    order = np.argsort(-confidence, kind="stable")[: cfg["max_nms_candidates"]]
    rows, labels, confidence = rows[order], labels[order], confidence[order]
    xywh = p[rows, :4]
    boxes = np.concatenate(
        (xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), axis=1
    )
    keep = class_nms(
        boxes, confidence, labels, cfg["nms"]["iou"], cfg["max_detections"]
    )
    rows, labels, confidence, boxes = (
        rows[keep],
        labels[keep],
        confidence[keep],
        boxes[keep],
    )
    restored = restore_boxes(boxes, meta)
    if face:
        result = []
        for b, score in zip(restored, confidence):
            x, y, x2, y2 = [
                int(float(v) + 0.5) for v in (b[0], b[1], b[0] + b[2], b[1] + b[3])
            ]
            result.append([x, y, x2 - x, y2 - y, float(f"{min(score, 1):.3f}")])
        return result
    import cv2
    import torch
    import torch.nn.functional as F
    from pycocotools import mask as mask_api

    proto = outputs["prototypes"][0]
    if proto.shape[0] != 32:
        raise ValueError("Expected 32 mask prototypes")
    result = []
    iw, ih = meta["input_size"]
    ow, oh = meta["original_size"]
    px, py = meta["pad"]
    for index, b, score, label, box in zip(rows, restored, confidence, labels, boxes):
        logits = p[index, 85:] @ proto.reshape(32, -1)
        probability = torch.from_numpy(logits.reshape(1, 1, *proto.shape[1:])).sigmoid()
        mask = F.interpolate(
            probability, size=(ih, iw), mode="bilinear", align_corners=False
        )[0, 0].numpy()
        xx = np.arange(iw)[None, :]
        yy = np.arange(ih)[:, None]
        mask = (
            (mask > 0.5)
            & (xx >= box[0])
            & (xx < box[2])
            & (yy >= box[1])
            & (yy < box[3])
        ).astype(np.uint8)
        right, bottom = int(iw - px), int(ih - py)
        mask = cv2.resize(
            mask[int(py) : bottom, int(px) : right],
            (ow, oh),
            interpolation=cv2.INTER_LINEAR,
        )
        rle = mask_api.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("ascii")
        if b[2] > 0 and b[3] > 0:
            result.append(
                dict(
                    bbox=b.tolist(),
                    category_id=categories[int(label)],
                    score=float(score),
                    segmentation=rle,
                )
            )
    return result


def segmentation(logits, meta, kind):
    import cv2
    import torch
    import torch.nn.functional as F

    if kind == "unet_letterbox":
        value = torch.from_numpy(logits[0]).permute(1, 2, 0).softmax(-1).numpy()
        left, top = meta["pad"]
        w, h = meta["valid_size"]
        value = value[top : top + h, left : left + w]
        return (
            cv2.resize(
                value, tuple(meta["original_size"]), interpolation=cv2.INTER_LINEAR
            )
            .argmax(-1)
            .astype(np.uint8)
        )
    value = F.interpolate(
        torch.from_numpy(logits),
        size=meta["input_size"][::-1],
        mode="bilinear",
        align_corners=False,
    )
    value = F.interpolate(
        value, size=meta["original_size"][::-1], mode="bilinear", align_corners=False
    )
    return value[0].argmax(0).numpy().astype(np.uint8)


def pose(outputs, meta):
    import torch
    from ._vendor.pose_multi import decode_multiple_poses

    values = [
        torch.from_numpy(outputs[k][0].copy())
        for k in ("heatmaps", "offsets", "displacements_fwd", "displacements_bwd")
    ]
    scores, parts, coords = decode_multiple_poses(
        *values, output_stride=16, max_pose_detections=10, min_pose_score=0.25
    )
    coords *= np.asarray(meta["output_scale_yx"])[None, None, :]
    return [
        dict(
            category_id=1,
            score=float(score),
            keypoints=np.column_stack((xy[:, 1], xy[:, 0], ps)).ravel().tolist(),
        )
        for score, ps, xy in zip(scores, parts, coords)
        if score >= 0.25
    ]
